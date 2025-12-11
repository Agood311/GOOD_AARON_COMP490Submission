# main.py
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("rfp.csv")
TEMPLATES_DIR = Path("templates")

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Globals
df: pd.DataFrame | None = None
texts: List[str] | None = None
tfidf_vectorizer: TfidfVectorizer | None = None
tfidf_matrix = None
embed_model: SentenceTransformer | None = None
embeddings: np.ndarray | None = None


def init_indexes() -> None:
    """
    Load rfp.csv, clean, and build:
      - combined_text
      - TF-IDF matrix
      - semantic embeddings
    """
    global df, texts, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings

    if not DATA_FILE.exists():
        raise RuntimeError("rfp.csv not found. Run ingest_sam.py first.")

    df_loaded = pd.read_csv(DATA_FILE, dtype=str).fillna("")

    if "id" not in df_loaded.columns:
        raise RuntimeError("rfp.csv must contain an 'id' column")

    # Deduplicate by id
    df_loaded = df_loaded.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Ensure columns exist
    for col in [
        "title",
        "description_text",
        "organization_name",
        "full_parent_path_name",
        "response_date",
        "ui_link",
        "source_url",
        "additional_info_link",
        "naics",
        "psc",
        "state",
        "place_of_performance",
    ]:
        if col not in df_loaded.columns:
            df_loaded[col] = ""

    # Build combined text
    combined_texts: list[str] = []
    for _, row in df_loaded.iterrows():
        pieces = []

        title = row.get("title", "").strip()
        if title:
            pieces.append(title)

        org = row.get("organization_name", "").strip()
        if org:
            pieces.append(org)

        parent = row.get("full_parent_path_name", "").strip()
        if parent:
            pieces.append(parent)

        naics = row.get("naics", "").strip()
        if naics:
            pieces.append(f"NAICS {naics}")

        psc = row.get("psc", "").strip()
        if psc:
            pieces.append(f"PSC {psc}")

        desc = row.get("description_text", "").strip()
        if desc:
            pieces.append(desc)

        combined_texts.append(" ".join(pieces))

    df_loaded["combined_text"] = combined_texts

    # Drop rows with no text
    mask = df_loaded["combined_text"].str.len() > 0
    df_loaded = df_loaded[mask].reset_index(drop=True)

    if df_loaded.empty:
        raise RuntimeError("No usable rows with text found in rfp.csv")

    combined = df_loaded["combined_text"].tolist()

    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(combined)

    # Build semantic embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(
        combined,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    df = df_loaded
    texts = combined
    tfidf_vectorizer = vectorizer
    tfidf_matrix = X
    embed_model = model
    embeddings = emb

    globals()["df"] = df
    globals()["texts"] = texts
    globals()["tfidf_vectorizer"] = tfidf_vectorizer
    globals()["tfidf_matrix"] = tfidf_matrix
    globals()["embed_model"] = embed_model
    globals()["embeddings"] = embeddings

    print(f"Initialized indexes on {len(df)} RFPs")


def run_tfidf(query: str, top_k: int = 50) -> pd.DataFrame:
    q_vec = tfidf_vectorizer.transform([query])
    sims = linear_kernel(q_vec, tfidf_matrix).ravel()
    out = df.copy()
    out["score"] = sims
    out = out.sort_values("score", ascending=False)
    out = out[out["score"] > 0]
    return out.head(top_k)


def run_semantic(query: str, top_k: int = 50) -> pd.DataFrame:
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_vec  # cosine similarity because normalized
    out = df.copy()
    out["score"] = sims
    out = out.sort_values("score", ascending=False)
    out = out[out["score"] > 0]
    return out.head(top_k)


def format_results(df_res: pd.DataFrame) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for _, row in df_res.iterrows():
        title = (row.get("title") or "").strip() or "(no title)"

        agency = (row.get("organization_name") or "").strip()
        if not agency:
            agency = (row.get("full_parent_path_name") or "").strip()

        deadline = (row.get("response_date") or "").strip()

        url = ""
        for col in ("source_url", "ui_link", "additional_info_link"):
            val = row.get(col)
            if isinstance(val, str) and val.startswith("http"):
                url = val
                break

        text = (
            row.get("description_text")
            or row.get("combined_text")
            or ""
        )
        text = str(text).replace("\n", " ")
        snippet = text[:400]

        score_val = float(row.get("score", 0.0) or 0.0)

        results.append(
            {
                "id": row.get("id"),
                "title": title,
                "agency": agency,
                "deadline": deadline,
                "url": url,
                "snippet": snippet,
                "score": f"{score_val:.4f}",
            }
        )
    return results


@app.on_event("startup")
def on_startup():
    init_indexes()


@app.get("/", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: str = "",
    mode: str = "semantic",
    page: int = 1,
    per_page: int = 10,
):
    q = (q or "").strip()
    mode = (mode or "semantic").lower()
    page = max(page, 1)
    per_page = max(per_page, 1)

    results: List[Dict[str, Any]] = []
    total = 0
    total_pages = 0

    if q:
        if mode == "tfidf":
            df_res = run_tfidf(q, top_k=50)
        else:
            mode = "semantic"
            df_res = run_semantic(q, top_k=50)

        all_results = format_results(df_res)
        total = len(all_results)
        total_pages = (total + per_page - 1) // per_page

        start = (page - 1) * per_page
        end = start + per_page
        results = all_results[start:end]

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": q,
            "mode": mode,
            "results": results,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
    )


@app.post("/refresh-local", response_class=HTMLResponse)
async def refresh_local():
    """
    Reload rfp.csv and rebuild TF-IDF + semantic indexes.
    Run ingest_sam.py again before pressing this if you want fresh data.
    """
    init_indexes()
    return RedirectResponse("/", status_code=303)
