"""
FastAPI backend — Online Query Pipeline (SLA1 Section 5.2 / SLA2 Section 3.2):

    User Enters Marathi Case Description -> Text Cleaning + Normalization
    (same as offline) -> Query Embedding (same model) -> FAISS Similarity
    Search (Top-K) -> Metadata Lookup + Operative-Sentence Extraction ->
    Results Displayed in Marathi

Robustness features:
  * Index/embedder consistency check: if the saved FAISS index was built with a
    different embedder (e.g. TF-IDF 256-dim vs MahaSBERT 768-dim), the index is
    rebuilt automatically instead of crashing at search time.
  * Relevance cutoff: judgments whose raw cosine similarity is below
    MIN_COSINE are never shown (the UI then shows "no matching judgments").
  * Optional strict mode: set env NYAYSATHI_STRICT=1 to refuse to run on the
    TF-IDF fallback (so weak results can never appear silently).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .embedder import TfidfFallbackEmbedder, get_embedder
from .indexer import (
    EMBEDDER_STATE_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    _l2_normalize,
    build_index,
)
from .preprocessing import preprocess_query
from .reranker import CategoryReranker

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Raw cosine similarity below this => judgment is treated as irrelevant and is
# not displayed. Tune: lower (0.25) if real queries return nothing, higher
# (0.35) if junk queries still return results.
MIN_COSINE = 0.30

MAX_TOP_K = 20
STRICT_MODE = os.getenv("NYAYSATHI_STRICT", "0") == "1"

app = FastAPI(title="न्यायसाथी — Marathi Legal Judgment Retrieval System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {}


# --------------------------------------------------------------------------
# Startup helpers
# --------------------------------------------------------------------------
def _index_files_exist() -> bool:
    return FAISS_INDEX_PATH.exists() and METADATA_PATH.exists() and EMBEDDER_STATE_PATH.exists()


def _load_saved_artifacts():
    """Load the FAISS index, metadata DataFrame and embedder-state from disk."""
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        df = pickle.load(f)
    with open(EMBEDDER_STATE_PATH, "rb") as f:
        embedder_state = pickle.load(f)
    return index, df, embedder_state


def _artifacts_match_embedder(index, embedder, embedder_state) -> bool:
    """True only if the saved index was built with the same kind of embedder
    (and the same vector size) as the one that is live right now."""
    if embedder_state.get("backend_type") != type(embedder).__name__:
        return False
    live_dim = embedder.encode(["परीक्षण"]).shape[1]
    return live_dim == index.d


def _safe_str(value) -> str:
    """Convert any cell (including NaN/None) to a clean string."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


@app.on_event("startup")
def startup():
    # 1) First run: build the index from the dataset.
    if not _index_files_exist():
        print("[startup] No saved index found — building one now...")
        build_index()

    # 2) Load saved artifacts (rebuild once if they are corrupt/unreadable).
    try:
        index, df, embedder_state = _load_saved_artifacts()
    except Exception as e:
        print(f"[startup] Saved index unreadable ({e.__class__.__name__}: {e}) — rebuilding...")
        build_index()
        index, df, embedder_state = _load_saved_artifacts()

    # 3) Load the live embedder, fitting the TF-IDF fallback on the corpus if needed.
    corpus = df["case_text"].tolist()
    embedder = get_embedder(corpus_for_fallback_fit=corpus)

    if STRICT_MODE and isinstance(embedder, TfidfFallbackEmbedder):
        raise RuntimeError(
            "NYAYSATHI_STRICT=1 but MahaSBERT could not be loaded. "
            "Install sentence-transformers (and torch) and check your internet connection."
        )

    # 4) Make sure index and embedder agree; otherwise rebuild automatically.
    if not _artifacts_match_embedder(index, embedder, embedder_state):
        print("[startup] Saved index was built with a different embedder — rebuilding to match...")
        build_index()
        index, df, embedder_state = _load_saved_artifacts()
        corpus = df["case_text"].tolist()
        embedder = get_embedder(corpus_for_fallback_fit=corpus)

    # 5) Category re-ranker (soft signal only).
    reranker = CategoryReranker()
    reranker.fit(corpus, df["category"].tolist())

    _state.update(
        index=index,
        df=df,
        embedder=embedder,
        embedder_state=embedder_state,
        reranker=reranker,
    )

    print(
        f"[startup] Ready. {len(df)} judgments indexed (dim={index.d}). "
        f"Embedding backend: {embedder_state['backend_name']} | MIN_COSINE={MIN_COSINE}"
    )


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category_filter: str | None = None


class SearchResult(BaseModel):
    judgment_id: str
    case_name: str
    court_name: str
    year: str
    category: str
    similarity_score: float
    final_decision: str
    legal_principle: str
    operative_sentence: str
    case_text: str
    source_pdf_url: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    engine_info: dict


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/api/status")
def status():
    return {
        "judgments_indexed": len(_state["df"]),
        "embedding_backend": _state["embedder_state"]["backend_name"],
        "categories": sorted(_state["df"]["category"].unique().tolist()),
        "min_cosine": MIN_COSINE,
    }


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    df = _state["df"]
    index = _state["index"]
    embedder = _state["embedder"]
    reranker = _state["reranker"]
    engine_info = {"embedding_backend": _state["embedder_state"]["backend_name"]}

    cleaned_query = preprocess_query(req.query)
    if not cleaned_query:
        return SearchResponse(query=req.query, results=[], engine_info=engine_info)

    top_k = max(1, min(req.top_k, MAX_TOP_K))
    use_category_filter = bool(req.category_filter) and req.category_filter != "सर्व श्रेणी"

    qv = _l2_normalize(embedder.encode([cleaned_query]))

    # With a category filter, search the whole (small) index so the filter can
    # never leave us with too few candidates; otherwise over-fetch a little.
    fetch_k = index.ntotal if use_category_filter else min(max(top_k * 4, 20), index.ntotal)
    scores, indices = index.search(qv, fetch_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        cosine = float(score)

        # Relevance cutoff on the RAW cosine similarity (before any boost).
        if cosine < MIN_COSINE:
            continue

        row = df.iloc[idx]
        if use_category_filter and row["category"] != req.category_filter:
            continue

        boosted_score = cosine + reranker.boost(cleaned_query, row["category"])
        candidates.append((boosted_score, row))

    candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, row in candidates[:top_k]:
        similarity_pct = max(0.0, min(1.0, (score + 1) / 2)) * 100  # cosine [-1,1] -> [0,100]%
        results.append(
            SearchResult(
                judgment_id=_safe_str(row["judgment_id"]),
                case_name=_safe_str(row["case_name"]),
                court_name=_safe_str(row["court_name"]),
                year=_safe_str(row["year"]),
                category=_safe_str(row["category"]),
                similarity_score=round(similarity_pct, 1),
                final_decision=_safe_str(row["final_decision"]),
                legal_principle=_safe_str(row["legal_principle"]),
                operative_sentence=_safe_str(row["operative_sentence"]),
                case_text=_safe_str(row["case_text"]),
                source_pdf_url=_safe_str(row["source_pdf_url"]),
            )
        )

    return SearchResponse(query=req.query, results=results, engine_info=engine_info)


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_landing():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/app")
def serve_app():
    return FileResponse(str(FRONTEND_DIR / "app.html"))