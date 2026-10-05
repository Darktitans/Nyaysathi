"""
FAISS indexing module — Offline Pipeline (SLA Stage 1 Report, Section 5.1 /
SLA Stage 2 Report, Section 3.1):

    Raw Marathi Judgment Dataset -> Text Cleaning + Unicode Normalization ->
    Sentence Tokenization -> Embedding Generation (MahaSBERT) ->
    FAISS Index Construction

Key Design Decision honoured here (SLA2 Section 3.3): "FAISS is used with an
exact search index (IndexFlatIP) given the project's dataset scale." IndexFlatIP
computes raw inner product, which only equals cosine similarity if every
vector is L2-normalized first — this module normalizes both at index-build
time and at query time so similarity scores are true cosine similarities.
"""
from __future__ import annotations
import numpy as np
import faiss
import pickle
from pathlib import Path

from .dataset import load_dataset
from .embedder import get_embedder
from .preprocessing import tokenize_sentences

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "judgments.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"
EMBEDDER_STATE_PATH = INDEX_DIR / "embedder_state.pkl"

# Anchor phrases used for lightweight operative-sentence extraction (SLA1
# Objective 5 / SLA2 NLP Concepts — Extractive Summarization). Rather than
# training a rhetorical-role classifier (infeasible at this dataset scale,
# per the Stage-1 literature review discussion of LegalSeg), sentences are
# ranked by lexical closeness to these "ruling" trigger phrases.
OPERATIVE_ANCHORS = [
    "निकाली काढण्यात येत आहे", "आदेश दिला", "मंजूर करण्यात येते",
    "फेटाळण्यात येत आहे", "याचिका निकाली", "अर्ज मंजूर", "जामीन मंजूर",
]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype("float32").copy()
    faiss.normalize_L2(mat)
    return mat


def extract_operative_sentence(case_text: str) -> str:
    """Pick the sentence most likely to state the operative order, using
    simple lexical overlap with anchor phrases. Falls back to the first
    sentence if no sentence scores above zero overlap."""
    sentences = tokenize_sentences(case_text)
    if not sentences:
        return case_text[:200]
    best, best_score = sentences[0], -1
    for s in sentences:
        score = sum(1 for anchor in OPERATIVE_ANCHORS if anchor in s)
        if score > best_score:
            best, best_score = s, score
    return best


def build_index():
    df = load_dataset()
    corpus = df["case_text"].tolist()

    embedder = get_embedder(corpus_for_fallback_fit=corpus)
    print(f"[indexer] Active embedding backend: {embedder.name}")

    vectors = embedder.encode(corpus)
    vectors = _l2_normalize(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    df = df.copy()
    df["operative_sentence"] = df["case_text"].apply(extract_operative_sentence)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(df, f)

    with open(EMBEDDER_STATE_PATH, "wb") as f:
        pickle.dump({"backend_name": embedder.name, "backend_type": type(embedder).__name__}, f)

    print(f"[indexer] Indexed {len(df)} judgments, dim={dim}")
    return embedder


if __name__ == "__main__":
    build_index()
