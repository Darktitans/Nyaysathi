"""
Embedding module — Marathi Legal Judgment Retrieval System.

Primary embedding model (as documented in SLA Stage 1 Report, Section 4 —
Technologies Used): L3Cube MahaSBERT, a pretrained Marathi Sentence-BERT
model, used purely for inference (no training from scratch — Key Design
Decisions, SLA2 Section 3.3).

IMPORTANT — environment note:
This build/demo environment's network policy blocks outbound access to
huggingface.co, so the MahaSBERT weights cannot be downloaded here. Rather
than fake the model, this module is written as a pluggable interface:

    - MahaSBERTEmbedder   : the real, documented model. Works as soon as this
                             code runs anywhere with normal internet access
                             (e.g. the student's own laptop, Colab, or a
                             deployed server) — no code changes needed.
    - TfidfFallbackEmbedder: a TF-IDF + character n-gram cosine-similarity
                             embedder, used ONLY when MahaSBERT cannot be
                             loaded, so the system remains fully functional
                             end-to-end in network-restricted environments.
                             This is explicitly a fallback, not a
                             representation of the documented architecture.

get_embedder() auto-detects which backend is available and returns the
appropriate one. The active backend is reported in every API response's
metadata so this substitution is never silently hidden.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

MODEL_NAME = "l3cube-pune/marathi-sentence-bert-nli"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "model_cache"


class BaseEmbedder(ABC):
    name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of embeddings."""
        ...


class MahaSBERTEmbedder(BaseEmbedder):
    """Real documented embedding model: pretrained Marathi Sentence-BERT.
    Used for inference only, consistent with the project's design decision
    to not train any model from scratch."""
    name = "MahaSBERT (l3cube-pune/marathi-sentence-bert-nli)"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype("float32")


class TfidfFallbackEmbedder(BaseEmbedder):
    """Fallback used only when MahaSBERT cannot be downloaded (e.g. this
    sandbox's blocked network egress to huggingface.co). Produces fixed-size
    dense vectors via TF-IDF + truncated SVD so it plugs into the same FAISS
    cosine-similarity pipeline as the real embedder. This is a stand-in for
    demonstration purposes only, not the documented production model."""
    name = "TF-IDF + SVD (fallback — MahaSBERT unreachable in this environment)"

    def __init__(self, n_components: int = 256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        # char n-grams: robust to Marathi inflection without needing a
        # language-specific tokenizer
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False
        self.n_components = n_components

    def fit(self, corpus: list[str]):
        tfidf = self.vectorizer.fit_transform(corpus)
        eff_components = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if eff_components < self.n_components:
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=max(eff_components, 2), random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfFallbackEmbedder must be fit() on the corpus before encode()")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        return vecs.astype("float32")


def get_embedder(corpus_for_fallback_fit: list[str] | None = None) -> BaseEmbedder:
    """Attempt to load the real MahaSBERT model; fall back to TF-IDF if the
    model cannot be downloaded (e.g. blocked network egress). Returns the
    active embedder with .name describing which backend is live."""
    try:
        return MahaSBERTEmbedder()
    except Exception as e:
        print(f"[embedder] MahaSBERT unavailable ({e.__class__.__name__}: {e}); "
              f"falling back to TF-IDF embedder for this environment.")
        embedder = TfidfFallbackEmbedder()
        if corpus_for_fallback_fit:
            embedder.fit(corpus_for_fallback_fit)
        return embedder
