"""
Text preprocessing module — Marathi Legal Judgment Retrieval System.

Implements the preprocessing steps documented in SLA Stage 1 Report, Section 4
and SLA Stage 2 Report, Section 3.1 (Offline Pipeline):
    - Unicode normalization (NFC)
    - Basic cleaning (whitespace, stray artifacts)
    - Sentence tokenization (Indic NLP Library)

Deliberately does NOT remove stopwords before embedding (see Key Design
Decisions, SLA2 Section 3.3): contextual sentence embeddings rely on full
grammatical structure, including negation words, to preserve legal meaning.
"""
import re
import unicodedata

try:
    from indicnlp.tokenize import sentence_tokenize
    _INDIC_AVAILABLE = True
except ImportError:
    _INDIC_AVAILABLE = False

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Unicode-normalize and clean raw Marathi text (NFC normalization,
    URL/HTML removal, whitespace collapse). Keeps punctuation, numerals and
    section markers intact since they carry legal meaning."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_sentences(text: str, lang: str = "mr") -> list[str]:
    """Split Marathi text into sentences using the Indic NLP Library sentence
    tokenizer (handles Devanagari danda/double-danda in addition to standard
    punctuation). Falls back to a simple danda/period split if the library is
    unavailable."""
    text = clean_text(text)
    if not text:
        return []
    if _INDIC_AVAILABLE:
        try:
            return [s.strip() for s in sentence_tokenize.sentence_split(text, lang=lang) if s.strip()]
        except Exception:
            pass
    # fallback: split on danda (।, ॥) and standard sentence-ending punctuation
    parts = re.split(r"(?<=[।॥.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def preprocess_query(raw_query: str) -> str:
    """Preprocessing applied to a user's live query at search time — must be
    IDENTICAL to what is applied to judgments at indexing time, so query and
    judgment vectors live in the same embedding space."""
    return clean_text(raw_query)
