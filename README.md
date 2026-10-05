# Nyaysathi
Semantic search for Marathi legal judgments. Type a case description in Marathi and get relevant Bombay High Court precedents, ranked by meaning using MahaSBERT, FAISS and FastAPI. A legal research assistant, not a decision-maker.

# न्यायसाथी (Nyay-Sathi): Marathi Legal Judgment Retrieval System

A semantic search tool for Marathi legal judgments. A user types a case description in Marathi, and the system retrieves the most relevant earlier judgments by **meaning**, not by exact keywords. It is a legal **research assistant**: it surfaces precedents to read and does not predict or influence case outcomes.

> NLP micro-project, Department of Computer Engineering, Shah & Anchor Kutchhi Engineering College (TYCM3).

---

## Features

- Marathi input and Marathi output (the whole interface is in Marathi)
- Semantic search with **MahaSBERT** (`l3cube-pune/marathi-sentence-bert-nli`), 768-dimensional embeddings
- Exact cosine-similarity search with **FAISS** (`IndexFlatIP` over L2-normalised vectors)
- Relevance cutoff: judgments below a minimum similarity are hidden, so unrelated queries show "no matching judgments"
- Soft category re-ranking (TF-IDF + Logistic Regression) plus an optional category filter
- Result cards show case name, court, year, category, similarity score, final decision, legal principle, operative sentence, full text and a link to the original PDF
- Automatic index rebuild if the saved index does not match the loaded embedding model
- Optional strict mode that refuses to run on the weak TF-IDF fallback

## How it works

**Offline (run once):**
raw Marathi dataset → cleaning and Unicode (NFC) normalization → MahaSBERT embeddings → L2 normalization → FAISS index, saved with the metadata.

**Online (every search):**
Marathi query → same cleaning → query embedding → FAISS cosine search → relevance cutoff (raw cosine ≥ 0.30) → category boost (+0.03) → Top-K results shown in Marathi.

Design decisions:
- Stopwords are **not** removed, because contextual embeddings need full grammar and words such as "नाही" can flip a legal meaning.
- No model is trained from scratch. Only the small category classifier is trained, on the project's own labels.
- Category is a soft signal by default, so a wrong category cannot hide a relevant judgment.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Embeddings | sentence-transformers (PyTorch), MahaSBERT |
| Vector search | FAISS (`faiss-cpu`) |
| Preprocessing | Unicode NFC, regex, Indic NLP Library |
| Re-ranking | scikit-learn |
| Data | Pandas, NumPy |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |

## Project structure

```
nyaysathi/
├── backend/
│   ├── main.py           # FastAPI app, search API, startup checks
│   ├── preprocessing.py  # cleaning, normalization, sentence tokenization
│   ├── dataset.py        # CSV loading and fallback chains
│   ├── embedder.py       # MahaSBERT (primary) and TF-IDF fallback
│   ├── indexer.py        # index build, operative-sentence extraction
│   └── reranker.py       # category prediction and boost
├── frontend/
│   ├── index.html        # landing page
│   ├── app.html          # search and results page
│   ├── app.js
│   └── style.css
├── data/
│   ├── reviewed_final.csv  # the judgment dataset (271 records)
│   ├── index/              # generated: FAISS index and metadata
│   └── model_cache/        # generated: downloaded MahaSBERT files
└── requirements.txt
```

## Setup and run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install PyTorch and sentence-transformers (needed for MahaSBERT)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers

# 3. Start the server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open:
- Landing page: http://127.0.0.1:8000/
- Search page: http://127.0.0.1:8000/app

Notes:
- On the **first run** the MahaSBERT weights are downloaded (a few hundred MB) and the index is built. This takes a few minutes. Later starts are faster.
- Open the pages through the server address above. Opening `index.html` directly from disk will not work.
- If you change the dataset, delete the `data/index/` folder so the index rebuilds.
- **Offline start (Windows):**
  ```
  set HF_HUB_OFFLINE=1
  set TRANSFORMERS_OFFLINE=1
  ```
- **Strict mode:** set `NYAYSATHI_STRICT=1` to stop the server with an error instead of falling back to TF-IDF.

## Example queries

| Query | Expected results |
|---|---|
| `आरोपीला जामीन मिळावा यासाठी अर्ज केला.` | Criminal bail cases |
| `शिक्षकाला रिक्त अनुदानित पदावर सेवा वरिष्ठतेनुसार बदली मिळावी.` | Service and teacher cases |
| `आयकर कायद्याच्या कलम १४८ अंतर्गत बजावलेल्या नोटीसला याचिकाकर्त्याने आव्हान दिले.` | Income-tax notice case |
| `जमीन संपादनाच्या भरपाईबाबत शेतकऱ्याने अपील केले.` | Land acquisition cases |
| `आजचे हवामान कसे आहे` | No matching judgments |

## Dataset

- 271 judgments and orders of the **Bombay High Court** (2022–2026).
- Sourced from the Indian High Court Judgments open data collection (CC-BY-4.0, Dattam Labs) via the KanoonGPT *Indian Case Laws* dataset.
- Each record has about 550 characters of the operative order, machine-translated into Marathi, with a decision, legal principle and source PDF link.

Category counts: फौजदारी 132, सेवा आणि शिक्षण 62, मालमत्ता 36, कर 15, कामगार 11, कौटुंबिक 9, पर्यावरणीय 3, संवैधानिक 2, ग्राहक 1.

## Limitations

- Small corpus. The family, environmental, constitutional and consumer categories have too few cases for reliable results.
- Categories are automatic or assistant-suggested and not fully manually verified.
- The text is machine-translated and still contains English words and names.
- Each record covers only the operative order, not the full facts.
- Close sub-types (for example anticipatory versus regular bail) can be confused.
- The percentage shown on each card is `(cosine + 1) / 2`, a relative ranking score and not a true accuracy percentage.
- No formal evaluation (Precision@K, Recall@K, MRR) has been run yet.

## Future work

- Build a relevance set and report Precision@K, Recall@K and MRR
- Compare MahaSBERT with `paraphrase-multilingual-mpnet-base-v2`
- Clean and balance the dataset
- Embedding-based operative-sentence extraction
- Add district-court and consumer-forum judgments

## Authors

- Dhruv Darji (TYCM3-18)
- Atharva Gaikwad (TYCM3-25)

## Acknowledgements and credits

- [L3Cube Pune](https://huggingface.co/l3cube-pune/marathi-sentence-bert-nli) for MahaSBERT
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI
- Dattam Labs, the Indian High Court Judgments dataset (CC-BY-4.0)
- KanoonGPT, *Indian Case Laws* on Hugging Face

## Disclaimer

This is an academic project and a research aid. It does not give legal advice, and its results should not be used to decide any case.
