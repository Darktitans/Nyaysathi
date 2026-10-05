"""
Category re-ranking module (SLA1 Section 5.4 / SLA2 Key Design Decisions):

"Category labels are used as an optional re-ranking signal rather than a
hard filter, so that a mis-predicted category cannot wrongly exclude a
relevant judgment."

A lightweight TF-IDF + Logistic Regression classifier predicts the likely
category of the user's query. If its predicted category matches a retrieved
judgment's category AND the classifier is reasonably confident, a small
score boost is applied. It never filters results outright — a wrong
prediction can only fail to help, never hide a correct match.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BOOST_WEIGHT = 0.03          # small, deliberately soft — never dominates semantic score
MIN_CONFIDENCE = 0.35        # below this, no boost is applied at all
MIN_CLASS_SAMPLES = 4        # categories with too few examples are excluded from training


class CategoryReranker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
        self.clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, texts: list[str], categories: list[str]):
        from collections import Counter
        counts = Counter(categories)
        keep_classes = {c for c, n in counts.items() if n >= MIN_CLASS_SAMPLES}
        if len(keep_classes) < 2:
            self._fitted = False
            return
        pairs = [(t, c) for t, c in zip(texts, categories) if c in keep_classes]
        texts_f, cats_f = zip(*pairs)
        X = self.vectorizer.fit_transform(texts_f)
        self.clf.fit(X, cats_f)
        self._fitted = True

    def predict(self, query: str) -> tuple[str | None, float]:
        if not self._fitted:
            return None, 0.0
        X = self.vectorizer.transform([query])
        proba = self.clf.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        return self.clf.classes_[idx], float(proba[idx])

    def boost(self, query: str, judgment_category: str) -> float:
        """Returns a small additive score boost (0 or BOOST_WEIGHT)."""
        pred_cat, conf = self.predict(query)
        if pred_cat is not None and conf >= MIN_CONFIDENCE and pred_cat == judgment_category:
            return BOOST_WEIGHT
        return 0.0
