"""
Scored retriever (RAG) over the deals dataset.

Uses TF-IDF + cosine similarity rather than a neural embedding model: it's a
legitimate "scored retriever" per the brief, runs fully offline with no
model download, and is fast enough that reranking adds no real latency.
To swap in sentence-transformer embeddings instead, replace the
TfidfVectorizer/doc_matrix in __init__ with a SentenceTransformer encode()
call — search()'s interface (query -> [(Deal, score)], weak: bool) doesn't
need to change.
"""
from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Deal, load_deals

DEFAULT_MIN_SCORE = 0.15  # below this, treat retrieval as unreliable -> abstain


def _deal_text(deal: Deal) -> str:
    parts = [deal.brand, deal.product, deal.category, deal.source_type, deal.description]
    if deal.card:
        parts.append(deal.card)
    return " ".join(p for p in parts if p)


class DealRetriever:
    def __init__(self, deals: list[Deal] | None = None):
        self.deals = deals or load_deals()
        self._texts = [_deal_text(d) for d in self.deals]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_matrix = self.vectorizer.fit_transform(self._texts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = DEFAULT_MIN_SCORE,
        brand: str | None = None,
        category: str | None = None,
        rerank: bool = True,
    ) -> tuple[list[tuple[Deal, float]], bool]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        candidates = list(zip(self.deals, sims))

        if rerank:
            candidates = [(d, self._rerank_score(query, d, s)) for d, s in candidates]

        if brand:
            candidates = [(d, s) for d, s in candidates if brand.lower() in d.brand.lower()]
        if category:
            candidates = [(d, s) for d, s in candidates if category.lower() == d.category.lower()]

        candidates.sort(key=lambda x: -x[1])
        # Only keep candidates that actually clear the relevance bar --
        # otherwise top_k padding lets irrelevant near-zero-score items
        # leak into downstream "cheapest of these results" logic (a real
        # bug caught in testing: an unrelated product won a price
        # comparison purely by being the padding with the lowest listed
        # price).
        results = [c for c in candidates if c[1] >= min_score][:top_k]
        weak = len(results) == 0
        return results, weak

    @staticmethod
    def _rerank_score(query: str, deal: Deal, base_score: float) -> float:
        """Lightweight lexical reranker on top of TF-IDF similarity: boosts
        exact brand/card token matches and product-word overlap, since those
        are strong signals TF-IDF under-weights when a short query is scored
        against longer description text."""
        q = query.lower()
        score = base_score
        if deal.brand.lower() in q:
            score += 0.25
        if deal.card and deal.card.lower() in q:
            score += 0.25
        product_tokens = set(deal.product.lower().split())
        query_tokens = set(q.split())
        score += 0.05 * len(product_tokens & query_tokens)
        return score


@lru_cache
def get_retriever() -> DealRetriever:
    return DealRetriever()
