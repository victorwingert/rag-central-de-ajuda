"""BM25 (Okapi) escrito à mão, para deixar a fórmula visível."""

import math
from collections import Counter

from rag.text import tokenize


class BM25Index:
    """Índice de palavras-chave sobre uma lista de textos.

    Exemplo: `BM25Index(["taxa de entrega", "reembolso"]).scores("taxa")`.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._term_counts = [Counter(tokenize(t)) for t in texts]
        self._lengths = [sum(c.values()) for c in self._term_counts]
        self._avg_length = sum(self._lengths) / max(len(texts), 1)
        self._idf = _inverse_document_frequency(self._term_counts)

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        return [
            sum(self._term_score(term, i) for term in terms)
            for i in range(len(self._term_counts))
        ]

    def _term_score(self, term: str, doc_index: int) -> float:
        tf = self._term_counts[doc_index].get(term, 0)
        if tf == 0:
            return 0.0
        length_ratio = self._lengths[doc_index] / self._avg_length
        norm = tf + self._k1 * (1 - self._b + self._b * length_ratio)
        return self._idf[term] * tf * (self._k1 + 1) / norm


def _inverse_document_frequency(term_counts: list[Counter[str]]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for counts in term_counts:
        doc_freq.update(counts.keys())
    n = len(term_counts)
    return {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in doc_freq.items()}
