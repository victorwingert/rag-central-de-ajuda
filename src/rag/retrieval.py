"""Três estratégias de busca: palavras-chave (BM25), vetorial e híbrida (RRF)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag.bm25 import BM25Index
from rag.corpus import Chunk
from rag.embeddings import Embedder

RRF_K = 60  # constante usual do RRF (Cormack et al., 2009)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


class BM25Retriever:
    """Busca por palavras-chave. Exemplo: `BM25Retriever(chunks).search("pix", 3)`."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._index = BM25Index([c.text for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        return top_hits(self._chunks, self._index.scores(query), k)


class VectorRetriever:
    """Busca semântica por cosseno.

    Exemplo: `VectorRetriever(chunks, embedder).search("reembolso", 3)`.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self._chunks = chunks
        self._embedder = embedder
        self._matrix = embedder.embed([c.text for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        query_vector = self._embedder.embed([query])[0]
        similarities = (self._matrix @ query_vector).tolist()
        return top_hits(self._chunks, similarities, k)


class HybridRetriever:
    """Junta BM25 e vetorial por RRF, sem precisar calibrar a escala das notas.

    Exemplo: `HybridRetriever(BM25Retriever(c), VectorRetriever(c, emb)).search(q, 3)`.
    """

    def __init__(self, *retrievers: Retriever, pool: int = 20) -> None:
        self._retrievers = retrievers
        self._pool = pool

    def search(self, query: str, k: int) -> list[Hit]:
        rankings = [r.search(query, self._pool) for r in self._retrievers]
        return reciprocal_rank_fusion(rankings, k)


def top_hits(chunks: list[Chunk], scores: Sequence[float], k: int) -> list[Hit]:
    """Os `k` trechos de maior nota; empate mantém a ordem original."""
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return [Hit(chunks[int(i)], float(scores[int(i)])) for i in order]


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int) -> list[Hit]:
    """Soma 1/(RRF_K + posição) de cada trecho em cada lista.

    Exemplo: `reciprocal_rank_fusion([bm25_hits, vector_hits], k=3)`.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1 / (RRF_K + rank)
            chunks[hit.chunk.id] = hit.chunk
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [Hit(chunks[chunk_id], score) for chunk_id, score in ordered]
