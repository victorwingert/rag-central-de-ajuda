from rag.corpus import Chunk
from rag.retrieval import (
    BM25Retriever,
    Hit,
    HybridRetriever,
    VectorRetriever,
    reciprocal_rank_fusion,
    top_hits,
)
from tests.conftest import make_chunk
from tests.fakes.fake_embedder import BagOfWordsEmbedder


def test_top_hits_orders_by_score_and_keeps_ties_stable(help_chunks: list[Chunk]):
    hits = top_hits(help_chunks, [0.5, 0.9, 0.5], k=3)
    assert [h.chunk.doc_id for h in hits] == ["cupons", "reembolso", "gorjeta"]


def test_rrf_rewards_chunks_found_by_both_lists():
    a, b, c = make_chunk("a", "x"), make_chunk("b", "y"), make_chunk("c", "z")
    first = [Hit(a, 9), Hit(b, 8)]
    second = [Hit(b, 0.9), Hit(c, 0.8)]
    fused = reciprocal_rank_fusion([first, second], k=3)
    assert [h.chunk.doc_id for h in fused] == ["b", "a", "c"]


def test_vector_retriever_uses_embedding_similarity(help_chunks: list[Chunk]):
    embedder = BagOfWordsEmbedder(["cupom", "cupons", "pix", "gorjeta"])
    hits = VectorRetriever(help_chunks, embedder).search("posso usar dois cupons?", 1)
    assert hits[0].chunk.doc_id == "cupons"


def test_hybrid_retriever_combines_both(help_chunks: list[Chunk]):
    embedder = BagOfWordsEmbedder(["entregador", "pix"])
    hybrid = HybridRetriever(
        BM25Retriever(help_chunks), VectorRetriever(help_chunks, embedder)
    )
    assert hybrid.search("gorjeta do entregador", 1)[0].chunk.doc_id == "gorjeta"
