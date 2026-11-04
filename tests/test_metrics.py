import pytest

from rag.metrics import ranked_doc_ids, recall_at_k, reciprocal_rank
from rag.retrieval import Hit
from tests.conftest import make_chunk


def test_ranked_doc_ids_counts_each_document_once():
    hits = [Hit(make_chunk("a", "x", 0), 1), Hit(make_chunk("a", "y", 1), 1)]
    assert ranked_doc_ids([*hits, Hit(make_chunk("b", "z"), 1)]) == ["a", "b"]


def test_recall_at_k_with_two_relevant_documents():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 2) == 0.5


def test_recall_requires_relevant_documents():
    with pytest.raises(ValueError, match="relevante"):
        recall_at_k(["a"], [], 1)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0
