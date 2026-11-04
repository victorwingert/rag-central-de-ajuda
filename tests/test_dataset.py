"""Integridade do conjunto de avaliação versionado em data/."""

from pathlib import Path

from rag.corpus import load_documents
from rag.golden import load_golden

DATA = Path(__file__).resolve().parents[1] / "data"


def test_golden_points_only_to_existing_documents():
    doc_ids = {d.id for d in load_documents(DATA / "docs")}
    questions = load_golden(DATA / "golden.jsonl")
    referenced = {doc for q in questions for doc in q.docs_relevantes}
    assert referenced <= doc_ids


def test_golden_ids_are_unique_and_scope_matches_documents():
    questions = load_golden(DATA / "golden.jsonl")
    assert len({q.id for q in questions}) == len(questions)
    assert all(q.no_escopo == bool(q.docs_relevantes) for q in questions)
