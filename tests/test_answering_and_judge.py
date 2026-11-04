import pytest

from rag.answering import (
    NOT_FOUND_ANSWER,
    answer_question,
    build_messages,
    parse_citations,
)
from rag.judge import parse_verdict
from rag.retrieval import Hit
from tests.conftest import make_chunk
from tests.fakes.fake_llm import constant


def test_prompt_has_context_with_doc_ids_and_refusal_rule():
    hits = [Hit(make_chunk("cliente-reembolso", "Pix em 1 dia."), 1)]
    system, user = build_messages("Quando cai o Pix?", hits)
    assert NOT_FOUND_ANSWER in system["content"]
    assert "[cliente-reembolso] Pix em 1 dia." in user["content"]


def test_parse_citations_deduplicates_in_order():
    text = "Até 1 dia [cliente-reembolso]. Sem custo [cupons] [cliente-reembolso]."
    assert parse_citations(text) == ["cliente-reembolso", "cupons"]


def test_answer_detects_refusal():
    answer = answer_question("Aceita bitcoin?", [], constant(NOT_FOUND_ANSWER))
    assert answer.refused
    assert answer.cited_doc_ids == []


def test_parse_verdict_accepts_json_inside_code_block():
    raw = '```json\n{"correta": true, "fiel": false, "motivo": "inventou"}\n```'
    verdict = parse_verdict(raw)
    assert (verdict.correta, verdict.fiel) == (True, False)


@pytest.mark.parametrize("raw", ["sem json", '{"correta": "talvez"}'])
def test_parse_verdict_rejects_bad_output(raw: str):
    with pytest.raises(ValueError, match="juiz"):
        parse_verdict(raw)
