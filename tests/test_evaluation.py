from rag.answering import NOT_FOUND_ANSWER
from rag.corpus import Chunk
from rag.evaluation import (
    GenerationSetup,
    evaluate_generation,
    evaluate_retrieval,
    summarize_generation,
)
from rag.golden import GoldenQuestion
from rag.llm import Message
from rag.retrieval import BM25Retriever
from tests.fakes.fake_llm import ScriptedLanguageModel, constant

VERDICT_OK = '{"correta": true, "fiel": true, "motivo": "ok"}'


def question(qid: str, text: str, docs: list[str]) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        pergunta=text,
        docs_relevantes=docs,
        resposta_esperada="-",
        no_escopo=bool(docs),
    )


def refuse_out_of_scope(messages: list[Message]) -> str:
    if "bitcoin" in messages[-1]["content"]:
        return NOT_FOUND_ANSWER
    return "Até 1 dia útil [reembolso]."


def test_retrieval_eval_reports_recall_and_misses(help_chunks: list[Chunk]):
    questions = [
        question("q1", "reembolso pix", ["reembolso"]),
        question("q2", "gorjeta entregador", ["cupons"]),
        question("oos", "bitcoin", []),
    ]
    score = evaluate_retrieval(BM25Retriever(help_chunks), questions)
    assert score.recall_at_1 == 0.5
    assert score.misses_top3 == []


def test_generation_eval_grades_answers_and_refusals(help_chunks: list[Chunk]):
    setup = GenerationSetup(
        retriever=BM25Retriever(help_chunks),
        answerer=ScriptedLanguageModel(refuse_out_of_scope),
        judge=constant(VERDICT_OK),
    )
    questions = [question("q1", "reembolso pix", ["reembolso"])]
    questions.append(question("oos", "aceita bitcoin?", []))
    records = evaluate_generation(setup, questions)
    summary = summarize_generation(records)
    assert summary["corretas (no escopo)"] == 1.0
    assert summary["recusa correta (fora do escopo)"] == 1.0
    assert records[0].cited_doc_ids == ["reembolso"]


def test_refusing_an_in_scope_question_counts_as_wrong(help_chunks: list[Chunk]):
    judge = constant(VERDICT_OK)
    setup = GenerationSetup(
        BM25Retriever(help_chunks), constant(NOT_FOUND_ANSWER), judge
    )
    [record] = evaluate_generation(setup, [question("q1", "pix", ["reembolso"])])
    assert (record.correct, record.refused) == (False, True)
    assert judge.received == []
