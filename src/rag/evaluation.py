"""Avalia a busca e a geração separadamente, para saber onde está cada erro."""

import statistics
from dataclasses import dataclass

from rag.answering import Answer, answer_question, format_context
from rag.golden import GoldenQuestion
from rag.judge import judge_answer
from rag.llm import LanguageModel
from rag.metrics import ranked_doc_ids, recall_at_k, reciprocal_rank
from rag.retrieval import Retriever

SEARCH_DEPTH = 20


@dataclass(frozen=True)
class RetrievalScore:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    misses_top3: list[str]


def evaluate_retrieval(
    retriever: Retriever, questions: list[GoldenQuestion]
) -> RetrievalScore:
    """Só as perguntas no escopo entram; fora do escopo não tem documento certo.

    Exemplo: `evaluate_retrieval(HybridRetriever(bm25, vector), questions)`.
    """
    ranked = {
        q.id: ranked_doc_ids(retriever.search(q.pergunta, SEARCH_DEPTH))
        for q in questions
        if q.no_escopo
    }
    golden = {q.id: q.docs_relevantes for q in questions if q.no_escopo}

    def mean_recall(k: int) -> float:
        return statistics.fmean(recall_at_k(ranked[i], golden[i], k) for i in ranked)

    misses = [i for i in ranked if recall_at_k(ranked[i], golden[i], 3) == 0]
    mrr = statistics.fmean(reciprocal_rank(ranked[i], golden[i]) for i in ranked)
    return RetrievalScore(mean_recall(1), mean_recall(3), mean_recall(5), mrr, misses)


@dataclass(frozen=True)
class GenerationRecord:
    question_id: str
    in_scope: bool
    answer: str
    cited_doc_ids: list[str]
    refused: bool
    correct: bool
    faithful: bool
    reason: str
    latency_s: float
    cost_usd: float


@dataclass(frozen=True)
class GenerationSetup:
    retriever: Retriever
    answerer: LanguageModel
    judge: LanguageModel
    k: int = 3


def evaluate_generation(
    setup: GenerationSetup, questions: list[GoldenQuestion]
) -> list[GenerationRecord]:
    """Exemplo: `evaluate_generation(GenerationSetup(r, modelo, juiz), questions)`."""
    return [_evaluate_one(setup, q) for q in questions]


def _evaluate_one(setup: GenerationSetup, question: GoldenQuestion) -> GenerationRecord:
    hits = setup.retriever.search(question.pergunta, setup.k)
    answer = answer_question(question.pergunta, hits, setup.answerer)
    correct, faithful, reason = _grade(setup.judge, question, answer)
    return GenerationRecord(
        question_id=question.id,
        in_scope=question.no_escopo,
        answer=answer.text,
        cited_doc_ids=answer.cited_doc_ids,
        refused=answer.refused,
        correct=correct,
        faithful=faithful,
        reason=reason,
        latency_s=answer.completion.latency_s,
        cost_usd=answer.completion.cost_usd,
    )


def _grade(
    judge: LanguageModel, question: GoldenQuestion, answer: Answer
) -> tuple[bool, bool, str]:
    # Recusa não precisa de juiz: no escopo é erro, fora do escopo é o acerto.
    if not question.no_escopo:
        return answer.refused, answer.refused, "fora do escopo"
    if answer.refused:
        return False, True, "recusou uma pergunta que tinha resposta"
    context = format_context(answer.hits)
    verdict = judge_answer(
        judge, question.pergunta, question.resposta_esperada, context, answer.text
    )
    return verdict.correta, verdict.fiel, verdict.motivo


def summarize_generation(records: list[GenerationRecord]) -> dict[str, float]:
    """Taxas separadas para perguntas no escopo e fora dele, latência e custo."""
    in_scope = [r for r in records if r.in_scope]
    out_scope = [r for r in records if not r.in_scope]
    latencies = [r.latency_s for r in records]
    return {
        "corretas (no escopo)": _rate(in_scope, "correct"),
        "fiéis aos trechos (no escopo)": _rate(in_scope, "faithful"),
        "recusa indevida (no escopo)": _rate(in_scope, "refused"),
        "recusa correta (fora do escopo)": _rate(out_scope, "refused"),
        "latência p50 (s)": statistics.median(latencies),
        "latência p95 (s)": statistics.quantiles(latencies, n=20)[-1],
        "custo total (US$)": sum(r.cost_usd for r in records),
    }


def _rate(records: list[GenerationRecord], field: str) -> float:
    if not records:
        return 0.0
    return sum(bool(getattr(r, field)) for r in records) / len(records)
