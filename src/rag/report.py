"""Tabelas em Markdown para o README e a pasta results/."""

from rag.evaluation import GenerationRecord, RetrievalScore

RETRIEVAL_HEADER = (
    "| Busca | Embeddings | Trecho (palavras) | Trechos | Recall@1 | Recall@3 "
    "| Recall@5 | MRR | Erros no top-3 |\n|---|---|---|---|---|---|---|---|---|"
)


def retrieval_row(
    name: str, model: str, max_words: int, chunk_count: int, s: RetrievalScore
) -> str:
    misses = ", ".join(s.misses_top3) or "-"
    short_model = model.split("/")[-1]
    return (
        f"| {name} | {short_model} | {max_words} | {chunk_count} | "
        f"{s.recall_at_1:.2f} | "
        f"{s.recall_at_3:.2f} | {s.recall_at_5:.2f} | {s.mrr:.2f} | {misses} |"
    )


def summary_table(summary: dict[str, float]) -> str:
    rows = [f"| {name} | {value:.3f} |" for name, value in summary.items()]
    return "\n".join(["| Métrica | Valor |", "|---|---|", *rows])


def failures_table(records: list[GenerationRecord]) -> str:
    """Só as respostas erradas, com o motivo dado pelo juiz."""
    rows = [
        f"| {r.question_id} | {r.answer.replace('|', '/')} | {r.reason} |"
        for r in records
        if not r.correct
    ]
    return "\n".join(["| Pergunta | Resposta | Motivo |", "|---|---|---|", *rows])
