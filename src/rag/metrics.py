"""Métricas de busca contadas por documento.

Vários trechos do mesmo documento contam uma vez: o que importa para a resposta é
o documento certo chegar ao contexto.
"""

from rag.retrieval import Hit


def ranked_doc_ids(hits: list[Hit]) -> list[str]:
    """Ids dos documentos na ordem em que aparecem pela primeira vez."""
    return list(dict.fromkeys(hit.chunk.doc_id for hit in hits))


def recall_at_k(doc_ids: list[str], relevant: list[str], k: int) -> float:
    """Fração dos documentos relevantes que está entre os `k` primeiros.

    Exemplo: `recall_at_k(["a", "b"], ["b"], 1)` devolve `0.0`.
    """
    if not relevant:
        msg = f"recall exige ao menos um documento relevante; recebido {relevant!r}"
        raise ValueError(msg)
    return len(set(doc_ids[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(doc_ids: list[str], relevant: list[str]) -> float:
    """1/posição do primeiro documento relevante, ou 0 se nenhum aparece.

    Exemplo: `reciprocal_rank(["a", "b"], ["b"])` devolve `0.5`.
    """
    for position, doc_id in enumerate(doc_ids, start=1):
        if doc_id in relevant:
            return 1 / position
    return 0.0
