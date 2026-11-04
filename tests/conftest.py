import pytest

from rag.corpus import Chunk


def make_chunk(doc_id: str, text: str, index: int = 0) -> Chunk:
    return Chunk(id=f"{doc_id}#{index}", doc_id=doc_id, text=text)


@pytest.fixture
def help_chunks() -> list[Chunk]:
    return [
        make_chunk("reembolso", "Reembolso por Pix cai em até 1 dia útil."),
        make_chunk("cupons", "Só um cupom por pedido; cupons não se somam."),
        make_chunk("gorjeta", "A gorjeta vai 100% para o entregador."),
    ]
