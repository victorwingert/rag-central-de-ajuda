"""Conjunto de avaliação escrito à mão (data/golden.jsonl)."""

from pathlib import Path

from pydantic import BaseModel


class GoldenQuestion(BaseModel):
    id: str
    pergunta: str
    docs_relevantes: list[str]
    resposta_esperada: str
    no_escopo: bool


def load_golden(path: Path) -> list[GoldenQuestion]:
    """Uma pergunta por linha. Exemplo: `load_golden(Path("data/golden.jsonl"))`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenQuestion.model_validate_json(line) for line in lines if line.strip()]
