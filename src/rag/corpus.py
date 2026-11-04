"""Leitura dos documentos da central de ajuda e divisão em trechos."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int
    overlap: int

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.max_words:
            msg = f"overlap deve estar em [0, max_words); recebido {self!r}"
            raise ValueError(msg)


def load_documents(folder: Path) -> list[Document]:
    """Lê cada `.md` da pasta; a primeira linha (`# Título`) vira o título.

    Exemplo: `load_documents(Path("data/docs"))`.
    """
    return [_parse_document(path) for path in sorted(folder.glob("*.md"))]


def _parse_document(path: Path) -> Document:
    first_line, _, body = path.read_text(encoding="utf-8").strip().partition("\n")
    title = first_line.removeprefix("#").strip()
    return Document(id=path.stem, title=title, text=body.strip())


def chunk_corpus(docs: list[Document], config: ChunkingConfig) -> list[Chunk]:
    """Divide todos os documentos com a mesma configuração.

    Exemplo: `chunk_corpus(docs, ChunkingConfig(max_words=150, overlap=30))`.
    """
    return [chunk for doc in docs for chunk in chunk_document(doc, config)]


def chunk_document(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    """Agrupa seções inteiras até o limite de palavras e só corta seções maiores.

    O título entra no começo de cada trecho para ele fazer sentido sozinho na busca.
    Exemplo: `chunk_document(doc, ChunkingConfig(60, 12))`.
    """
    pieces = _pack_sections(_split_sections(doc.text), config)
    return [
        Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=f"{doc.title}. {piece}")
        for i, piece in enumerate(pieces)
    ]


def _split_sections(text: str) -> list[list[str]]:
    sections = re.split(r"\n(?=## )", text)
    return [section.split() for section in sections if section.strip()]


def _pack_sections(sections: list[list[str]], config: ChunkingConfig) -> list[str]:
    pieces: list[list[str]] = []
    current: list[str] = []
    for words in sections:
        if len(current) + len(words) > config.max_words and current:
            pieces.append(current)
            current = []
        if len(words) > config.max_words:
            pieces.extend(_windows(words, config))
            continue
        current = current + words
    if current:
        pieces.append(current)
    return [" ".join(piece) for piece in pieces]


def _windows(words: list[str], config: ChunkingConfig) -> list[list[str]]:
    step = config.max_words - config.overlap
    last_start = max(len(words) - config.overlap, 1)
    return [
        words[start : start + config.max_words] for start in range(0, last_start, step)
    ]
