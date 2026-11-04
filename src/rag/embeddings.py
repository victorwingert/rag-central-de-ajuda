"""Embeddings: interface própria, adaptador do fastembed e cache em disco."""

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type Matrix = NDArray[np.float32]

SMALL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_DIR = Path.home() / ".cache" / "fastembed"


class Embedder(Protocol):
    """Transforma textos em vetores normalizados (uma linha por texto)."""

    def embed(self, texts: list[str]) -> Matrix: ...


class FastEmbedEmbedder:
    """Modelo local via ONNX, sem GPU e sem chave de API.

    Exemplo: `FastEmbedEmbedder().embed(["reembolso por Pix"])`.
    """

    def __init__(
        self, model_name: str = DEFAULT_EMBEDDING_MODEL, model_dir: Path = MODEL_DIR
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        # O padrão do fastembed é /tmp, que costuma ser um tmpfs pequeno.
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(model_dir))

    def embed(self, texts: list[str]) -> Matrix:
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return normalize_rows(vectors)


class DiskCachedEmbedder:
    """Guarda os vetores de cada lista de textos para não recalcular a cada avaliação.

    Exemplo: `DiskCachedEmbedder(FastEmbedEmbedder(), Path(".cache"), "minilm")`.
    """

    def __init__(self, inner: Embedder, folder: Path, model_name: str) -> None:
        self._inner = inner
        self._folder = folder
        self._model_name = model_name

    def embed(self, texts: list[str]) -> Matrix:
        path = self._folder / f"emb-{self._cache_key(texts)}.npy"
        if path.exists():
            return np.load(path).astype(np.float32)
        vectors = self._inner.embed(texts)
        self._folder.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def _cache_key(self, texts: list[str]) -> str:
        payload = "\x00".join([self._model_name, *texts]).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def normalize_rows(vectors: Matrix) -> Matrix:
    """Normaliza cada linha para que o produto escalar seja a similaridade cosseno."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)
