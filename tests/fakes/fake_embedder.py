"""Embedders falsos: determinísticos e sem baixar modelo."""

import numpy as np

from rag.embeddings import Matrix, normalize_rows
from rag.text import tokenize


class BagOfWordsEmbedder:
    """Um eixo por palavra do vocabulário dado; serve para testar a busca vetorial."""

    def __init__(self, vocabulary: list[str]) -> None:
        self._axis = {word: i for i, word in enumerate(vocabulary)}

    def embed(self, texts: list[str]) -> Matrix:
        matrix = np.zeros((len(texts), len(self._axis) + 1), dtype=np.float32)
        matrix[:, -1] = (
            1e-3  # evita vetor nulo quando o texto não tem palavra conhecida
        )
        for row, text in enumerate(texts):
            for token in tokenize(text):
                if token in self._axis:
                    matrix[row, self._axis[token]] += 1
        return normalize_rows(matrix)


class CountingEmbedder:
    """Conta as chamadas para verificar o cache em disco."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> Matrix:
        self.calls += 1
        return normalize_rows(np.ones((len(texts), 3), dtype=np.float32))
