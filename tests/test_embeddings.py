from pathlib import Path

import numpy as np

from rag.embeddings import DiskCachedEmbedder, normalize_rows
from tests.fakes.fake_embedder import CountingEmbedder


def test_disk_cache_avoids_recomputing(tmp_path: Path):
    inner = CountingEmbedder()
    cached = DiskCachedEmbedder(inner, tmp_path, "modelo-x")
    first = cached.embed(["a", "b"])
    second = cached.embed(["a", "b"])
    assert inner.calls == 1
    np.testing.assert_array_equal(first, second)


def test_normalize_rows_gives_unit_vectors():
    rows = normalize_rows(np.array([[3, 4], [0, 0]], dtype=np.float32))
    assert rows[0].tolist() == [0.6000000238418579, 0.800000011920929]
    np.testing.assert_array_equal(rows[1], [0, 0])
