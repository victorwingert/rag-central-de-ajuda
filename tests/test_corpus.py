from pathlib import Path

import pytest

from rag.corpus import ChunkingConfig, Document, chunk_document, load_documents


def test_load_documents_reads_title_and_body(tmp_path: Path):
    (tmp_path / "cliente-taxas.md").write_text("# Taxas\nTexto da taxa.\n", "utf-8")
    [doc] = load_documents(tmp_path)
    assert (doc.id, doc.title, doc.text) == ("cliente-taxas", "Taxas", "Texto da taxa.")


def test_short_document_becomes_one_chunk_with_title():
    doc = Document("d", "Título", "uma duas três\n## Seção\nquatro cinco")
    [chunk] = chunk_document(doc, ChunkingConfig(max_words=50, overlap=5))
    assert chunk.id == "d#0"
    assert chunk.text.startswith("Título. uma duas três ## Seção")


def test_sections_are_not_merged_past_the_limit():
    doc = Document("d", "T", "a b c d\n## S\ne f g h")
    chunks = chunk_document(doc, ChunkingConfig(max_words=6, overlap=1))
    assert [c.text for c in chunks] == ["T. a b c d", "T. ## S e f g h"]


def test_long_section_is_split_with_overlap():
    words = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_document(Document("d", "T", words), ChunkingConfig(6, 2))
    assert [c.text for c in chunks] == ["T. w0 w1 w2 w3 w4 w5", "T. w4 w5 w6 w7 w8 w9"]


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(max_words=5, overlap=5)
