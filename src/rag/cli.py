"""Linha de comando: perguntar, avaliar a busca e avaliar a geração."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag.answering import answer_question
from rag.corpus import Chunk, ChunkingConfig, chunk_corpus, load_documents
from rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SMALL_EMBEDDING_MODEL,
    DiskCachedEmbedder,
    Embedder,
    FastEmbedEmbedder,
)
from rag.evaluation import (
    GenerationSetup,
    evaluate_generation,
    evaluate_retrieval,
    summarize_generation,
)
from rag.golden import load_golden
from rag.llm import LiteLLMModel
from rag.report import RETRIEVAL_HEADER, failures_table, retrieval_row, summary_table
from rag.retrieval import BM25Retriever, HybridRetriever, Retriever, VectorRetriever

DOCS_DIR = Path("data/docs")
GOLDEN_PATH = Path("data/golden.jsonl")
RESULTS_DIR = Path("results")
CACHE_DIR = Path(".cache")
DEFAULT_LLM = "gemini/gemini-2.5-flash"
STRATEGIES = ("bm25", "vetorial", "hibrida")


def load_chunks(max_words: int) -> list[Chunk]:
    config = ChunkingConfig(max_words=max_words, overlap=max_words // 5)
    return chunk_corpus(load_documents(DOCS_DIR), config)


def make_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    return DiskCachedEmbedder(FastEmbedEmbedder(model_name), CACHE_DIR, model_name)


def make_retriever(strategy: str, chunks: list[Chunk], embedder: Embedder) -> Retriever:
    if strategy == "bm25":
        return BM25Retriever(chunks)
    if strategy == "vetorial":
        return VectorRetriever(chunks, embedder)
    if strategy == "hibrida":
        return HybridRetriever(BM25Retriever(chunks), VectorRetriever(chunks, embedder))
    msg = f"estratégia deve ser uma de {STRATEGIES}; recebido {strategy!r}"
    raise ValueError(msg)


def run_ask(args: argparse.Namespace) -> None:
    chunks = load_chunks(args.palavras)
    retriever = make_retriever(args.busca, chunks, make_embedder())
    hits = retriever.search(args.pergunta, args.k)
    for hit in hits:
        print(f"  {hit.score:.3f}  {hit.chunk.id}")
    print("\n" + answer_question(args.pergunta, hits, LiteLLMModel(args.modelo)).text)


def run_retrieval_eval(_: argparse.Namespace) -> None:
    rows = [RETRIEVAL_HEADER]
    for max_words in (60, 150):
        rows.extend(_retrieval_rows(max_words))
    _write_result("busca.md", "\n".join(rows))


def _retrieval_rows(max_words: int) -> list[str]:
    """BM25 uma vez; vetorial e híbrida com cada modelo de embeddings."""
    questions = load_golden(GOLDEN_PATH)
    chunks = load_chunks(max_words)
    configs = [("bm25", "-")] + [
        (strategy, model)
        for model in (SMALL_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL)
        for strategy in ("vetorial", "hibrida")
    ]
    rows: list[str] = []
    for strategy, model in configs:
        embedder = make_embedder(model if model != "-" else DEFAULT_EMBEDDING_MODEL)
        score = evaluate_retrieval(
            make_retriever(strategy, chunks, embedder), questions
        )
        rows.append(retrieval_row(strategy, model, max_words, len(chunks), score))
    return rows


def run_generation_eval(args: argparse.Namespace) -> None:
    chunks = load_chunks(args.palavras)
    setup = GenerationSetup(
        retriever=make_retriever(args.busca, chunks, make_embedder()),
        answerer=LiteLLMModel(args.modelo),
        judge=LiteLLMModel(args.juiz),
        k=args.k,
    )
    records = evaluate_generation(setup, load_golden(GOLDEN_PATH))
    report = (
        f"{summary_table(summarize_generation(records))}\n\n{failures_table(records)}"
    )
    _write_result("geracao.md", report)
    raw = json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2)
    (RESULTS_DIR / "geracao.json").write_text(raw + "\n", encoding="utf-8")


def _write_result(name: str, content: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / name).write_text(content + "\n", encoding="utf-8")
    print(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(required=True)
    ask = sub.add_parser("perguntar", help="responde uma pergunta")
    ask.add_argument("pergunta")
    _add_pipeline_args(ask)
    ask.set_defaults(func=run_ask)
    sub.add_parser("avaliar-busca", help="compara as buscas").set_defaults(
        func=run_retrieval_eval
    )
    gen = sub.add_parser("avaliar-geracao", help="avalia as respostas com LLM-as-judge")
    _add_pipeline_args(gen)
    gen.add_argument("--juiz", default=DEFAULT_LLM)
    gen.set_defaults(func=run_generation_eval)
    return parser


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--busca", default="hibrida", choices=STRATEGIES)
    parser.add_argument("--palavras", type=int, default=150)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--modelo", default=DEFAULT_LLM)


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
