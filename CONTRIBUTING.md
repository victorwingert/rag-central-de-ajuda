# Contributing

## Code style

- Functions: 4-20 lines. Split if longer.
- Files: under 500 lines. Split by responsibility.
- One thing per function, one responsibility per module (SRP).
- Names: specific and unique. Avoid `data`, `handler`, `Manager`.
- Types: explicit. No `Any`, no bare `dict`, no untyped functions. Pyright runs in strict mode.
- No code duplication. Extract shared logic into a function/module.
- Early returns over nested ifs. Max 2 levels of indentation.
- Exception messages must include the offending value and expected shape.

## Comments

- Write WHY, not WHAT.
- Docstrings on public functions: intent + one usage example.

## Tests

- Tests run with a single command: `uv run pytest`.
- Every new function gets a test. Bug fixes get a regression test.
- Mock external I/O (LLM, embeddings, filesystem) with named fake classes in `tests/fakes/`.
- Tests must be fast, independent and offline.

## Dependencies

- Inject dependencies through constructor/parameter.
- Wrap third-party libs behind a thin interface owned by this project
  (`rag.llm.LanguageModel`, `rag.embeddings.Embedder`).

## Evaluation

- Retrieval and generation are evaluated separately.
- `data/golden.jsonl` is hand-written. Never edit an expected answer to make a run pass.

## Formatting

- `uv run ruff format` and `uv run ruff check`.
