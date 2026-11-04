"""LLM-as-judge: a resposta está correta e é fiel aos trechos recebidos?"""

import json

from pydantic import BaseModel, ValidationError

from rag.llm import LanguageModel, Message

JUDGE_PROMPT = """Você avalia respostas de um assistente de suporte.
Responda só com um JSON: {"correta": true|false, "fiel": true|false, "motivo": "..."}
- "correta": traz a mesma informação essencial da referência (prazos, valores, \
condições).
- "fiel": tudo o que a resposta afirma está nos trechos; dado inventado torna false."""


class Verdict(BaseModel):
    correta: bool
    fiel: bool
    motivo: str


def build_judge_messages(
    question: str, reference: str, context: str, answer: str
) -> list[Message]:
    user = (
        f"Pergunta: {question}\n\nReferência: {reference}\n\n"
        f"Trechos:\n{context}\n\nResposta do assistente: {answer}"
    )
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_verdict(raw: str) -> Verdict:
    """Aceita o JSON mesmo cercado de texto ou de bloco de código.

    Exemplo: `parse_verdict('{"correta": true, "fiel": true, "motivo": "ok"}')`.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        msg = f"juiz não devolveu um objeto JSON; recebido {raw!r}"
        raise ValueError(msg)
    try:
        return Verdict.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as exc:
        msg = f"JSON do juiz fora do formato {{correta, fiel, motivo}}: {raw!r}"
        raise ValueError(msg) from exc


def judge_answer(
    model: LanguageModel, question: str, reference: str, context: str, answer: str
) -> Verdict:
    messages = build_judge_messages(question, reference, context, answer)
    return parse_verdict(model.complete(messages).text)
