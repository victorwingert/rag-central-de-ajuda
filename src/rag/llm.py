"""Interface própria para modelos de linguagem e o adaptador do LiteLLM."""

import time
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    latency_s: float
    cost_usd: float


class LanguageModel(Protocol):
    def complete(self, messages: list[Message]) -> Completion: ...


class LiteLLMModel:
    """Qualquer provedor suportado pelo LiteLLM, com temperatura 0.

    Exemplo: `LiteLLMModel("gemini/gemini-2.5-flash").complete(messages)`.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, messages: list[Message]) -> Completion:
        import litellm

        start = time.perf_counter()
        response = litellm.completion(
            model=self.model, messages=messages, temperature=0
        )  # pyright: ignore[reportUnknownMemberType]
        latency = time.perf_counter() - start
        content = cast("str | None", response.choices[0].message.content)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        text = (content or "").strip()
        return Completion(text, latency, _cost_usd(response))


def _cost_usd(response: object) -> float:
    import litellm

    try:
        return float(litellm.completion_cost(completion_response=response))  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
    except Exception:  # noqa: BLE001 - modelo sem preço na tabela do LiteLLM
        return 0.0
