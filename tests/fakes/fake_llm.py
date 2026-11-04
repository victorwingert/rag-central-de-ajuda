"""Modelo de linguagem falso que devolve respostas roteirizadas."""

from collections.abc import Callable

from rag.llm import Completion, Message


class ScriptedLanguageModel:
    """Responde com `reply(messages)` e guarda o que recebeu."""

    def __init__(self, reply: Callable[[list[Message]], str]) -> None:
        self._reply = reply
        self.received: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> Completion:
        self.received.append(messages)
        return Completion(text=self._reply(messages), latency_s=0.5, cost_usd=0.001)


def constant(text: str) -> ScriptedLanguageModel:
    return ScriptedLanguageModel(lambda _: text)
