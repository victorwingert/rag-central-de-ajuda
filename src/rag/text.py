"""Normalização de texto em português para a busca por palavras-chave."""

import re
import unicodedata

_STOPWORDS_RAW = """a ao aos as ate com como da das de do dos e em entre era essa esse
esta este eu foi ha isso ja lhe mais mas me meu minha na nas no nos nao o os ou para
pela pelo pode por pra qual quando que se seu sua sao so tambem te tem um uma umas uns
voce voces vai ser ta ter sem sobre esta estou"""
STOPWORDS = frozenset(_STOPWORDS_RAW.split())


def strip_accents(text: str) -> str:
    """Remove acentos para "reembolso" e "reembólso" virarem o mesmo termo.

    Exemplo: `strip_accents("não")` devolve `"nao"`.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acento, sem stopwords e sem tokens de uma letra.

    Exemplo: `tokenize("Quanto custa a taxa?")` devolve `["quanto", "custa", "taxa"]`.
    """
    tokens = re.findall(r"[a-z0-9]+", strip_accents(text.lower()))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
