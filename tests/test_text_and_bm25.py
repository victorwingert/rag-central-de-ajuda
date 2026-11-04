from rag.bm25 import BM25Index
from rag.text import strip_accents, tokenize


def test_tokenize_drops_accents_stopwords_and_single_letters():
    assert tokenize("Não é a taxa de serviço?") == ["taxa", "servico"]


def test_strip_accents():
    assert strip_accents("reembolsô") == "reembolso"


def test_bm25_ranks_the_matching_text_first():
    index = BM25Index(["taxa de entrega", "reembolso por pix", "cupom de desconto"])
    scores = index.scores("quando cai o reembolso do pix")
    assert scores.index(max(scores)) == 1


def test_bm25_unknown_term_scores_zero():
    assert BM25Index(["taxa de entrega"]).scores("criptomoeda") == [0.0]
