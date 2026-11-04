# rag-central-de-ajuda

Assistente de perguntas e respostas sobre a central de ajuda de um app de delivery, com busca híbrida, resposta com citação da fonte e avaliação da busca e da geração em separado.

A base é **fictícia**: 27 artigos de ajuda escritos para este projeto (clientes, entregadores e restaurantes), com prazos, valores e regras próprios. Nenhum conteúdo foi copiado de empresa real.

## Como funciona

```
pergunta
  → busca por palavras-chave (BM25)  ┐
  → busca vetorial (embeddings)      ┴→ fusão por posição (RRF) → 3 trechos
  → LLM responde só com os trechos, citando [id-do-documento]
  → ou responde "Não encontrei essa informação na central de ajuda."
```

- **Trechos:** cada artigo é dividido respeitando as seções (`##`); só seções maiores que o limite são cortadas em janelas com sobreposição. O título do artigo vai no começo de cada trecho.
- **BM25:** escrito à mão em `src/rag/bm25.py`, com normalização de acentos e stopwords em português.
- **Embeddings:** modelos multilíngues rodando localmente via ONNX (`fastembed`), sem GPU e sem chave de API. Os vetores ficam em cache em disco.
- **Híbrida:** Reciprocal Rank Fusion, que junta as duas listas pela posição e dispensa calibrar a escala das notas de cada busca.
- **Geração:** qualquer modelo suportado pelo LiteLLM, com temperatura 0. O prompt exige citação, limita a resposta aos trechos e manda ignorar instruções que apareçam dentro deles.

## Avaliação

O conjunto de avaliação (`data/golden.jsonl`) tem 32 perguntas escritas à mão, com palavras diferentes das dos artigos ("motoboy", "refrigerante", "fui desativado"), e 5 perguntas fora do escopo, que devem ser recusadas.

A busca e a geração são medidas separadamente. Assim dá para saber se uma resposta errada veio de um documento que não chegou ao contexto ou de um modelo que leu o documento certo e respondeu mal.

### Busca

Métricas por documento (vários trechos do mesmo artigo contam uma vez). `uv run rag avaliar-busca`:

| Busca | Embeddings | Trecho (palavras) | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| bm25 | - | 60 | 0.67 | 0.81 | 0.84 | 0.77 |
| vetorial | MiniLM-L12 | 60 | 0.58 | 0.67 | 0.78 | 0.67 |
| híbrida | MiniLM-L12 | 60 | 0.55 | 0.81 | 0.91 | 0.70 |
| vetorial | mpnet-base | 60 | 0.61 | 0.84 | 0.91 | 0.74 |
| híbrida | mpnet-base | 60 | 0.61 | 0.88 | 0.97 | 0.76 |
| bm25 | - | 150 | **0.73** | 0.84 | 0.91 | **0.81** |
| vetorial | MiniLM-L12 | 150 | 0.45 | 0.58 | 0.73 | 0.58 |
| híbrida | MiniLM-L12 | 150 | 0.55 | 0.78 | 0.94 | 0.69 |
| vetorial | mpnet-base | 150 | 0.61 | 0.78 | 0.91 | 0.72 |
| **híbrida** | **mpnet-base** | **150** | 0.67 | **0.88** | **0.97** | 0.80 |

Tabela completa, com as perguntas que erraram, em [`results/busca.md`](results/busca.md).

O que a tabela mostra:

- Com o modelo pequeno (MiniLM), a busca vetorial perde para o BM25 em todas as métricas. Nesta base os artigos são curtos e cheios de termos específicos ("Pix", "CNH", "Clube"), o que favorece palavras-chave.
- O modelo maior (mpnet) melhora a vetorial, mas o que mais ajuda é combinar as duas. A híbrida com mpnet e trechos de 150 palavras coloca o documento certo entre os 5 primeiros em 97% das perguntas.
- O BM25 sozinho ainda acerta mais no primeiro lugar (recall@1 de 0,73). Como a geração recebe 3 trechos, o recall@3 importa mais, e por isso a configuração padrão é a híbrida.

### Onde a busca erra

| Pergunta | Documento certo | O que veio na frente | Causa |
|---|---|---|---|
| "Paguei com Pix e o pedido foi cancelado. Em quanto tempo o dinheiro volta?" | reembolso (4º) | formas de pagamento | "Pix" puxa o artigo de pagamento, que fala do Pix do checkout |
| "Faltou o refrigerante no meu pedido, até quando posso reclamar?" | item faltando | atraso, retirada | o artigo diz "item faltando" e "solicitação"; a pergunta não usa nenhuma das duas |
| "O entregador pediu dinheiro a mais na porta, é normal?" | golpes e código de entrega | gorjeta, ganhos do entregador | a regra está num artigo cujo título fala de golpes, não de cobrança |
| "Fui desativado, tenho como me defender?" | conta bloqueada (4º) | cancelamento, conta do cliente | sem stemming, "desativado" não casa com "desativada", nem "defender" com "defesa" |

Os quatro erros são de vocabulário. Os próximos passos que atacam esse problema estão no fim.

### Geração

`uv run rag avaliar-geracao` responde todas as perguntas e usa um segundo modelo como juiz (LLM-as-judge) para marcar cada resposta como **correta** (mesma informação essencial da resposta de referência) e **fiel** (não afirma nada que não esteja nos trechos). Recusas não passam pelo juiz: no escopo contam como erro e fora do escopo contam como acerto.

Resultado com busca híbrida (mpnet, trechos de 150 palavras, 3 trechos no contexto), `gpt-4.1-mini` respondendo e `gpt-4.1` como juiz:

| Métrica | Valor |
|---|---|
| Corretas (32 perguntas no escopo) | 84% |
| Fiéis aos trechos | 100% |
| Recusa indevida (no escopo) | 9% |
| Recusa correta (5 fora do escopo) | 100% |
| Latência p50 / p95 | 0,8 s / 1,4 s |
| Custo da rodada inteira | US$ 0,01 |

Detalhes por pergunta em [`results/geracao.md`](results/geracao.md) e [`results/geracao.json`](results/geracao.json).

Das 5 respostas erradas, 4 são as mesmas perguntas em que a busca não trouxe o documento certo entre os 3 primeiros (q01, q08, q18 e q27). Em três delas o modelo recusou, como o prompt manda quando o contexto não tem a resposta. Na do Pix, respondeu sobre o Pix do checkout, que era o que estava no contexto, sem inventar nada. A quinta (q19) é um caso de juiz rigoroso: a resposta deu o preço certo do plano, mas não mencionou o primeiro mês grátis.

Ou seja, com o prompt atual o modelo não alucina; quando erra, é porque a busca não entregou o trecho certo. Melhorar a busca é o que mais aumenta o acerto.

## Como rodar

Requer [uv](https://docs.astral.sh/uv/). Os modelos de embeddings (~1,2 GB no total) são baixados na primeira execução para `~/.cache/fastembed`.

```bash
uv sync
uv run rag avaliar-busca          # não precisa de chave

cp .env.example .env              # coloque a chave do provedor do LLM
set -a; source .env; set +a
uv run rag perguntar "Quanto tempo leva o reembolso no cartão de crédito?"
uv run rag avaliar-geracao --modelo gemini/gemini-2.5-flash --juiz gemini/gemini-2.5-flash
```

Qualquer modelo do LiteLLM funciona em `--modelo` e `--juiz` (por exemplo `openai/gpt-4.1-mini` ou `anthropic/claude-haiku-4-5`).

## Qualidade do código

```bash
uv run pytest        # 30 testes, sem rede e sem modelo
uv run ruff check
uv run pyright       # modo strict
```

Os testes usam fakes nomeados em `tests/fakes/`: um embedder de bag-of-words e um modelo de linguagem com respostas roteirizadas. O LiteLLM e o fastembed ficam atrás de interfaces próprias (`LanguageModel` e `Embedder`), então nada no núcleo depende da biblioteca de um provedor.

## Limitações e próximos passos

- **Stemming e sinônimos:** um stemmer para português (RSLP) no BM25 resolveria parte dos erros de flexão, como "desativado" e "desativada".
- **Reranker:** um cross-encoder multilíngue sobre os 20 primeiros trechos deve corrigir casos como o do Pix, em que o documento certo está em 4º.
- **Reescrita da pergunta:** pedir ao LLM uma versão da pergunta no vocabulário da central antes de buscar.
- **Conjunto de avaliação pequeno:** 32 perguntas dão uma direção, não uma diferença estatisticamente segura entre configurações próximas.
- **Juiz não calibrado:** o juiz não foi comparado com rótulos humanos. O passo seguinte seria rotular uma amostra à mão e medir a concordância.
