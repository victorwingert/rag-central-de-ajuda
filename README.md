# rag-central-de-ajuda

Assistente que responde perguntas sobre a central de ajuda de um app de delivery. Ele busca os trechos relevantes, responde citando a fonte e recusa o que não está na base. A busca e a geração são avaliadas separadamente.

A base é fictícia: escrevi 27 artigos de ajuda para clientes, entregadores e restaurantes, com prazos, valores e regras inventados. Nada foi copiado de empresa real.

## Como funciona

```
pergunta
  → busca por palavras-chave (BM25)  ┐
  → busca vetorial (embeddings)      ┴→ fusão por posição (RRF) → 3 trechos
  → LLM responde só com os trechos, citando [id-do-documento]
  → ou responde "Não encontrei essa informação na central de ajuda."
```

Os artigos são divididos em trechos respeitando as seções (`##`). Só uma seção maior que o limite é cortada em janelas com sobreposição, e o título do artigo entra no começo de cada trecho para ele fazer sentido sozinho.

O BM25 está escrito à mão em `src/rag/bm25.py`, com remoção de acentos e de stopwords em português. A busca vetorial usa modelos multilíngues que rodam localmente via ONNX (`fastembed`), sem GPU e sem chave de API, com os vetores guardados em cache. As duas listas são combinadas por Reciprocal Rank Fusion, que olha só a posição de cada trecho e por isso não exige calibrar as notas de uma busca contra a outra.

A resposta sai de qualquer modelo suportado pelo LiteLLM, com temperatura 0. O prompt pede citação, limita a resposta aos trechos e manda ignorar instruções que apareçam dentro deles.

## Avaliação

O conjunto de avaliação (`data/golden.jsonl`) tem 32 perguntas escritas à mão, com palavras diferentes das dos artigos ("motoboy", "refrigerante", "fui desativado"), e 5 perguntas fora do escopo, que devem ser recusadas.

Medir busca e geração separadamente mostra se uma resposta errada veio de um documento que não chegou ao contexto ou de um modelo que recebeu o documento certo e respondeu mal.

### Busca

Métricas por documento (vários trechos do mesmo artigo contam uma vez). `uv run rag avaliar-busca`:

| Busca | Embeddings | Trecho (palavras) | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| bm25 | - | 60 | 0.67 | 0.81 | 0.84 | 0.77 |
| vetorial | MiniLM-L12 | 60 | 0.58 | 0.67 | 0.78 | 0.67 |
| híbrida | MiniLM-L12 | 60 | 0.55 | 0.81 | 0.91 | 0.70 |
| vetorial | mpnet-base | 60 | 0.61 | 0.84 | 0.91 | 0.74 |
| híbrida | mpnet-base | 60 | 0.61 | 0.88 | 0.97 | 0.76 |
| bm25 | - | 150 | 0.73 | 0.84 | 0.91 | 0.81 |
| vetorial | MiniLM-L12 | 150 | 0.45 | 0.58 | 0.73 | 0.58 |
| híbrida | MiniLM-L12 | 150 | 0.55 | 0.78 | 0.94 | 0.69 |
| vetorial | mpnet-base | 150 | 0.61 | 0.78 | 0.91 | 0.72 |
| híbrida | mpnet-base | 150 | 0.67 | 0.88 | 0.97 | 0.80 |

Tabela completa, com as perguntas que erraram, em [`results/busca.md`](results/busca.md).

Com o modelo pequeno (MiniLM), a busca vetorial perde para o BM25 em todas as métricas. Os artigos são curtos e cheios de termos específicos como "Pix", "CNH" e "Clube", e isso favorece a busca por palavras-chave.

O modelo maior (mpnet) melhora a vetorial, mas o ganho maior vem de combinar as duas: a híbrida com mpnet e trechos de 150 palavras traz o documento certo entre os 5 primeiros em 97% das perguntas. O BM25 sozinho ainda acerta mais o primeiro lugar (recall@1 de 0,73), mas a geração recebe 3 trechos, então o recall@3 pesa mais. Por isso deixei a híbrida como padrão.

### Onde a busca erra

Estas são as perguntas que a configuração padrão mais erra. Em todas, quem pergunta usa palavras que o artigo não usa.

| Pergunta | Documento certo | O que veio na frente | Causa |
|---|---|---|---|
| "Paguei com Pix e o pedido foi cancelado. Em quanto tempo o dinheiro volta?" | reembolso (4º) | formas de pagamento | "Pix" puxa o artigo de pagamento, que fala do Pix do checkout |
| "Faltou o refrigerante no meu pedido, até quando posso reclamar?" | item faltando | atraso, retirada | o artigo diz "item faltando" e "solicitação"; a pergunta não usa nenhuma das duas |
| "O entregador pediu dinheiro a mais na porta, é normal?" | golpes e código de entrega | gorjeta, ganhos do entregador | a regra está no artigo sobre golpes, e o título não menciona cobrança |
| "Fui desativado, tenho como me defender?" | conta bloqueada (4º) | cancelamento, conta do cliente | sem stemming, "desativado" não casa com "desativada", nem "defender" com "defesa" |

### Geração

`uv run rag avaliar-geracao` responde todas as perguntas e usa um segundo modelo como juiz (LLM-as-judge) para marcar cada resposta como correta (traz a mesma informação essencial da resposta de referência) e fiel (não afirma nada fora dos trechos). Recusas não passam pelo juiz: no escopo contam como erro e fora do escopo contam como acerto.

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

Das 5 respostas erradas, 4 são as perguntas em que a busca não trouxe o documento certo entre os 3 primeiros (q01, q08, q18 e q27). Em três delas o modelo recusou, que é o que o prompt pede quando o contexto não tem a resposta. Na do Pix ele respondeu sobre o Pix do checkout, que era o que estava no contexto. A quinta (q19) foi rigor do juiz: a resposta deu o preço certo do plano e não mencionou o primeiro mês grátis. Como nenhuma resposta inventou informação, o acerto que falta depende de a busca trazer o trecho certo.

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

- Um stemmer para português (RSLP) no BM25 resolveria parte dos erros de flexão, como "desativado" contra "desativada".
- Um reranker (cross-encoder multilíngue) sobre os 20 primeiros trechos deve corrigir casos como o do Pix, em que o documento certo aparece em 4º.
- Reescrever a pergunta com o LLM, no vocabulário da central, antes de buscar.
- 32 perguntas mostram a direção, mas não separam com segurança estatística configurações com números próximos.
- O juiz não foi comparado com rótulos humanos. Falta rotular uma amostra à mão e medir a concordância.
