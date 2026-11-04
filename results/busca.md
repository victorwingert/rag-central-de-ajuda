| Busca | Embeddings | Trecho (palavras) | Trechos | Recall@1 | Recall@3 | Recall@5 | MRR | Erros no top-3 |
|---|---|---|---|---|---|---|---|---|
| bm25 | - | 60 | 60 | 0.67 | 0.81 | 0.84 | 0.77 | q01, q08, q09, q16, q18, q27 |
| vetorial | paraphrase-multilingual-MiniLM-L12-v2 | 60 | 60 | 0.58 | 0.67 | 0.78 | 0.67 | q01, q08, q11, q12, q16, q18, q20, q25, q29, q32 |
| hibrida | paraphrase-multilingual-MiniLM-L12-v2 | 60 | 60 | 0.55 | 0.81 | 0.91 | 0.70 | q01, q08, q16, q18, q27, q32 |
| vetorial | paraphrase-multilingual-mpnet-base-v2 | 60 | 60 | 0.61 | 0.84 | 0.91 | 0.74 | q09, q11, q18, q29, q32 |
| hibrida | paraphrase-multilingual-mpnet-base-v2 | 60 | 60 | 0.61 | 0.88 | 0.97 | 0.76 | q08, q16, q18, q27 |
| bm25 | - | 150 | 28 | 0.73 | 0.84 | 0.91 | 0.81 | q01, q08, q16, q18, q27 |
| vetorial | paraphrase-multilingual-MiniLM-L12-v2 | 150 | 28 | 0.45 | 0.58 | 0.73 | 0.58 | q01, q08, q09, q11, q12, q16, q18, q20, q21, q22, q25, q29, q32 |
| hibrida | paraphrase-multilingual-MiniLM-L12-v2 | 150 | 28 | 0.55 | 0.78 | 0.94 | 0.69 | q01, q08, q09, q16, q18, q20, q27 |
| vetorial | paraphrase-multilingual-mpnet-base-v2 | 150 | 28 | 0.61 | 0.78 | 0.91 | 0.72 | q01, q11, q18, q20, q29, q31, q32 |
| hibrida | paraphrase-multilingual-mpnet-base-v2 | 150 | 28 | 0.67 | 0.88 | 0.97 | 0.80 | q01, q08, q18, q27 |
