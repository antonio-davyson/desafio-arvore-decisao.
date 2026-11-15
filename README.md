# Detecção de fraudes com Árvore de Decisão — Banco SeguraMais

Desafio Rocketseat: classificar transações bancárias como fraudulentas ou não usando uma árvore de decisão,
avaliada por sensibilidade e especificidade.

## Conteúdo
- `arvore_decisao_fraude.py` — exploração, preparação, treino, avaliação e análise crítica (no final do arquivo)
- `transacoes_fraude.csv` — dataset (13.000 transações, ~11% de fraudes)
- `arvore.png` — visualização da árvore treinada

## Como executar
```bash
pip install -r requirements.txt
python arvore_decisao_fraude.py
```

## Resultados (30% de teste, estratificado)
| Modelo | Sensibilidade | Especificidade |
|---|---|---|
| Árvore sem poda | ~14% | ~88% |
| Árvore ajustada (`max_depth=3`, `class_weight="balanced"`) | ~48,5% | ~73,9% |

Principal sinal encontrado: transações entre 23h e 4h59 têm ~18% de fraude contra ~8% no resto do dia.
O desempenho é modesto; limitações e melhorias estão na análise crítica no final do código.
