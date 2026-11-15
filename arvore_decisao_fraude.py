# %% [markdown]
# # Desafio: Árvore de Decisão para detecção de fraudes — Banco SeguraMais
# %%
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# %% 1. Exploração do dataset
df = pd.read_csv("transacoes_fraude.csv")
print(df.shape); print(df.dtypes); print(df.isna().sum())
print(df["Classe"].value_counts(normalize=True))          # ~11% fraudes (desbalanceado)
print(df.groupby("Tipo de Transacao")["Classe"].mean())
# Valor Anterior - Valor Após == Valor da Transação em 100% das linhas (variáveis redundantes)
print(np.allclose(df["Valor Anterior a Transacao"] - df["Valor Apos a Transacao"], df["Valor da Transacao"]))

# %% 2. Preparação dos dados
df["Horario da Transacao"] = pd.to_datetime(df["Horario da Transacao"])
df["hora"] = df["Horario da Transacao"].dt.hour
df["madrugada"] = df["hora"].isin([23, 0, 1, 2, 3, 4]).astype(int)  # fraude ~18% vs ~8% no resto do dia
print(df.groupby("madrugada")["Classe"].mean())
# sem valores ausentes; "Cliente" é identificador (descartado); categórica -> one-hot
X = pd.get_dummies(
    df[["Tipo de Transacao", "Valor da Transacao", "Valor Anterior a Transacao",
        "Valor Apos a Transacao", "hora", "madrugada"]],
    columns=["Tipo de Transacao"])
y = df["Classe"]

# %% 3. Divisão treino/teste (estratificada)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def avaliar(nome, modelo):
    tn, fp, fn, tp = confusion_matrix(y_te, modelo.predict(X_te)).ravel()
    sens, esp = tp / (tp + fn), tn / (tn + fp)
    print(f"{nome}: sensibilidade={sens:.3f} especificidade={esp:.3f} | VP={tp} FN={fn} VN={tn} FP={fp}")
    return sens, esp

# %% 4. Modelos de árvore de decisão
base = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)        # sem restrição (overfit)
avaliar("Árvore sem poda", base)

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [20, 50, 100, 200]},
    scoring="balanced_accuracy", cv=5).fit(X_tr, y_tr)               # média de sensib. e especif.
print("Melhores parâmetros:", grid.best_params_)
modelo = grid.best_estimator_

# %% 5. Avaliação (sensibilidade e especificidade)
avaliar("Árvore ajustada", modelo)
print(classification_report(y_te, modelo.predict(X_te), target_names=["Normal", "Fraude"]))
print(pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False).round(3))

plt.figure(figsize=(18, 8))
plot_tree(modelo, feature_names=X.columns, class_names=["Normal", "Fraude"], filled=True, max_depth=3, fontsize=8)
plt.savefig("arvore.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## 6. Análise crítica dos resultados
#
# **Resultados (30% de teste, estratificado):**
# - Árvore sem poda: sensibilidade ≈ 14% e especificidade ≈ 88%. Decorou o treino (overfitting) e quase não detecta fraudes.
# - Árvore ajustada (max_depth=3, min_samples_leaf=100, class_weight="balanced"):
#   sensibilidade ≈ 48,5% (208 de 429 fraudes detectadas) e especificidade ≈ 73,9% (906 falsos alarmes em 3.471 transações normais).
#   Precisão da classe Fraude ≈ 19%.
#
# **Interpretação:**
# - O sinal mais forte é o horário: transações entre 23h e 4h59 têm ~18% de fraude contra ~8% no resto do dia.
#   A variável "madrugada" responde por ~78% da importância da árvore. Valores da transação e tipo (PIX levemente
#   mais fraudado) contribuem pouco.
# - "Valor Anterior" e "Valor Após" são redundantes: Anterior − Após = Valor da Transação em 100% das linhas.
#   Não trazem informação nova e podem confundir a árvore.
# - Não houve valores ausentes. O balanço entre sensibilidade e especificidade foi ajustado com class_weight e
#   escolha de hiperparâmetros por balanced_accuracy (média das duas métricas), refletindo o equilíbrio pedido pelo banco.
#
# **Limitações:**
# - Desempenho modesto: o modelo erra muito (cerca de 4 em 5 alertas são falsos alarmes), o que prejudicaria a
#   experiência do cliente; e ainda deixa passar ~51% das fraudes. As variáveis disponíveis têm pouco poder
#   preditivo, e uma árvore única é limitada.
# - Classes desbalanceadas (~11% de fraude) e um único split; os resultados podem variar com outra semente (use validação cruzada).
# - O dataset parece ter pouca relação entre valor/tipo e fraude, então o resultado não precisa ser alto, como o próprio desafio admite.
#
# **Melhorias possíveis:**
# - Criar variáveis de comportamento por cliente (frequência de transações, desvio em relação ao valor habitual,
#   intervalo entre transações, dia da semana, saldo restante / saldo anterior).
# - Testar Random Forest ou Gradient Boosting e ajustar o limiar de decisão conforme o custo de uma fraude não detectada
#   versus um falso alarme; usar curva ROC/PR para escolher o limiar.
# - Obter dados adicionais (localização, dispositivo, destinatário).
