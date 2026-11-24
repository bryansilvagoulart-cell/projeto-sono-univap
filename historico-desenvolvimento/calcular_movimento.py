import pandas as pd
import numpy as np


# ==========================================================
# 1. CARREGAR O DATASET DE MOVIMENTO
# ==========================================================

arquivo = "data/raw/amostra/motion.csv"

df = pd.read_csv(arquivo)

print("=" * 60)
print("ANÁLISE DO MOVIMENTO")
print("=" * 60)

print("\nQuantidade de medições:", len(df))

print("\nColunas:")
print(df.columns.tolist())


# ==========================================================
# 2. CONVERTER X, Y E Z PARA NÚMEROS
# ==========================================================

df["x"] = pd.to_numeric(df["x"])
df["y"] = pd.to_numeric(df["y"])
df["z"] = pd.to_numeric(df["z"])


# ==========================================================
# 3. CALCULAR A MAGNITUDE DO ACELERÔMETRO
# ==========================================================

df["magnitude"] = np.sqrt(
    df["x"] ** 2 +
    df["y"] ** 2 +
    df["z"] ** 2
)


print("\n" + "=" * 60)
print("MAGNITUDE DO ACELERÔMETRO")
print("=" * 60)

print(df["magnitude"].describe())


# ==========================================================
# 4. CALCULAR A VARIAÇÃO ENTRE MEDIÇÕES
# ==========================================================

df["movimento"] = df["magnitude"].diff().abs()


print("\n" + "=" * 60)
print("VARIAÇÃO DO MOVIMENTO")
print("=" * 60)

print(df["movimento"].describe())


# ==========================================================
# 5. REMOVER O PRIMEIRO VALOR
# ==========================================================

df = df.dropna(subset=["movimento"])


# ==========================================================
# 6. RESULTADO
# ==========================================================

print("\n" + "=" * 60)
print("PRIMEIRAS MEDIÇÕES")
print("=" * 60)

print(
    df[
        [
            "Timestamp",
            "x",
            "y",
            "z",
            "magnitude",
            "movimento"
        ]
    ].head(10)
)


# ==========================================================
# 7. SALVAR
# ==========================================================

arquivo_saida = "data/processed/movimento_calculado.csv"

df.to_csv(
    arquivo_saida,
    index=False
)


print("\n" + "=" * 60)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 60)

print("\nArquivo salvo em:")
print(arquivo_saida)