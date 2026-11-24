import pandas as pd


# ==========================================================
# 1. CARREGAR DATASET
# ==========================================================

arquivo = "data/processed/dataset_processado.csv"

df = pd.read_csv(arquivo)

print("=" * 60)
print("DIAGNÓSTICO DO DATASET")
print("=" * 60)


# ==========================================================
# 2. INFORMAÇÕES GERAIS
# ==========================================================

print("\nQuantidade de linhas:", len(df))
print("Quantidade de colunas:", len(df.columns))


# ==========================================================
# 3. VALORES AUSENTES
# ==========================================================

print("\n" + "=" * 60)
print("VALORES AUSENTES")
print("=" * 60)

print(df.isna().sum())


# ==========================================================
# 4. ÉPOCAS COM DADOS AUSENTES
# ==========================================================

print("\n" + "=" * 60)
print("ÉPOCAS COM DADOS AUSENTES")
print("=" * 60)

colunas_sinais = [
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",
    "motion_mean",
    "motion_std",
    "motion_max",
    "motion_change_mean"
]

epocas_com_falta = df[df[colunas_sinais].isna().any(axis=1)]

print(epocas_com_falta[
    [
        "epoca",
        "inicio",
        "fim",
        "label"
    ] + colunas_sinais
].to_string(index=False))


# ==========================================================
# 5. QUANTIDADE DE ÉPOCAS COMPLETAS
# ==========================================================

print("\n" + "=" * 60)
print("ÉPOCAS COMPLETAS")
print("=" * 60)

completas = df[colunas_sinais].notna().all(axis=1)

print("Épocas completas:", completas.sum())
print("Épocas com algum dado ausente:", (~completas).sum())


# ==========================================================
# 6. DISTRIBUIÇÃO DOS LABELS
# ==========================================================

print("\n" + "=" * 60)
print("DISTRIBUIÇÃO DOS LABELS")
print("=" * 60)

contagem = df["label"].value_counts().sort_index()

for label, quantidade in contagem.items():

    percentual = quantidade / len(df) * 100

    print(
        f"Label {label}: "
        f"{quantidade} épocas "
        f"({percentual:.2f}%)"
    )


# ==========================================================
# 7. LABELS DAS ÉPOCAS COM DADOS AUSENTES
# ==========================================================

print("\n" + "=" * 60)
print("LABELS DAS ÉPOCAS COM DADOS AUSENTES")
print("=" * 60)

print(
    epocas_com_falta["label"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# 8. INTERVALO TEMPORAL
# ==========================================================

print("\n" + "=" * 60)
print("INTERVALO TEMPORAL")
print("=" * 60)

print("Primeira época:")
print(df.iloc[0]["inicio"])

print("\nÚltima época:")
print(df.iloc[-1]["fim"])


print("\n" + "=" * 60)
print("DIAGNÓSTICO FINALIZADO")
print("=" * 60)