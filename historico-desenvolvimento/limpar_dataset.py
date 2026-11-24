import pandas as pd


# ==========================================================
# 1. CARREGAR DATASET PROCESSADO
# ==========================================================

arquivo_entrada = "data/processed/dataset_processado.csv"

df = pd.read_csv(arquivo_entrada)

print("=" * 60)
print("LIMPEZA DO DATASET")
print("=" * 60)

print("\nDataset original:")
print("Linhas:", len(df))
print("Colunas:", len(df.columns))


# ==========================================================
# 2. VERIFICAR VALORES AUSENTES ANTES DA LIMPEZA
# ==========================================================

print("\n" + "=" * 60)
print("VALORES AUSENTES ANTES DA LIMPEZA")
print("=" * 60)

print(df.isna().sum())


# ==========================================================
# 3. TRATAR A ÉPOCA 933
# ==========================================================

print("\n" + "=" * 60)
print("TRATAMENTO DA ÉPOCA 933")
print("=" * 60)

if 933 in df["epoca"].values:

    indice_933 = df["epoca"] == 933

    print("Época 933 encontrada.")

    print("\nDados antes do tratamento:")

    print(
        df.loc[
            indice_933,
            [
                "epoca",
                "hr_mean",
                "hr_min",
                "hr_max",
                "hr_std",
                "motion_mean",
                "motion_std",
                "motion_max",
                "motion_change_mean",
                "label"
            ]
        ]
    )

    # Como existe apenas uma medição de HR,
    # não é possível calcular desvio padrão.
    # Nesse caso, não houve variação observada.
    df.loc[indice_933, "hr_std"] = 0

    print("\nhr_std da época 933 definido como 0.")


# ==========================================================
# 4. REMOVER ÉPOCAS SEM NENHUM SINAL
# ==========================================================

print("\n" + "=" * 60)
print("REMOÇÃO DE ÉPOCAS SEM DADOS")
print("=" * 60)

colunas_principais = [
    "hr_mean",
    "motion_mean"
]

antes = len(df)

df = df.dropna(
    subset=colunas_principais
)

depois = len(df)

print("Linhas antes:", antes)
print("Linhas depois:", depois)
print("Linhas removidas:", antes - depois)


# ==========================================================
# 5. VERIFICAR VALORES AUSENTES DEPOIS DA LIMPEZA
# ==========================================================

print("\n" + "=" * 60)
print("VALORES AUSENTES DEPOIS DA LIMPEZA")
print("=" * 60)

print(df.isna().sum())


# ==========================================================
# 6. SALVAR DATASET LIMPO
# ==========================================================

arquivo_saida = "data/processed/dataset_limpo.csv"

df.to_csv(
    arquivo_saida,
    index=False
)


# ==========================================================
# 7. RESULTADO FINAL
# ==========================================================

print("\n" + "=" * 60)
print("LIMPEZA CONCLUÍDA")
print("=" * 60)

print("\nLinhas finais:", len(df))

print("\nPrimeiras linhas:")

print(df.head())

print("\nArquivo salvo em:")

print(arquivo_saida)