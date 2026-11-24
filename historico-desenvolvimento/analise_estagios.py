import pandas as pd


# ==========================================================
# 1. CARREGAR O DATASET LIMPO
# ==========================================================

arquivo = "data/processed/dataset_limpo.csv"

df = pd.read_csv(arquivo)

print("=" * 60)
print("ANÁLISE DOS ESTÁGIOS DO SONO")
print("=" * 60)

print("\nTotal de épocas:", len(df))


# ==========================================================
# 2. CONVERTER OS LABELS PARA NOMES
# ==========================================================

nomes_estagios = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

df["estagio"] = df["label"].map(nomes_estagios)


# ==========================================================
# 3. QUANTIDADE DE ÉPOCAS POR ESTÁGIO
# ==========================================================

print("\n" + "=" * 60)
print("QUANTIDADE DE ÉPOCAS POR ESTÁGIO")
print("=" * 60)

quantidade = df["estagio"].value_counts()

print(quantidade)


# ==========================================================
# 4. PORCENTAGEM DE CADA ESTÁGIO
# ==========================================================

print("\n" + "=" * 60)
print("PORCENTAGEM DE CADA ESTÁGIO")
print("=" * 60)

porcentagem = (
    df["estagio"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

for estagio, valor in porcentagem.items():
    print(f"{estagio}: {valor}%")


# ==========================================================
# 5. ESTATÍSTICAS DA FREQUÊNCIA CARDÍACA
# ==========================================================

print("\n" + "=" * 60)
print("FREQUÊNCIA CARDÍACA POR ESTÁGIO")
print("=" * 60)

estatisticas_hr = (
    df.groupby("estagio")["hr_mean"]
    .agg(
        media="mean",
        minimo="min",
        maximo="max",
        desvio_padrao="std"
    )
    .round(2)
)

print(estatisticas_hr)


# ==========================================================
# 6. ESTATÍSTICAS DO MOVIMENTO
# ==========================================================

print("\n" + "=" * 60)
print("MOVIMENTO POR ESTÁGIO")
print("=" * 60)

estatisticas_motion = (
    df.groupby("estagio")["motion_mean"]
    .agg(
        media="mean",
        minimo="min",
        maximo="max",
        desvio_padrao="std"
    )
    .round(4)
)

print(estatisticas_motion)


# ==========================================================
# 7. COMPARAÇÃO GERAL
# ==========================================================

print("\n" + "=" * 60)
print("RESUMO COMPARATIVO")
print("=" * 60)

resumo = pd.DataFrame({
    "quantidade_epocas": quantidade,
    "porcentagem": porcentagem,
    "hr_media": df.groupby("estagio")["hr_mean"].mean(),
    "hr_desvio": df.groupby("estagio")["hr_mean"].std(),
    "movimento_medio": df.groupby("estagio")["motion_mean"].mean(),
    "movimento_desvio": df.groupby("estagio")["motion_mean"].std()
})

print(resumo.round(3))


# ==========================================================
# 8. SALVAR RESUMO
# ==========================================================

arquivo_saida = "data/processed/resumo_estagios.csv"

resumo.round(3).to_csv(arquivo_saida)

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA")
print("=" * 60)

print("\nResumo salvo em:")
print(arquivo_saida)