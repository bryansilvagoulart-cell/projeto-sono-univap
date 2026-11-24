import pandas as pd
import numpy as np

print("=" * 60)
print("ANÁLISE COMPORTAMENTAL DO SONO")
print("=" * 60)

# ============================================================
# 1. CARREGAR DATASET
# ============================================================

arquivo = "data/processed/dataset_final.csv"

df = pd.read_csv(arquivo)

df["inicio"] = pd.to_datetime(df["inicio"], utc=True)
df["fim"] = pd.to_datetime(df["fim"], utc=True)

mapa_estagios = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

df["estagio"] = df["label"].map(mapa_estagios)

print(f"\nTotal de épocas: {len(df)}")

# ============================================================
# 2. TRANSIÇÕES ENTRE ESTÁGIOS
# ============================================================

df["estagio_anterior"] = df["estagio"].shift(1)

transicoes = df[
    df["estagio"] != df["estagio_anterior"]
].copy()

# Ignorar a primeira linha, pois ela não possui estágio anterior
transicoes = transicoes.iloc[1:]

print("\n" + "=" * 60)
print("TRANSIÇÕES ENTRE ESTÁGIOS")
print("=" * 60)

print(f"Quantidade de transições: {len(transicoes)}")

print("\nPrimeiras transições:")

for _, linha in transicoes.head(15).iterrows():

    anterior = linha["estagio_anterior"]
    atual = linha["estagio"]

    horario = linha["inicio"]

    print(
        f"{horario} | "
        f"{anterior} → {atual}"
    )

# ============================================================
# 3. QUANTIDADE DE DESPERTARES
# ============================================================

# Uma entrada em Wake depois de um estágio de sono
# é considerada aqui como um possível despertar.

despertares = df[
    (df["estagio"] == "Wake") &
    (df["estagio_anterior"].isin(["N1", "N2", "N3", "REM"]))
]

print("\n" + "=" * 60)
print("DESPERTARES")
print("=" * 60)

print(f"Possíveis despertares: {len(despertares)}")

# ============================================================
# 4. DURAÇÃO CONTÍNUA DOS ESTÁGIOS
# ============================================================

df["grupo_estagio"] = (
    df["estagio"] != df["estagio_anterior"]
).cumsum()

blocos = (
    df.groupby(["grupo_estagio", "estagio"], sort=False)
    .agg(
        inicio=("inicio", "first"),
        fim=("fim", "last"),
        epocas=("epoca", "count")
    )
    .reset_index()
)

blocos["duracao_minutos"] = (
    blocos["epocas"] * 30 / 60
)

print("\n" + "=" * 60)
print("BLOCOS CONTÍNUOS DE SONO")
print("=" * 60)

print(f"Quantidade de blocos: {len(blocos)}")

print("\nPrimeiros blocos:")

print(
    blocos[
        ["estagio", "inicio", "fim", "epocas", "duracao_minutos"]
    ].head(15).to_string(index=False)
)

# ============================================================
# 5. MAIOR BLOCO CONTÍNUO DE CADA ESTÁGIO
# ============================================================

print("\n" + "=" * 60)
print("MAIOR PERÍODO CONTÍNUO POR ESTÁGIO")
print("=" * 60)

for estagio in ["Wake", "N1", "N2", "N3", "REM"]:

    blocos_estagio = blocos[
        blocos["estagio"] == estagio
    ]

    if len(blocos_estagio) > 0:

        maior = blocos_estagio.loc[
            blocos_estagio["duracao_minutos"].idxmax()
        ]

        print(
            f"{estagio}: "
            f"{maior['duracao_minutos']:.1f} minutos"
        )

# ============================================================
# 6. VARIAÇÃO DA FREQUÊNCIA CARDÍACA
# ============================================================

print("\n" + "=" * 60)
print("VARIAÇÃO DA FREQUÊNCIA CARDÍACA")
print("=" * 60)

df["hr_variacao"] = df["hr_mean"].diff().abs()

print(
    f"Variação média entre épocas: "
    f"{df['hr_variacao'].mean():.2f} bpm"
)

print(
    f"Maior variação entre épocas: "
    f"{df['hr_variacao'].max():.2f} bpm"
)

# ============================================================
# 7. PICO DE MOVIMENTO
# ============================================================

print("\n" + "=" * 60)
print("PICOS DE MOVIMENTO")
print("=" * 60)

# Percentil 95 usado como referência para movimentos elevados
limite_movimento = df["motion_mean"].quantile(0.95)

epocas_movimento_alto = df[
    df["motion_mean"] >= limite_movimento
]

print(
    f"Limite de movimento elevado (P95): "
    f"{limite_movimento:.6f}"
)

print(
    f"Épocas com movimento elevado: "
    f"{len(epocas_movimento_alto)}"
)

# ============================================================
# 8. MOVIMENTO POR ESTÁGIO
# ============================================================

print("\n" + "=" * 60)
print("MOVIMENTO POR ESTÁGIO")
print("=" * 60)

movimento_estagio = (
    df.groupby("estagio")["motion_mean"]
    .agg(["mean", "std", "max"])
)

movimento_estagio.columns = [
    "media",
    "desvio_padrao",
    "maximo"
]

print(movimento_estagio.round(6))

# ============================================================
# 9. FREQUÊNCIA CARDÍACA POR ESTÁGIO
# ============================================================

print("\n" + "=" * 60)
print("FREQUÊNCIA CARDÍACA POR ESTÁGIO")
print("=" * 60)

hr_estagio = (
    df.groupby("estagio")["hr_mean"]
    .agg(["mean", "std", "min", "max"])
)

hr_estagio.columns = [
    "media",
    "desvio_padrao",
    "minimo",
    "maximo"
]

print(hr_estagio.round(2))

# ============================================================
# 10. RESUMO FINAL
# ============================================================

print("\n" + "=" * 60)
print("RESUMO COMPORTAMENTAL")
print("=" * 60)

print(f"Total de transições: {len(transicoes)}")
print(f"Possíveis despertares: {len(despertares)}")
print(f"Blocos contínuos: {len(blocos)}")
print(
    f"Variação média da FC: "
    f"{df['hr_variacao'].mean():.2f} bpm"
)
print(
    f"Épocas com movimento elevado: "
    f"{len(epocas_movimento_alto)}"
)

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA")
print("=" * 60)