import pandas as pd

# ============================================================
# CÁLCULO DOS INDICADORES DO SONO
# ============================================================

print("=" * 60)
print("INDICADORES DE SONO")
print("=" * 60)

# ------------------------------------------------------------
# 1. CARREGAR DATASET
# ------------------------------------------------------------

arquivo = "data/processed/dataset_final.csv"

df = pd.read_csv(arquivo)

print(f"\nTotal de épocas: {len(df)}")

# ------------------------------------------------------------
# 2. CONVERTER DATAS
# ------------------------------------------------------------

df["inicio"] = pd.to_datetime(df["inicio"], utc=True)
df["fim"] = pd.to_datetime(df["fim"], utc=True)

# Cada época possui 30 segundos
duracao_epoca_segundos = 30

# ------------------------------------------------------------
# 3. TRADUZIR OS LABELS
# ------------------------------------------------------------

mapa_estagios = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

df["estagio"] = df["label"].map(mapa_estagios)

# ------------------------------------------------------------
# 4. DURAÇÃO TOTAL
# ------------------------------------------------------------

duracao_total_segundos = len(df) * duracao_epoca_segundos
duracao_total_horas = duracao_total_segundos / 3600

# ------------------------------------------------------------
# 5. QUANTIDADE DE ÉPOCAS POR ESTÁGIO
# ------------------------------------------------------------

quantidade_estagios = df["estagio"].value_counts()

# Garantir que todos os estágios existam
estagios = ["Wake", "N1", "N2", "N3", "REM"]

for estagio in estagios:
    if estagio not in quantidade_estagios:
        quantidade_estagios[estagio] = 0

# ------------------------------------------------------------
# 6. TEMPO EM CADA ESTÁGIO
# ------------------------------------------------------------

tempo_estagios = {}

for estagio in estagios:
    segundos = quantidade_estagios[estagio] * duracao_epoca_segundos
    minutos = segundos / 60

    tempo_estagios[estagio] = minutos

# ------------------------------------------------------------
# 7. TEMPO DORMINDO
# ------------------------------------------------------------

epocas_dormindo = len(df[df["estagio"] != "Wake"])

tempo_dormindo_minutos = (
    epocas_dormindo * duracao_epoca_segundos / 60
)

# ------------------------------------------------------------
# 8. PERCENTUAIS
# ------------------------------------------------------------

percentuais = {}

for estagio in estagios:
    percentuais[estagio] = (
        quantidade_estagios[estagio] / len(df)
    ) * 100

# ------------------------------------------------------------
# 9. FREQUÊNCIA CARDÍACA
# ------------------------------------------------------------

hr_media = df["hr_mean"].mean()
hr_minima = df["hr_min"].min()
hr_maxima = df["hr_max"].max()

# ------------------------------------------------------------
# 10. MOVIMENTO
# ------------------------------------------------------------

movimento_medio = df["motion_mean"].mean()
movimento_maximo = df["motion_max"].max()

# ------------------------------------------------------------
# 11. EXIBIR RESULTADOS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DURAÇÃO")
print("=" * 60)

print(f"Tempo total analisado: {duracao_total_horas:.2f} horas")
print(f"Tempo total dormindo: {tempo_dormindo_minutos / 60:.2f} horas")

print("\n" + "=" * 60)
print("TEMPO POR ESTÁGIO")
print("=" * 60)

for estagio in estagios:
    print(
        f"{estagio}: "
        f"{tempo_estagios[estagio]:.1f} minutos "
        f"({percentuais[estagio]:.2f}%)"
    )

print("\n" + "=" * 60)
print("FREQUÊNCIA CARDÍACA")
print("=" * 60)

print(f"Média: {hr_media:.2f} bpm")
print(f"Mínima: {hr_minima:.2f} bpm")
print(f"Máxima: {hr_maxima:.2f} bpm")

print("\n" + "=" * 60)
print("MOVIMENTO")
print("=" * 60)

print(f"Movimento médio: {movimento_medio:.6f}")
print(f"Movimento máximo: {movimento_maximo:.6f}")

print("\n" + "=" * 60)
print("QUANTIDADE DE ÉPOCAS")
print("=" * 60)

for estagio in estagios:
    print(f"{estagio}: {quantidade_estagios[estagio]} épocas")

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA")
print("=" * 60)