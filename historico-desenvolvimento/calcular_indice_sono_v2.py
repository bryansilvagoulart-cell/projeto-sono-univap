import pandas as pd

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = "data/processed/dataset_final.csv"
SAIDA = "data/processed/indice_sono_v2.csv"

print("=" * 60)
print("ÍNDICE EXPERIMENTAL DE QUALIDADE DO SONO - V2")
print("=" * 60)

# ============================================================
# CARREGAMENTO
# ============================================================

df = pd.read_csv(ARQUIVO)

print(f"\nTotal de épocas: {len(df)}")

# ============================================================
# 1. CONTINUIDADE
# ============================================================

transicoes = (df["label"] != df["label"].shift()).sum() - 1

indice_continuidade = max(
    0,
    min(
        100,
        100 * (1 - transicoes / len(df))
    )
)

print("\n" + "=" * 60)
print("1. CONTINUIDADE")
print("=" * 60)

print(f"Transições: {transicoes}")
print(f"Índice: {indice_continuidade:.2f}/100")

# ============================================================
# 2. ESTABILIDADE CARDÍACA
# ============================================================

variacao_hr = df["hr_mean"].diff().abs()
variacao_media_hr = variacao_hr.mean()

indice_cardiaco = max(
    0,
    min(
        100,
        100 * (1 - variacao_media_hr / 10)
    )
)

print("\n" + "=" * 60)
print("2. ESTABILIDADE CARDÍACA")
print("=" * 60)

print(f"Variação média da FC: {variacao_media_hr:.2f} bpm")
print(f"Índice: {indice_cardiaco:.2f}/100")

# ============================================================
# 3. MOVIMENTAÇÃO
# ============================================================

movimento_medio = df["motion_mean"].mean()

indice_movimento = max(
    0,
    min(
        100,
        100 * (1 - movimento_medio / 0.02)
    )
)

print("\n" + "=" * 60)
print("3. MOVIMENTAÇÃO")
print("=" * 60)

print(f"Movimento médio: {movimento_medio:.6f}")
print(f"Índice: {indice_movimento:.2f}/100")

# ============================================================
# 4. ESTRUTURA DO SONO
# ============================================================

print("\n" + "=" * 60)
print("4. ESTRUTURA DO SONO")
print("=" * 60)

# Quantidade de cada estágio
contagem = df["label"].value_counts()

wake = contagem.get(0, 0)
n1 = contagem.get(1, 0)
n2 = contagem.get(2, 0)
n3 = contagem.get(3, 0)
rem = contagem.get(4, 0)

total = len(df)

# Porcentagens
wake_pct = wake / total * 100
n1_pct = n1 / total * 100
n2_pct = n2 / total * 100
n3_pct = n3 / total * 100
rem_pct = rem / total * 100

print(f"Wake: {wake_pct:.2f}%")
print(f"N1:   {n1_pct:.2f}%")
print(f"N2:   {n2_pct:.2f}%")
print(f"N3:   {n3_pct:.2f}%")
print(f"REM:  {rem_pct:.2f}%")

# ------------------------------------------------------------
# Pontuação experimental da estrutura
# ------------------------------------------------------------
#
# A ideia aqui não é dizer qual porcentagem é "saudável".
# Estamos apenas criando uma métrica experimental para o projeto.
#
# N3 e REM recebem maior importância.
# Wake recebe uma penalização.
#
# Os valores de referência são parâmetros experimentais
# definidos pelo projeto.

pontuacao_n3 = min(100, (n3_pct / 25) * 100)
pontuacao_rem = min(100, (rem_pct / 20) * 100)

penalidade_wake = min(100, wake_pct * 5)

indice_estrutura = (
    pontuacao_n3 * 0.45
    + pontuacao_rem * 0.35
    + (100 - penalidade_wake) * 0.20
)

indice_estrutura = max(
    0,
    min(100, indice_estrutura)
)

print(f"\nPontuação N3: {pontuacao_n3:.2f}/100")
print(f"Pontuação REM: {pontuacao_rem:.2f}/100")
print(f"Penalidade Wake: {penalidade_wake:.2f}")

print(f"\nÍndice de estrutura: {indice_estrutura:.2f}/100")

# ============================================================
# 5. ÍNDICE GERAL V2
# ============================================================

print("\n" + "=" * 60)
print("5. ÍNDICE GERAL V2")
print("=" * 60)

indice_geral = (
    indice_continuidade * 0.25
    + indice_cardiaco * 0.25
    + indice_movimento * 0.20
    + indice_estrutura * 0.30
)

indice_geral = max(
    0,
    min(100, indice_geral)
)

# ============================================================
# CLASSIFICAÇÃO
# ============================================================

if indice_geral >= 80:
    classificacao = "Alta estabilidade"
elif indice_geral >= 60:
    classificacao = "Boa estabilidade"
elif indice_geral >= 40:
    classificacao = "Estabilidade moderada"
else:
    classificacao = "Baixa estabilidade"

print(f"Continuidade:          {indice_continuidade:.2f}")
print(f"Estabilidade cardíaca: {indice_cardiaco:.2f}")
print(f"Movimentação:           {indice_movimento:.2f}")
print(f"Estrutura do sono:      {indice_estrutura:.2f}")

print("\n" + "-" * 60)

print(f"ÍNDICE GERAL V2: {indice_geral:.2f}/100")
print(f"CLASSIFICAÇÃO: {classificacao}")

# ============================================================
# SALVAR RESULTADO
# ============================================================

resultado = pd.DataFrame({
    "indice_continuidade": [indice_continuidade],
    "indice_cardiaco": [indice_cardiaco],
    "indice_movimento": [indice_movimento],
    "indice_estrutura": [indice_estrutura],
    "indice_geral": [indice_geral],
    "classificacao": [classificacao]
})

resultado.to_csv(SAIDA, index=False)

print("\n" + "=" * 60)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 60)

print("\nResultado:")
print(resultado.to_string(index=False))

print(f"\nArquivo salvo em:")
print(SAIDA)