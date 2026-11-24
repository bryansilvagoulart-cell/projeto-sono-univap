import pandas as pd
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = "data/processed/dataset_final.csv"
SAIDA = "data/processed/indicadores_proprios.csv"

print("=" * 60)
print("INDICADORES PRÓPRIOS DE ANÁLISE DO SONO")
print("=" * 60)

# ============================================================
# CARREGAMENTO
# ============================================================

df = pd.read_csv(ARQUIVO)

print(f"\nTotal de épocas: {len(df)}")

# ============================================================
# 1. ÍNDICE DE CONTINUIDADE
# ============================================================

print("\n" + "=" * 60)
print("1. ÍNDICE DE CONTINUIDADE")
print("=" * 60)

# Identifica quando ocorre uma mudança de estágio
transicoes = (df["label"] != df["label"].shift()).sum() - 1

# Quantidade de épocas
total_epocas = len(df)

# Quanto menos transições, maior a continuidade.
# A fórmula transforma isso em uma escala de 0 a 100.
continuidade = max(
    0,
    min(
        100,
        100 * (1 - (transicoes / total_epocas))
    )
)

print(f"Transições: {transicoes}")
print(f"Índice de continuidade: {continuidade:.2f}/100")

# ============================================================
# 2. ÍNDICE DE ESTABILIDADE CARDÍACA
# ============================================================

print("\n" + "=" * 60)
print("2. ÍNDICE DE ESTABILIDADE CARDÍACA")
print("=" * 60)

# Calculamos quanto a frequência cardíaca muda
# de uma época para a próxima.

variacao_hr = df["hr_mean"].diff().abs()

variacao_media_hr = variacao_hr.mean()

# Quanto menor a variação, maior a estabilidade.
# Usamos 10 bpm como referência para normalização.
estabilidade_cardiaca = max(
    0,
    min(
        100,
        100 * (1 - (variacao_media_hr / 10))
    )
)

print(f"Variação média da FC: {variacao_media_hr:.2f} bpm")
print(f"Índice de estabilidade cardíaca: {estabilidade_cardiaca:.2f}/100")

# ============================================================
# 3. ÍNDICE DE MOVIMENTAÇÃO
# ============================================================

print("\n" + "=" * 60)
print("3. ÍNDICE DE MOVIMENTAÇÃO")
print("=" * 60)

movimento_medio = df["motion_mean"].mean()

print(f"Movimento médio: {movimento_medio:.6f}")

# Quanto menor o movimento, maior o índice de estabilidade.
# O valor 0.02 é utilizado como referência experimental.
indice_movimento = max(
    0,
    min(
        100,
        100 * (1 - (movimento_medio / 0.02))
    )
)

print(f"Índice de movimentação: {indice_movimento:.2f}/100")

# ============================================================
# 4. ÍNDICE GERAL EXPERIMENTAL
# ============================================================

print("\n" + "=" * 60)
print("4. ÍNDICE GERAL EXPERIMENTAL")
print("=" * 60)

# Combinação dos três indicadores.
#
# Continuidade:       40%
# Estabilidade cardíaca: 30%
# Movimento:          30%

indice_geral = (
    continuidade * 0.40
    + estabilidade_cardiaca * 0.30
    + indice_movimento * 0.30
)

indice_geral = max(0, min(100, indice_geral))

print(f"Índice geral experimental: {indice_geral:.2f}/100")

# ============================================================
# CLASSIFICAÇÃO EXPERIMENTAL
# ============================================================

if indice_geral >= 80:
    classificacao = "Alta estabilidade"
elif indice_geral >= 60:
    classificacao = "Boa estabilidade"
elif indice_geral >= 40:
    classificacao = "Estabilidade moderada"
else:
    classificacao = "Baixa estabilidade"

print(f"Classificação: {classificacao}")

# ============================================================
# CRIAÇÃO DO RESULTADO
# ============================================================

resultado = pd.DataFrame({
    "indice_continuidade": [continuidade],
    "indice_estabilidade_cardiaca": [estabilidade_cardiaca],
    "indice_movimento": [indice_movimento],
    "indice_geral": [indice_geral],
    "classificacao": [classificacao]
})

resultado.to_csv(SAIDA, index=False)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 60)

print("\nResultado:")
print(resultado.to_string(index=False))

print(f"\nArquivo salvo em:")
print(SAIDA)