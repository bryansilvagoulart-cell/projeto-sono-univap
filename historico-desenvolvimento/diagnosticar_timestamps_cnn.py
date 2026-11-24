from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
)

ARQUIVO_MOTION = Path(
    "data/raw/participantes/Bidslab02/noite_1/motion.csv"
)


# ============================================================
# INÍCIO
# ============================================================

print("=" * 90)
print("DIAGNÓSTICO DE TIMESTAMPS - DATASET CNN")
print("=" * 90)


# ============================================================
# 1. PRIMEIRA ÉPOCA DO Bidslab02
# ============================================================

dataset = pd.read_csv(
    ARQUIVO_DATASET
)


participante = dataset[
    dataset["participante"] == "Bidslab02"
].copy()


participante = participante.sort_values(
    [
        "noite",
        "epoca"
    ]
)


primeira_epoca = participante.iloc[0]


print(
    "\nPrimeira época encontrada:"
)

print(
    f"Participante: {primeira_epoca['participante']}"
)

print(
    f"Noite: {primeira_epoca['noite']}"
)

print(
    f"Época: {primeira_epoca['epoca']}"
)

print(
    f"Início original: {primeira_epoca['inicio']}"
)


# ============================================================
# 2. CONVERSÃO CORRETA VIA .timestamp()
# ============================================================

inicio_datetime = pd.to_datetime(
    primeira_epoca["inicio"],
    utc=True
)


inicio_timestamp_correto = (
    inicio_datetime.timestamp()
)


print(
    "\n"
    + "=" * 90
)

print(
    "CONVERSÃO DO INÍCIO DA ÉPOCA"
)

print(
    "=" * 90
)


print(
    f"\nDatetime convertido:"
)

print(
    inicio_datetime
)


print(
    f"\nTimestamp Unix usando .timestamp():"
)

print(
    f"{inicio_timestamp_correto:.6f}"
)


# ============================================================
# 3. TESTAR O MÉTODO ANTIGO
# ============================================================

serie_teste = pd.to_datetime(
    pd.Series(
        [
            primeira_epoca["inicio"]
        ]
    ),
    utc=True
)


print(
    "\nTipo interno do pandas:"
)

print(
    serie_teste.dtype
)


valor_inteiro = int(
    serie_teste
    .astype("int64")
    .iloc[0]
)


timestamp_metodo_antigo = (
    valor_inteiro
    / 1_000_000_000.0
)


print(
    "\nValor inteiro interno:"
)

print(
    valor_inteiro
)


print(
    "\nTimestamp calculado pelo método antigo:"
)

print(
    f"{timestamp_metodo_antigo:.6f}"
)


print(
    "\nDiferença entre os dois métodos:"
)

print(
    f"{abs(inicio_timestamp_correto - timestamp_metodo_antigo):,.3f} segundos"
)


# ============================================================
# 4. LER TIMESTAMPS DO MOTION
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "MOTION.CSV"
)

print(
    "=" * 90
)


cabecalho = pd.read_csv(
    ARQUIVO_MOTION,
    nrows=0
)


coluna_timestamp = cabecalho.columns[0]


print(
    f"\nColuna de timestamp:"
)

print(
    coluna_timestamp
)


motion_tempo = pd.read_csv(

    ARQUIVO_MOTION,

    usecols=[
        coluna_timestamp
    ]
)


timestamps = pd.to_numeric(

    motion_tempo[
        coluna_timestamp
    ],

    errors="coerce"

).dropna().to_numpy(
    dtype=np.float64
)


# ============================================================
# 5. NORMALIZAR UNIDADE DO MOTION
# ============================================================

referencia = np.median(
    np.abs(
        timestamps
    )
)


print(
    "\nMediana absoluta do timestamp bruto:"
)

print(
    f"{referencia:.6f}"
)


if referencia > 1e14:

    print(
        "Detectado: microssegundos Unix"
    )

    timestamps = (
        timestamps
        / 1_000_000.0
    )


elif referencia > 1e11:

    print(
        "Detectado: milissegundos Unix"
    )

    timestamps = (
        timestamps
        / 1_000.0
    )


else:

    print(
        "Detectado/assumido: segundos"
    )


# ============================================================
# 6. INTERVALO DO MOTION
# ============================================================

timestamps = timestamps[
    np.isfinite(
        timestamps
    )
]


timestamps = np.sort(
    timestamps
)


motion_inicio = float(
    timestamps[0]
)


motion_fim = float(
    timestamps[-1]
)


print(
    "\nPrimeiro timestamp motion:"
)

print(
    f"{motion_inicio:.6f}"
)


print(
    "\nÚltimo timestamp motion:"
)

print(
    f"{motion_fim:.6f}"
)


print(
    "\nDuração coberta pelo motion:"
)

print(
    f"{(motion_fim - motion_inicio) / 3600:.2f} horas"
)


# ============================================================
# 7. COMPARAR ÉPOCA COM MOTION
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "COMPARAÇÃO"
)

print(
    "=" * 90
)


dentro_correto = (
    motion_inicio
    <= inicio_timestamp_correto
    <= motion_fim
)


dentro_antigo = (
    motion_inicio
    <= timestamp_metodo_antigo
    <= motion_fim
)


print(
    "\nTimestamp correto está dentro do motion?"
)

print(
    dentro_correto
)


print(
    "\nTimestamp do método antigo está dentro do motion?"
)

print(
    dentro_antigo
)


# ============================================================
# 8. DISTÂNCIA PARA O PRIMEIRO PONTO
# ============================================================

print(
    "\nDiferença entre início da época "
    "e primeiro ponto do motion:"
)


print(
    f"{inicio_timestamp_correto - motion_inicio:.3f} segundos"
)


# ============================================================
# 9. QUANTAS AMOSTRAS EXISTEM NA ÉPOCA 0
# ============================================================

fim_epoca = (
    inicio_timestamp_correto
    + 30.0
)


indice_inicio = np.searchsorted(

    timestamps,

    inicio_timestamp_correto,

    side="left"
)


indice_fim = np.searchsorted(

    timestamps,

    fim_epoca,

    side="left"
)


quantidade_epoca = (
    indice_fim
    - indice_inicio
)


print(
    "\nAmostras do motion entre início e +30 segundos:"
)

print(
    quantidade_epoca
)


# ============================================================
# RESULTADO
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "DIAGNÓSTICO FINALIZADO"
)

print(
    "=" * 90
)