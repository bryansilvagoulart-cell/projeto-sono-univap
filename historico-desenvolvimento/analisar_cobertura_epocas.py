from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PARTICIPANTES = [
    "Bidslab02",
    "Bidslab07",
    "Bidslab09",
    "Bidslab11",
    "Bidslab13",
    "Bidslab14",
    "Bidslab16",
    "Bidslab17",
    "Bidslab18",
    "Bidslab19",
    "Bidslab20",
    "Bidslab22",
    "Bidslab30",
    "Bidslab31",
    "Bidslab32"
]

NOITES = [1, 2, 3, 4, 5]

PASTA = Path(
    "data/processed/participantes"
)


# ============================================================
# LISTAS PARA JUNTAR OS DADOS
# ============================================================

dados_hr = []
dados_motion = []

total_epocas = 0


print("=" * 70)
print("ANÁLISE DE COBERTURA DAS ÉPOCAS")
print("=" * 70)


# ============================================================
# LER AS 75 NOITES
# ============================================================

for participante in PARTICIPANTES:

    for noite in NOITES:

        arquivo = (
            PASTA
            / participante
            / f"noite_{noite}.csv"
        )

        df = pd.read_csv(
            arquivo,
            usecols=[
                "participante",
                "noite",
                "epoca",
                "hr_samples",
                "motion_samples"
            ]
        )

        total_epocas += len(df)

        dados_hr.extend(
            df["hr_samples"].tolist()
        )

        dados_motion.extend(
            df["motion_samples"].tolist()
        )


# ============================================================
# TRANSFORMAR EM SERIES
# ============================================================

hr = pd.Series(
    dados_hr,
    dtype="float64"
)

motion = pd.Series(
    dados_motion,
    dtype="float64"
)


# ============================================================
# RESUMO
# ============================================================

print(
    f"\nTotal de épocas analisadas: "
    f"{total_epocas}"
)


print("\n" + "=" * 70)
print("FREQUÊNCIA CARDÍACA")
print("=" * 70)

print(
    hr.describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


print("\n" + "=" * 70)
print("ACELERÔMETRO")
print("=" * 70)

print(
    motion.describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# VERIFICAR POSSÍVEIS ÉPOCAS INCOMPLETAS
# ============================================================

print("\n" + "=" * 70)
print("QUANTIDADE DE ÉPOCAS POR NÚMERO DE AMOSTRAS")
print("=" * 70)


# Acelerômetro esperado:
# aproximadamente 50 Hz durante 30 segundos
# ≈ 1500 amostras.

limites_motion = [
    1500,
    1400,
    1200,
    1000,
    750,
    500
]


print("\nAcelerômetro:")

for limite in limites_motion:

    quantidade = int(
        (motion < limite).sum()
    )

    percentual = (
        quantidade
        / total_epocas
        * 100
    )

    print(
        f"Menos de {limite:4d} amostras: "
        f"{quantidade:6d} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# HR
# ============================================================

print("\nFrequência cardíaca:")


limites_hr = [
    6,
    5,
    4,
    3,
    2
]


for limite in limites_hr:

    quantidade = int(
        (hr < limite).sum()
    )

    percentual = (
        quantidade
        / total_epocas
        * 100
    )

    print(
        f"Menos de {limite} amostras: "
        f"{quantidade:6d} "
        f"({percentual:.2f}%)"
    )


print("\n" + "=" * 70)
print("ANÁLISE CONCLUÍDA")
print("=" * 70)