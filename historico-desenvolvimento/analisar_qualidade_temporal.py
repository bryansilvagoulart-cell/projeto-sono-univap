from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.io import loadmat


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

PASTA_RAW = Path(
    "data/raw/participantes"
)

PASTA_PROCESSADOS = Path(
    "data/processed/participantes"
)

ARQUIVO_SAIDA = Path(
    "data/processed/"
    "qualidade_temporal_epocas.csv"
)


# ============================================================
# LISTA DE RESULTADOS
# ============================================================

resultados = []

contador_noite = 0

total_noites = (
    len(PARTICIPANTES)
    * len(NOITES)
)


print("=" * 75)
print("ANÁLISE TEMPORAL DE TODAS AS ÉPOCAS")
print("=" * 75)


# ============================================================
# PERCORRER TODAS AS NOITES
# ============================================================

for participante in PARTICIPANTES:

    for noite in NOITES:

        contador_noite += 1

        print(
            f"\n[{contador_noite:02d}/{total_noites}] "
            f"{participante} / noite {noite}"
        )


        # ====================================================
        # CAMINHOS
        # ====================================================

        pasta = (
            PASTA_RAW
            / participante
            / f"noite_{noite}"
        )

        arquivo_motion = (
            pasta
            / "motion.csv"
        )

        arquivo_labels = (
            pasta
            / "labels.mat"
        )

        arquivo_processado = (
            PASTA_PROCESSADOS
            / participante
            / f"noite_{noite}.csv"
        )


        # ====================================================
        # LABELS / HORÁRIO INICIAL
        # ====================================================

        dados_labels = loadmat(
            arquivo_labels
        )

        rec_start_texto = (
            dados_labels[
                "recStart"
            ].item()
        )

        rec_start_local = datetime.strptime(
            rec_start_texto,
            "%Y-%m-%d %H:%M:%S"
        )

        rec_start_local = (
            rec_start_local.replace(
                tzinfo=ZoneInfo(
                    "America/New_York"
                )
            )
        )

        rec_start_utc = (
            rec_start_local.astimezone(
                ZoneInfo("UTC")
            )
        )

        rec_start_timestamp = (
            rec_start_utc.timestamp()
        )


        # ====================================================
        # QUAIS ÉPOCAS REALMENTE ESTÃO NO CSV PROCESSADO
        # ====================================================

        processado = pd.read_csv(
            arquivo_processado,
            usecols=[
                "epoca",
                "hr_samples"
            ]
        )

        epocas_validas = set(
            processado[
                "epoca"
            ].astype(int)
        )


        # ====================================================
        # CARREGAR SOMENTE TIMESTAMP DO MOTION
        # ====================================================

        motion = pd.read_csv(
            arquivo_motion,
            usecols=[
                "Timestamp"
            ],
            dtype={
                "Timestamp":
                    "float64"
            }
        )


        # ====================================================
        # IDENTIFICAR ÉPOCA
        # ====================================================

        motion["epoca"] = np.floor(
            (
                motion["Timestamp"]
                - rec_start_timestamp
            )
            / 30
        ).astype(int)


        # Manter somente épocas que já passaram
        # pelo processamento anterior.

        motion = motion[
            motion["epoca"].isin(
                epocas_validas
            )
        ].copy()


        # ====================================================
        # INTERVALO ENTRE AMOSTRAS
        # ====================================================

        motion["gap"] = (
            motion["Timestamp"]
            .diff()
        )

        mudou_epoca = (
            motion["epoca"]
            != motion["epoca"].shift()
        )

        motion.loc[
            mudou_epoca,
            "gap"
        ] = np.nan


        # ====================================================
        # RESUMIR POR ÉPOCA
        # ====================================================

        resumo = (
            motion
            .groupby("epoca")
            .agg(

                motion_samples=(
                    "Timestamp",
                    "count"
                ),

                primeiro_timestamp=(
                    "Timestamp",
                    "min"
                ),

                ultimo_timestamp=(
                    "Timestamp",
                    "max"
                ),

                maior_gap=(
                    "gap",
                    "max"
                )
            )
            .reset_index()
        )


        # ====================================================
        # COBERTURA TEMPORAL
        # ====================================================

        resumo[
            "duracao_coberta"
        ] = (

            resumo[
                "ultimo_timestamp"
            ]

            -

            resumo[
                "primeiro_timestamp"
            ]
        )


        resumo[
            "cobertura_temporal_pct"
        ] = (

            resumo[
                "duracao_coberta"
            ]

            / 30

            * 100
        )


        # ====================================================
        # ADICIONAR HR
        # ====================================================

        resumo = resumo.merge(
            processado,
            on="epoca",
            how="left"
        )


        resumo[
            "participante"
        ] = participante

        resumo[
            "noite"
        ] = noite


        resultados.append(
            resumo
        )


        print(
            f"Épocas analisadas: "
            f"{len(resumo)}"
        )


# ============================================================
# JUNTAR TUDO
# ============================================================

qualidade = pd.concat(
    resultados,
    ignore_index=True
)


# ============================================================
# ORGANIZAR COLUNAS
# ============================================================

qualidade = qualidade[
    [
        "participante",
        "noite",
        "epoca",
        "hr_samples",
        "motion_samples",
        "duracao_coberta",
        "cobertura_temporal_pct",
        "maior_gap"
    ]
]


# ============================================================
# SALVAR
# ============================================================

qualidade.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


# ============================================================
# RESUMO GERAL
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "RESULTADO GERAL"
)

print(
    "=" * 75
)


print(
    f"\nÉpocas analisadas: "
    f"{len(qualidade)}"
)


# ============================================================
# COBERTURA TEMPORAL
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "COBERTURA TEMPORAL (%)"
)

print(
    "=" * 75
)


print(
    qualidade[
        "cobertura_temporal_pct"
    ].describe(
        percentiles=[
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999
        ]
    )
)


# ============================================================
# MAIOR GAP
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "MAIOR GAP DENTRO DA ÉPOCA (SEGUNDOS)"
)

print(
    "=" * 75
)


print(
    qualidade[
        "maior_gap"
    ].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999
        ]
    )
)


# ============================================================
# QUANTIDADE ABAIXO DE DIFERENTES COBERTURAS
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "ÉPOCAS ABAIXO DE DIFERENTES COBERTURAS"
)

print(
    "=" * 75
)


for limite in [
    99,
    95,
    90,
    80,
    70,
    50
]:

    quantidade = int(
        (
            qualidade[
                "cobertura_temporal_pct"
            ]
            < limite
        ).sum()
    )

    percentual = (
        quantidade
        / len(qualidade)
        * 100
    )

    print(
        f"Cobertura < {limite}%: "
        f"{quantidade:6d} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# GAPS
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "ÉPOCAS COM GRANDES INTERVALOS SEM AMOSTRAS"
)

print(
    "=" * 75
)


for limite in [
    0.1,
    0.5,
    1,
    2,
    5,
    10,
    15
]:

    quantidade = int(
        (
            qualidade[
                "maior_gap"
            ]
            > limite
        ).sum()
    )

    percentual = (
        quantidade
        / len(qualidade)
        * 100
    )

    print(
        f"Gap > {limite:4.1f} s: "
        f"{quantidade:6d} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# HR
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "AMOSTRAS DE FREQUÊNCIA CARDÍACA"
)

print(
    "=" * 75
)


for limite in [
    6,
    5,
    4,
    3,
    2
]:

    quantidade = int(
        (
            qualidade[
                "hr_samples"
            ]
            < limite
        ).sum()
    )

    percentual = (
        quantidade
        / len(qualidade)
        * 100
    )

    print(
        f"HR < {limite} amostras: "
        f"{quantidade:6d} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\nArquivo criado:"
)

print(
    ARQUIVO_SAIDA
)


print(
    "\n"
    + "=" * 75
)

print(
    "ANÁLISE CONCLUÍDA"
)

print(
    "=" * 75
)