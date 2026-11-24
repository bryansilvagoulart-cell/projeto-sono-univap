from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# NOITES QUE QUEREMOS INVESTIGAR
# ============================================================

CASOS = [
    ("Bidslab09", 1),
    ("Bidslab17", 3),
    ("Bidslab31", 3),
    ("Bidslab31", 5)
]

PASTA_RAW = Path(
    "data/raw/participantes"
)


# ============================================================
# FUNÇÃO PARA INVESTIGAR UMA NOITE
# ============================================================

def investigar_noite(
    participante,
    noite
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"{participante} / noite {noite}"
    )

    print(
        "=" * 75
    )


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


    # ========================================================
    # LABELS E HORÁRIO INICIAL
    # ========================================================

    dados_labels = loadmat(
        arquivo_labels
    )


    labels = (
        dados_labels[
            "expert_label"
        ]
        .flatten()
        .astype(int)
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


    # ========================================================
    # CARREGAR SOMENTE O TIMESTAMP DO ACELERÔMETRO
    # ========================================================

    print(
        "\nCarregando timestamps..."
    )


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


    # ========================================================
    # IDENTIFICAR ÉPOCA
    # ========================================================

    motion["epoca"] = np.floor(
        (
            motion["Timestamp"]
            - rec_start_timestamp
        )
        / 30
    ).astype(int)


    motion = motion[
        (
            motion["epoca"] >= 0
        )
        &
        (
            motion["epoca"]
            < len(labels)
        )
    ].copy()


    # ========================================================
    # DIFERENÇA ENTRE AMOSTRAS
    # ========================================================

    motion["gap"] = (
        motion["Timestamp"]
        .diff()
    )


    # Quando muda de época,
    # não queremos considerar o gap
    # entre épocas diferentes.
    mudou_epoca = (
        motion["epoca"]
        != motion["epoca"].shift()
    )


    motion.loc[
        mudou_epoca,
        "gap"
    ] = np.nan


    # ========================================================
    # RESUMO TEMPORAL POR ÉPOCA
    # ========================================================

    resumo = (
        motion
        .groupby("epoca")
        .agg(

            amostras=(
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


    # ========================================================
    # QUANTO DOS 30 SEGUNDOS FOI COBERTO
    # ========================================================

    resumo["duracao_coberta"] = (
        resumo["ultimo_timestamp"]
        - resumo["primeiro_timestamp"]
    )


    resumo["cobertura_temporal_pct"] = (
        resumo["duracao_coberta"]
        / 30
        * 100
    )


    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    print(
        f"\nÉpocas analisadas: "
        f"{len(resumo)}"
    )


    print(
        "\nAMOSTRAS POR ÉPOCA:"
    )

    print(
        resumo[
            "amostras"
        ].describe(
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


    print(
        "\nCOBERTURA TEMPORAL (%):"
    )

    print(
        resumo[
            "cobertura_temporal_pct"
        ].describe(
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


    print(
        "\nMAIOR INTERVALO SEM AMOSTRA "
        "DENTRO DA ÉPOCA (segundos):"
    )

    print(
        resumo[
            "maior_gap"
        ].describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99
            ]
        )
    )


    # ========================================================
    # ÉPOCAS COM POUCAS AMOSTRAS
    # ========================================================

    mediana = (
        resumo[
            "amostras"
        ].median()
    )


    limite_antigo = (
        mediana * 0.80
    )


    poucas_amostras = resumo[
        resumo[
            "amostras"
        ] < limite_antigo
    ].copy()


    print(
        "\n"
        + "-" * 75
    )

    print(
        "ÉPOCAS QUE O NOSSO FILTRO "
        "ANTIGO REMOVERIA"
    )

    print(
        "-" * 75
    )


    print(
        f"\nMediana da noite: "
        f"{mediana:.1f}"
    )


    print(
        f"Limite antigo de 80%: "
        f"{limite_antigo:.1f}"
    )


    print(
        f"Quantidade marcada: "
        f"{len(poucas_amostras)}"
    )


    if len(
        poucas_amostras
    ) > 0:

        print(
            "\nCobertura temporal dessas épocas:"
        )

        print(
            poucas_amostras[
                "cobertura_temporal_pct"
            ].describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90
                ]
            )
        )


        print(
            "\nMaior gap dessas épocas:"
        )

        print(
            poucas_amostras[
                "maior_gap"
            ].describe(
                percentiles=[
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99
                ]
            )
        )


# ============================================================
# EXECUTAR
# ============================================================

print(
    "=" * 75
)

print(
    "INVESTIGAÇÃO DA COBERTURA TEMPORAL"
)

print(
    "=" * 75
)


for participante, noite in CASOS:

    investigar_noite(
        participante,
        noite
    )


print(
    "\n"
    + "=" * 75
)

print(
    "INVESTIGAÇÃO CONCLUÍDA"
)

print(
    "=" * 75
)