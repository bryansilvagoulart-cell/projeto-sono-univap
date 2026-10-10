from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PARTICIPANTES_TESTE = [
    "Bidslab01",
    "Bidslab35",
    "Bidslab47"
]

NOITES_TESTE = [
    1,
    2,
    3,
    4,
    5
]


# ============================================================
# REGRAS DE QUALIDADE CONGELADAS
# ============================================================

COBERTURA_MINIMA = 95.0

MAIOR_GAP_MAXIMO = 1.0

HR_SAMPLES_MINIMO = 4


# ============================================================
# PASTAS
# ============================================================

PASTA_RAW = Path(
    "data/raw/participantes"
)

PASTA_BASE = Path(
    "data/processed/participantes"
)

PASTA_SAIDA = Path(
    "data/processed/teste_final_4_classes"
)

PASTA_QC_DETALHADO = (
    PASTA_SAIDA
    / "qc_detalhado"
)


ARQUIVO_DATASET_FINAL = (
    PASTA_SAIDA
    / "dataset_teste_final_qc.csv"
)

ARQUIVO_RESUMO = (
    PASTA_SAIDA
    / "resumo_qc_teste_final.csv"
)


# ============================================================
# DATASET ANTIGO PARA AUTOVERIFICAÇÃO
# ============================================================

ARQUIVO_REFERENCIA_DESENVOLVIMENTO = Path(
    "data/processed/dataset_geral_epocas.csv"
)


# ============================================================
# NOITES ANTIGAS ESCOLHIDAS PARA VERIFICAR O FILTRO
# ============================================================
#
# Escolhemos noites que apresentaram comportamentos
# diferentes de amostragem.
#
# Elas NÃO fazem parte do novo teste final.
# ============================================================

NOITES_REFERENCIA = [
    ("Bidslab09", 1),
    ("Bidslab17", 3),
    ("Bidslab31", 5)
]


# ============================================================
# OBTER recStart EM UTC
# ============================================================

def obter_rec_start_timestamp(
    participante,
    noite
):

    arquivo_labels = (
        PASTA_RAW
        / participante
        / f"noite_{noite}"
        / "labels.mat"
    )


    if not arquivo_labels.exists():

        raise FileNotFoundError(
            f"labels.mat não encontrado:\n"
            f"{arquivo_labels}"
        )


    labels = loadmat(
        arquivo_labels
    )


    if "recStart" not in labels:

        raise KeyError(
            f"recStart não encontrado em "
            f"{arquivo_labels}"
        )


    rec_start_texto = (
        labels["recStart"]
        .item()
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


    return (
        rec_start_utc.timestamp()
    )


# ============================================================
# CALCULAR QUALIDADE DE UMA NOITE
# ============================================================

def calcular_qualidade_noite(
    participante,
    noite
):

    arquivo_base = (
        PASTA_BASE
        / participante
        / f"noite_{noite}.csv"
    )


    arquivo_motion = (
        PASTA_RAW
        / participante
        / f"noite_{noite}"
        / "motion.csv"
    )


    if not arquivo_base.exists():

        raise FileNotFoundError(
            f"CSV base não encontrado:\n"
            f"{arquivo_base}"
        )


    if not arquivo_motion.exists():

        raise FileNotFoundError(
            f"motion.csv não encontrado:\n"
            f"{arquivo_motion}"
        )


    # ========================================================
    # 1. CARREGAR DATASET BASE
    # ========================================================

    base = pd.read_csv(
        arquivo_base
    )


    colunas_obrigatorias = {
        "epoca",
        "participante",
        "noite",
        "hr_samples"
    }


    if not colunas_obrigatorias.issubset(
        set(base.columns)
    ):

        raise RuntimeError(
            f"Colunas obrigatórias ausentes em "
            f"{arquivo_base}"
        )


    base["epoca"] = (
        base["epoca"]
        .astype(int)
    )


    # ========================================================
    # 2. PEGAR INÍCIO DA GRAVAÇÃO
    # ========================================================

    rec_start_timestamp = (
        obter_rec_start_timestamp(
            participante,
            noite
        )
    )


    # ========================================================
    # 3. CARREGAR SOMENTE TIMESTAMP DO MOVIMENTO
    # ========================================================

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
    # 4. CALCULAR ÉPOCA DE CADA AMOSTRA
    # ========================================================

    motion["epoca"] = np.floor(
        (
            motion["Timestamp"]
            - rec_start_timestamp
        )
        / 30.0
    ).astype(int)


    # ========================================================
    # 5. MANTER APENAS PERÍODO ÚTIL
    # ========================================================

    menor_epoca = int(
        base["epoca"].min()
    )

    maior_epoca = int(
        base["epoca"].max()
    )


    motion = motion[
        (
            motion["epoca"]
            >= menor_epoca
        )
        &
        (
            motion["epoca"]
            <= maior_epoca
        )
    ].copy()


    # ========================================================
    # 6. ORDENAR TIMESTAMPS
    # ========================================================

    motion = motion.sort_values(
        [
            "epoca",
            "Timestamp"
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # 7. CALCULAR INTERVALO ENTRE AMOSTRAS
    # ========================================================

    motion["gap"] = (
        motion
        .groupby(
            "epoca"
        )[
            "Timestamp"
        ]
        .diff()
    )


    # ========================================================
    # 8. RESUMIR CADA ÉPOCA
    # ========================================================

    qualidade_motion = (
        motion
        .groupby(
            "epoca"
        )
        .agg(

            primeiro_timestamp=(
                "Timestamp",
                "min"
            ),

            ultimo_timestamp=(
                "Timestamp",
                "max"
            ),

            motion_samples_temporal=(
                "Timestamp",
                "count"
            ),

            maior_gap=(
                "gap",
                "max"
            )

        )
        .reset_index()
    )


    # ========================================================
    # 9. COBERTURA TEMPORAL
    # ========================================================
    #
    # Exemplo:
    #
    # se as amostras cobrem praticamente todos os
    # 30 segundos da época, cobertura ≈ 100%.
    # ========================================================

    qualidade_motion[
        "duracao_coberta"
    ] = (

        qualidade_motion[
            "ultimo_timestamp"
        ]

        -

        qualidade_motion[
            "primeiro_timestamp"
        ]
    )


    qualidade_motion[
        "cobertura_temporal_pct"
    ] = (

        qualidade_motion[
            "duracao_coberta"
        ]

        / 30.0

        * 100.0
    )


    # ========================================================
    # 10. JUNTAR COM O DATASET BASE
    # ========================================================

    resultado = base.merge(

        qualidade_motion[
            [
                "epoca",
                "motion_samples_temporal",
                "duracao_coberta",
                "cobertura_temporal_pct",
                "maior_gap"
            ]
        ],

        on="epoca",

        how="left"
    )


    # ========================================================
    # 11. CRIAR AS TRÊS REGRAS DE FALHA
    # ========================================================

    resultado[
        "falha_hr"
    ] = (

        resultado[
            "hr_samples"
        ]
        < HR_SAMPLES_MINIMO
    )


    resultado[
        "falha_cobertura"
    ] = (

        resultado[
            "cobertura_temporal_pct"
        ].isna()

        |

        (
            resultado[
                "cobertura_temporal_pct"
            ]
            < COBERTURA_MINIMA
        )
    )


    resultado[
        "falha_gap"
    ] = (

        resultado[
            "maior_gap"
        ].isna()

        |

        (
            resultado[
                "maior_gap"
            ]
            > MAIOR_GAP_MAXIMO
        )
    )


    # ========================================================
    # 12. RESULTADO DO CONTROLE DE QUALIDADE
    # ========================================================

    resultado[
        "qualidade_ok"
    ] = ~(

        resultado[
            "falha_hr"
        ]

        |

        resultado[
            "falha_cobertura"
        ]

        |

        resultado[
            "falha_gap"
        ]

    )


    return resultado


# ============================================================
# VALIDAR NOSSO FILTRO COM O DESENVOLVIMENTO ANTIGO
# ============================================================

def validar_implementacao():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AUTOVERIFICAÇÃO DO FILTRO"
    )

    print(
        "=" * 100
    )


    if not (
        ARQUIVO_REFERENCIA_DESENVOLVIMENTO.exists()
    ):

        raise FileNotFoundError(
            "Não encontrei o dataset de referência:\n"
            f"{ARQUIVO_REFERENCIA_DESENVOLVIMENTO}"
        )


    referencia = pd.read_csv(
        ARQUIVO_REFERENCIA_DESENVOLVIMENTO
    )


    colunas_referencia = {
        "participante",
        "noite",
        "epoca"
    }


    if not colunas_referencia.issubset(
        set(
            referencia.columns
        )
    ):

        raise RuntimeError(
            "O dataset de referência não possui "
            "as colunas esperadas."
        )


    for (
        participante,
        noite
    ) in NOITES_REFERENCIA:

        print(
            f"\nVerificando "
            f"{participante} / noite {noite}..."
        )


        recalculado = (
            calcular_qualidade_noite(
                participante,
                noite
            )
        )


        epocas_recalculadas = set(

            recalculado.loc[
                recalculado[
                    "qualidade_ok"
                ],
                "epoca"
            ]
            .astype(int)
            .tolist()

        )


        epocas_referencia = set(

            referencia[
                (
                    referencia[
                        "participante"
                    ]
                    == participante
                )
                &
                (
                    referencia[
                        "noite"
                    ]
                    == noite
                )
            ][
                "epoca"
            ]
            .astype(int)
            .tolist()

        )


        if (
            epocas_recalculadas
            != epocas_referencia
        ):

            faltando = sorted(
                epocas_referencia
                - epocas_recalculadas
            )


            extras = sorted(
                epocas_recalculadas
                - epocas_referencia
            )


            print(
                "\n❌ O filtro não reproduziu "
                "o desenvolvimento."
            )


            print(
                f"Referência: "
                f"{len(epocas_referencia)} épocas"
            )


            print(
                f"Recalculado: "
                f"{len(epocas_recalculadas)} épocas"
            )


            print(
                f"Faltando no recalculado: "
                f"{faltando[:10]}"
            )


            print(
                f"Extras no recalculado: "
                f"{extras[:10]}"
            )


            raise RuntimeError(
                "AUTOVERIFICAÇÃO FALHOU. "
                "O teste final NÃO será processado."
            )


        print(
            f"✅ Coincidência exata: "
            f"{len(epocas_referencia)} épocas."
        )


    print(
        "\n✅ O filtro reproduziu corretamente "
        "as noites de desenvolvimento escolhidas."
    )


# ============================================================
# PROCESSAR TESTE FINAL
# ============================================================

def processar_teste_final():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONTROLE DE QUALIDADE DO TESTE FINAL"
    )

    print(
        "=" * 100
    )


    PASTA_QC_DETALHADO.mkdir(
        parents=True,
        exist_ok=True
    )


    datasets_validos = []

    resumos = []


    contador = 0

    total_noites = (
        len(PARTICIPANTES_TESTE)
        * len(NOITES_TESTE)
    )


    for participante in PARTICIPANTES_TESTE:

        pasta_participante = (
            PASTA_QC_DETALHADO
            / participante
        )


        pasta_participante.mkdir(
            parents=True,
            exist_ok=True
        )


        for noite in NOITES_TESTE:

            contador += 1


            print(
                "\n"
                + "-" * 100
            )


            print(
                f"[{contador:02d}/{total_noites}] "
                f"{participante} / noite {noite}"
            )


            resultado = (
                calcular_qualidade_noite(
                    participante,
                    noite
                )
            )


            antes = len(
                resultado
            )


            depois = int(
                resultado[
                    "qualidade_ok"
                ].sum()
            )


            removidas = (
                antes
                - depois
            )


            retencao = (
                depois
                / antes
                * 100.0
            )


            falha_hr = int(
                resultado[
                    "falha_hr"
                ].sum()
            )


            falha_cobertura = int(
                resultado[
                    "falha_cobertura"
                ].sum()
            )


            falha_gap = int(
                resultado[
                    "falha_gap"
                ].sum()
            )


            # =================================================
            # SALVAR DIAGNÓSTICO COMPLETO DA NOITE
            # =================================================

            arquivo_detalhado = (

                pasta_participante
                / f"noite_{noite}_qc.csv"

            )


            resultado.to_csv(
                arquivo_detalhado,
                index=False
            )


            # =================================================
            # DATASET APROVADO
            # =================================================

            aprovado = resultado[
                resultado[
                    "qualidade_ok"
                ]
            ].copy()


            # Mantemos somente as métricas de qualidade
            # necessárias no dataset final.
            # Os flags ficam nos arquivos de diagnóstico.

            colunas_remover = [
                "motion_samples_temporal",
                "primeiro_timestamp",
                "ultimo_timestamp",
                "falha_hr",
                "falha_cobertura",
                "falha_gap",
                "qualidade_ok"
            ]


            aprovado = aprovado.drop(
                columns=[
                    coluna

                    for coluna
                    in colunas_remover

                    if coluna
                    in aprovado.columns
                ]
            )


            datasets_validos.append(
                aprovado
            )


            resumos.append(
                {
                    "participante":
                        participante,

                    "noite":
                        noite,

                    "epocas_base":
                        antes,

                    "epocas_aprovadas":
                        depois,

                    "epocas_removidas":
                        removidas,

                    "retencao_pct":
                        round(
                            retencao,
                            4
                        ),

                    "falha_hr":
                        falha_hr,

                    "falha_cobertura":
                        falha_cobertura,

                    "falha_gap":
                        falha_gap
                }
            )


            print(
                f"Épocas base:      {antes}"
            )


            print(
                f"Épocas aprovadas: {depois}"
            )


            print(
                f"Removidas:        {removidas}"
            )


            print(
                f"Retenção:         "
                f"{retencao:.2f}%"
            )


    # ========================================================
    # CONSOLIDAR
    # ========================================================

    dataset_final = pd.concat(
        datasets_validos,
        ignore_index=True
    )


    resumo = pd.DataFrame(
        resumos
    )


    # ========================================================
    # VALIDAR DUPLICATAS
    # ========================================================

    duplicadas = (
        dataset_final
        .duplicated(
            subset=[
                "participante",
                "noite",
                "epoca"
            ]
        )
        .sum()
    )


    if duplicadas > 0:

        raise RuntimeError(
            f"Foram encontradas "
            f"{duplicadas} épocas duplicadas."
        )


    # ========================================================
    # VALIDAR AUSENTES NAS VARIÁVEIS PRINCIPAIS
    # ========================================================

    colunas_principais = [
        "hr_mean",
        "hr_min",
        "hr_max",
        "hr_std",
        "motion_mean",
        "motion_std",
        "motion_max",
        "duracao_coberta",
        "cobertura_temporal_pct",
        "maior_gap"
    ]


    ausentes = (
        dataset_final[
            colunas_principais
        ]
        .isna()
        .sum()
        .sum()
    )


    if ausentes != 0:

        raise RuntimeError(
            f"O dataset final possui "
            f"{ausentes} valores ausentes."
        )


    # ========================================================
    # ORDENAR
    # ========================================================

    dataset_final = (
        dataset_final
        .sort_values(
            [
                "participante",
                "noite",
                "epoca"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    resumo = (
        resumo
        .sort_values(
            [
                "participante",
                "noite"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # SALVAR
    # ========================================================

    dataset_final.to_csv(
        ARQUIVO_DATASET_FINAL,
        index=False
    )


    resumo.to_csv(
        ARQUIVO_RESUMO,
        index=False
    )


    # ========================================================
    # RESUMO FINAL
    # ========================================================

    total_base = int(
        resumo[
            "epocas_base"
        ].sum()
    )


    total_aprovadas = len(
        dataset_final
    )


    total_removidas = (
        total_base
        - total_aprovadas
    )


    retencao_total = (
        total_aprovadas
        / total_base
        * 100.0
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONTROLE DE QUALIDADE CONCLUÍDO"
    )

    print(
        "=" * 100
    )


    print(
        f"\nÉpocas antes do QC: "
        f"{total_base}"
    )


    print(
        f"Épocas aprovadas: "
        f"{total_aprovadas}"
    )


    print(
        f"Épocas removidas: "
        f"{total_removidas}"
    )


    print(
        f"Retenção total: "
        f"{retencao_total:.2f}%"
    )


    print(
        "\nParticipantes:"
    )


    por_participante = (

        dataset_final
        .groupby(
            "participante"
        )
        .size()

    )


    for participante, quantidade in (
        por_participante.items()
    ):

        print(
            f"  {participante}: "
            f"{quantidade} épocas válidas"
        )


    print(
        "\nDataset final após qualidade:"
    )

    print(
        ARQUIVO_DATASET_FINAL
    )


    print(
        "\nResumo por noite:"
    )

    print(
        ARQUIVO_RESUMO
    )


    print(
        "\n✅ Nenhuma acurácia foi calculada."
    )

    print(
        "✅ Nenhuma distribuição de classes "
        "foi analisada."
    )

    print(
        "✅ Nenhum parâmetro do modelo "
        "foi alterado."
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 100
    )

    print(
        "CONTROLE DE QUALIDADE - TESTE FINAL 4 CLASSES"
    )

    print(
        "=" * 100
    )


    print(
        "\nRegras congeladas:"
    )

    print(
        f"  Cobertura temporal >= "
        f"{COBERTURA_MINIMA}%"
    )

    print(
        f"  Maior gap <= "
        f"{MAIOR_GAP_MAXIMO} s"
    )

    print(
        f"  HR samples >= "
        f"{HR_SAMPLES_MINIMO}"
    )


    # Primeiro provamos que o filtro é o mesmo
    # do desenvolvimento.

    validar_implementacao()


    # Somente depois aplicamos aos participantes novos.

    processar_teste_final()