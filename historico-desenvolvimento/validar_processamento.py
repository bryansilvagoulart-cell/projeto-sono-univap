from pathlib import Path

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
    "data/processed/controle_qualidade_noites.csv"
)


# ============================================================
# RESULTADOS
# ============================================================

resultados = []

problemas = []


print("=" * 70)
print("VALIDAÇÃO DOS DADOS PROCESSADOS")
print("=" * 70)


contador = 0

total = (
    len(PARTICIPANTES)
    * len(NOITES)
)


# ============================================================
# PERCORRER AS 75 NOITES
# ============================================================

for participante in PARTICIPANTES:

    for noite in NOITES:

        contador += 1

        print(
            f"\n[{contador:02d}/{total}] "
            f"{participante} / noite {noite}"
        )


        # ----------------------------------------------------
        # CAMINHOS
        # ----------------------------------------------------

        arquivo_labels = (
            PASTA_RAW
            / participante
            / f"noite_{noite}"
            / "labels.mat"
        )

        arquivo_processado = (
            PASTA_PROCESSADOS
            / participante
            / f"noite_{noite}.csv"
        )


        try:

            # =================================================
            # 1. LABELS ORIGINAIS
            # =================================================

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


            total_original = len(
                labels
            )


            # =================================================
            # 2. UNKNOWN
            # =================================================

            total_unknown = int(
                (labels == 5).sum()
            )


            # Quantas épocas poderiam
            # ser utilizadas teoricamente.
            total_utilizavel = (
                total_original
                - total_unknown
            )


            # =================================================
            # 3. DATASET PROCESSADO
            # =================================================

            df = pd.read_csv(
                arquivo_processado
            )


            total_final = len(
                df
            )


            # =================================================
            # 4. VERIFICAR NaN
            # =================================================

            total_nan = int(
                df.isna()
                .sum()
                .sum()
            )


            # =================================================
            # 5. VERIFICAR LABELS
            # =================================================

            labels_invalidos = (
                ~df["label"]
                .isin(
                    [0, 1, 2, 3, 4]
                )
            ).sum()


            labels_invalidos = int(
                labels_invalidos
            )


            # =================================================
            # 6. ÉPOCAS PERDIDAS POR SINAL
            # =================================================

            removidas_sinal = (
                total_utilizavel
                - total_final
            )


            # =================================================
            # 7. COBERTURA
            # =================================================

            if total_utilizavel > 0:

                cobertura = (
                    total_final
                    / total_utilizavel
                    * 100
                )

            else:

                cobertura = 0


            # =================================================
            # 8. TEMPO FINAL
            # =================================================

            minutos_validos = (
                total_final
                * 30
                / 60
            )


            horas_validas = (
                minutos_validos
                / 60
            )


            # =================================================
            # 9. DISTRIBUIÇÃO DOS ESTÁGIOS
            # =================================================

            contagens = (
                df["label"]
                .value_counts()
            )


            wake = int(
                contagens.get(
                    0,
                    0
                )
            )

            n1 = int(
                contagens.get(
                    1,
                    0
                )
            )

            n2 = int(
                contagens.get(
                    2,
                    0
                )
            )

            n3 = int(
                contagens.get(
                    3,
                    0
                )
            )

            rem = int(
                contagens.get(
                    4,
                    0
                )
            )


            # =================================================
            # 10. GUARDAR RESULTADO
            # =================================================

            resultados.append({

                "participante":
                    participante,

                "noite":
                    noite,

                "epocas_originais":
                    total_original,

                "unknown":
                    total_unknown,

                "epocas_utilizaveis":
                    total_utilizavel,

                "epocas_finais":
                    total_final,

                "removidas_sinal":
                    removidas_sinal,

                "cobertura_percentual":
                    cobertura,

                "horas_validas":
                    horas_validas,

                "valores_ausentes":
                    total_nan,

                "labels_invalidos":
                    labels_invalidos,

                "wake":
                    wake,

                "n1":
                    n1,

                "n2":
                    n2,

                "n3":
                    n3,

                "rem":
                    rem
            })


            print(
                f"Épocas originais: "
                f"{total_original}"
            )

            print(
                f"Unknown: "
                f"{total_unknown}"
            )

            print(
                f"Épocas finais: "
                f"{total_final}"
            )

            print(
                f"Cobertura: "
                f"{cobertura:.2f}%"
            )


        except Exception as erro:

            print(
                f"ERRO: {erro}"
            )

            problemas.append(
                (
                    participante,
                    noite,
                    str(erro)
                )
            )


# ============================================================
# CRIAR DATAFRAME
# ============================================================

controle = pd.DataFrame(
    resultados
)


# ============================================================
# SALVAR
# ============================================================

controle.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


# ============================================================
# RESUMO GERAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RESUMO DO CONTROLE DE QUALIDADE"
)

print(
    "=" * 70
)


print(
    f"\nNoites analisadas: "
    f"{len(controle)}"
)

print(
    f"Falhas na validação: "
    f"{len(problemas)}"
)


if len(controle) > 0:

    print(
        f"\nCobertura média: "
        f"{controle['cobertura_percentual'].mean():.2f}%"
    )

    print(
        f"Cobertura mediana: "
        f"{controle['cobertura_percentual'].median():.2f}%"
    )

    print(
        f"Menor cobertura: "
        f"{controle['cobertura_percentual'].min():.2f}%"
    )

    print(
        f"Maior cobertura: "
        f"{controle['cobertura_percentual'].max():.2f}%"
    )


    print(
        f"\nTotal de épocas finais: "
        f"{controle['epocas_finais'].sum()}"
    )


    print(
        f"Valores ausentes totais: "
        f"{controle['valores_ausentes'].sum()}"
    )


    print(
        f"Labels inválidos totais: "
        f"{controle['labels_invalidos'].sum()}"
    )


    # ========================================================
    # 10 PIORES COBERTURAS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "10 NOITES COM MENOR COBERTURA"
    )

    print(
        "=" * 70
    )


    piores = (
        controle
        .sort_values(
            "cobertura_percentual"
        )
        .head(10)
    )


    print(
        piores[
            [
                "participante",
                "noite",
                "epocas_utilizaveis",
                "epocas_finais",
                "removidas_sinal",
                "cobertura_percentual"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# ARQUIVO
# ============================================================

print(
    "\nArquivo criado:"
)

print(
    ARQUIVO_SAIDA
)


if (
    len(problemas) == 0
    and controle[
        "valores_ausentes"
    ].sum() == 0
    and controle[
        "labels_invalidos"
    ].sum() == 0
):

    print(
        "\n✅ ESTRUTURA DOS "
        "75 DATASETS VALIDADA"
    )

else:

    print(
        "\n⚠️ EXISTEM PROBLEMAS "
        "QUE PRECISAM SER ANALISADOS"
    )