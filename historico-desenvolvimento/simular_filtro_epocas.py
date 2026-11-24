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

HR_MINIMO = 4

PERCENTUAL_MOTION = 0.80


# ============================================================
# RESULTADOS
# ============================================================

resultados = []

total_antes = 0
total_depois = 0


print("=" * 75)
print("SIMULAÇÃO DO FILTRO DE QUALIDADE DAS ÉPOCAS")
print("=" * 75)

print(
    f"\nRegra HR: pelo menos "
    f"{HR_MINIMO} amostras"
)

print(
    f"Regra movimento: pelo menos "
    f"{PERCENTUAL_MOTION * 100:.0f}% "
    f"da mediana da própria noite"
)


# ============================================================
# PROCESSAR AS 75 NOITES
# ============================================================

for participante in PARTICIPANTES:

    for noite in NOITES:

        arquivo = (
            PASTA
            / participante
            / f"noite_{noite}.csv"
        )

        df = pd.read_csv(
            arquivo
        )


        # ----------------------------------------------------
        # MEDIANA DO MOVIMENTO NAQUELA NOITE
        # ----------------------------------------------------

        mediana_motion = (
            df["motion_samples"]
            .median()
        )


        limite_motion = (
            mediana_motion
            * PERCENTUAL_MOTION
        )


        # ----------------------------------------------------
        # CRIAR REGRAS
        # ----------------------------------------------------

        hr_ok = (
            df["hr_samples"]
            >= HR_MINIMO
        )


        motion_ok = (
            df["motion_samples"]
            >= limite_motion
        )


        qualidade_ok = (
            hr_ok
            &
            motion_ok
        )


        # ----------------------------------------------------
        # CONTAGENS
        # ----------------------------------------------------

        antes = len(
            df
        )


        depois = int(
            qualidade_ok.sum()
        )


        removidas = (
            antes
            - depois
        )


        percentual_mantido = (
            depois
            / antes
            * 100
        )


        total_antes += antes
        total_depois += depois


        # ----------------------------------------------------
        # MOTIVO DAS REMOÇÕES
        # ----------------------------------------------------

        falha_hr = int(
            (~hr_ok).sum()
        )


        falha_motion = int(
            (~motion_ok).sum()
        )


        falha_ambos = int(
            (
                (~hr_ok)
                &
                (~motion_ok)
            ).sum()
        )


        # ----------------------------------------------------
        # SALVAR RESULTADO
        # ----------------------------------------------------

        resultados.append({

            "participante":
                participante,

            "noite":
                noite,

            "epocas_antes":
                antes,

            "mediana_motion":
                mediana_motion,

            "limite_motion_80":
                limite_motion,

            "falha_hr":
                falha_hr,

            "falha_motion":
                falha_motion,

            "falha_ambos":
                falha_ambos,

            "epocas_depois":
                depois,

            "epocas_removidas":
                removidas,

            "percentual_mantido":
                percentual_mantido
        })


# ============================================================
# DATAFRAME
# ============================================================

resultado = pd.DataFrame(
    resultados
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
    f"\nÉpocas antes: "
    f"{total_antes}"
)


print(
    f"Épocas depois: "
    f"{total_depois}"
)


removidas_total = (
    total_antes
    - total_depois
)


print(
    f"Épocas removidas: "
    f"{removidas_total}"
)


percentual_total = (
    total_depois
    / total_antes
    * 100
)


print(
    f"Percentual mantido: "
    f"{percentual_total:.2f}%"
)


# ============================================================
# ESTATÍSTICAS DOS LIMITES
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "MEDIANA DE AMOSTRAS DE MOVIMENTO POR NOITE"
)

print(
    "=" * 75
)


print(
    resultado[
        "mediana_motion"
    ].describe()
)


# ============================================================
# NOITES MAIS AFETADAS PELO FILTRO
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "10 NOITES MAIS AFETADAS PELO FILTRO"
)

print(
    "=" * 75
)


piores = (
    resultado
    .sort_values(
        "percentual_mantido"
    )
    .head(10)
)


print(
    piores[
        [
            "participante",
            "noite",
            "epocas_antes",
            "mediana_motion",
            "limite_motion_80",
            "falha_hr",
            "falha_motion",
            "epocas_depois",
            "percentual_mantido"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# DISTRIBUIÇÃO
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "QUANTIDADE DE NOITES POR RETENÇÃO"
)

print(
    "=" * 75
)


faixas = [
    99,
    95,
    90,
    80
]


for limite in faixas:

    quantidade = int(
        (
            resultado[
                "percentual_mantido"
            ]
            >= limite
        ).sum()
    )

    print(
        f"Noites mantendo pelo menos "
        f"{limite}% das épocas: "
        f"{quantidade}"
    )


# ============================================================
# SALVAR SIMULAÇÃO
# ============================================================

arquivo_saida = Path(
    "data/processed/"
    "simulacao_filtro_epocas.csv"
)


resultado.to_csv(
    arquivo_saida,
    index=False
)


print(
    "\nArquivo criado:"
)

print(
    arquivo_saida
)


print(
    "\n"
    + "=" * 75
)

print(
    "SIMULAÇÃO CONCLUÍDA"
)

print(
    "=" * 75
)