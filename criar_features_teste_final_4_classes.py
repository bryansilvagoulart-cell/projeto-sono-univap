from pathlib import Path
import itertools
import math

import numpy as np
import pandas as pd


# ============================================================
# ARQUIVOS
# ============================================================

ARQUIVO_BASE_DESENVOLVIMENTO = Path(
    "data/processed/dataset_geral_epocas.csv"
)

ARQUIVO_FEATURES_REFERENCIA = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
)

ARQUIVO_TESTE_QC = Path(
    "data/processed/teste_final_4_classes/dataset_teste_final_qc.csv"
)

ARQUIVO_SAIDA = Path(
    "data/processed/teste_final_4_classes/"
    "dataset_teste_final_features_4_classes.csv"
)


# ============================================================
# 30 FEATURES OFICIAIS
# ============================================================

FEATURES = [

    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",
    "motion_mean",
    "motion_std",
    "motion_max",

    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",
    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean",

    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",
    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean",

    "hr_mean_roll20_mean",
    "hr_mean_roll20_std",
    "hr_std_roll20_mean",
    "motion_mean_roll20_mean",
    "motion_mean_roll20_std",
    "motion_max_roll20_mean",

    "tempo_desde_inicio_min",
    "hora_local_sin",
    "hora_local_cos"
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def detectar_coluna_inicio(df):

    if "inicio" in df.columns:
        return "inicio"

    if "inicio_utc" in df.columns:
        return "inicio_utc"

    raise RuntimeError(
        "Não encontrei coluna de início "
        "(inicio ou inicio_utc)."
    )


def preparar_chaves(df):

    obrigatorias = {
        "participante",
        "noite",
        "epoca"
    }

    if not obrigatorias.issubset(
        set(df.columns)
    ):

        raise RuntimeError(
            "Faltam colunas obrigatórias."
        )

    df = df.copy()

    df["participante"] = (
        df["participante"]
        .astype(str)
    )

    df["noite"] = (
        df["noite"]
        .astype(int)
    )

    df["epoca"] = (
        df["epoca"]
        .astype(int)
    )

    df = (
        df
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

    return df


# ============================================================
# CRIAR SEGMENTOS
# ============================================================

def criar_segmentos(df):

    df = df.copy()

    diferenca = (
        df
        .groupby(
            [
                "participante",
                "noite"
            ]
        )[
            "epoca"
        ]
        .diff()
    )


    quebra = (
        diferenca
        .ne(1)
    )


    df["segmento"] = (

        quebra
        .groupby(
            [
                df["participante"],
                df["noite"]
            ]
        )
        .cumsum()
        .astype(int)
        - 1
    )


    return df


# ============================================================
# FEATURES DE DELTA + ROLLING
# ============================================================

def criar_features_estruturais(
    df,
    usar_segmento=True,
    center=False,
    ddof=1
):

    df = criar_segmentos(
        preparar_chaves(df)
    )


    if usar_segmento:

        grupos = [
            "participante",
            "noite",
            "segmento"
        ]

    else:

        grupos = [
            "participante",
            "noite"
        ]


    # ========================================================
    # DELTAS
    # ========================================================

    df["hr_mean_delta1"] = (

        df
        .groupby(
            grupos
        )[
            "hr_mean"
        ]
        .diff()
        .fillna(0.0)
    )


    df["motion_mean_delta1"] = (

        df
        .groupby(
            grupos
        )[
            "motion_mean"
        ]
        .diff()
        .fillna(0.0)
    )


    # ========================================================
    # ROLLINGS
    # ========================================================

    for janela in [
        5,
        10,
        20
    ]:

        df[
            f"hr_mean_roll{janela}_mean"
        ] = (

            df
            .groupby(
                grupos,
                group_keys=False
            )[
                "hr_mean"
            ]
            .transform(
                lambda serie:
                    serie
                    .rolling(
                        janela,
                        min_periods=1,
                        center=center
                    )
                    .mean()
            )
        )


        df[
            f"hr_mean_roll{janela}_std"
        ] = (

            df
            .groupby(
                grupos,
                group_keys=False
            )[
                "hr_mean"
            ]
            .transform(
                lambda serie:
                    serie
                    .rolling(
                        janela,
                        min_periods=1,
                        center=center
                    )
                    .std(
                        ddof=ddof
                    )
            )
            .fillna(0.0)
        )


        df[
            f"hr_std_roll{janela}_mean"
        ] = (

            df
            .groupby(
                grupos,
                group_keys=False
            )[
                "hr_std"
            ]
            .transform(
                lambda serie:
                    serie
                    .rolling(
                        janela,
                        min_periods=1,
                        center=center
                    )
                    .mean()
            )
        )


        df[
            f"motion_mean_roll{janela}_mean"
        ] = (

            df
            .groupby(
                grupos,
                group_keys=False
            )[
                "motion_mean"
            ]
            .transform(
                lambda serie:
                    serie
                    .rolling(
                        janela,
                        min_periods=1,
                        center=center
                    )
                    .mean()
            )
        )


        df[
            f"motion_mean_roll{janela}_std"
        ] = (

            df
            .groupby(
                grupos,
                group_keys=False
            )[
                "motion_mean"
            ]
            .transform(
                lambda serie:
                    serie
                    .rolling(
                        janela,
                        min_periods=1,
                        center=center
                    )
                    .std(
                        ddof=ddof
                    )
            )
            .fillna(0.0)
        )


        df[
            f"motion_max_roll{janela}_mean"
        ] = (

            df
            .groupby(
                grupos,
                group_keys=False
            )[
                "motion_max"
            ]
            .transform(
                lambda serie:
                    serie
                    .rolling(
                        janela,
                        min_periods=1,
                        center=center
                    )
                    .mean()
            )
        )


    return df


# ============================================================
# TEMPO DESDE INÍCIO
# ============================================================

def aplicar_tempo_desde_inicio(
    df,
    modo
):

    df = df.copy()

    coluna_inicio = (
        detectar_coluna_inicio(
            df
        )
    )


    inicio = pd.to_datetime(
        df[
            coluna_inicio
        ],
        utc=True
    )


    if modo == "epoca":

        df[
            "tempo_desde_inicio_min"
        ] = (

            df["epoca"]
            * 0.5
        )


    elif modo == "primeira_epoca_valida":

        primeiro_inicio = (

            inicio
            .groupby(
                [
                    df["participante"],
                    df["noite"]
                ]
            )
            .transform(
                "min"
            )
        )


        df[
            "tempo_desde_inicio_min"
        ] = (

            (
                inicio
                - primeiro_inicio
            )
            .dt
            .total_seconds()
            / 60.0
        )


    elif modo == "inicio_segmento":

        primeiro_inicio = (

            inicio
            .groupby(
                [
                    df["participante"],
                    df["noite"],
                    df["segmento"]
                ]
            )
            .transform(
                "min"
            )
        )


        df[
            "tempo_desde_inicio_min"
        ] = (

            (
                inicio
                - primeiro_inicio
            )
            .dt
            .total_seconds()
            / 60.0
        )


    else:

        raise ValueError(
            f"Modo de tempo inválido: {modo}"
        )


    return df


# ============================================================
# HORA CÍCLICA
# ============================================================

def aplicar_hora_ciclica(
    df,
    modo
):

    df = df.copy()

    coluna_inicio = (
        detectar_coluna_inicio(
            df
        )
    )


    instante = pd.to_datetime(
        df[
            coluna_inicio
        ],
        utc=True
    )


    if modo.startswith(
        "local"
    ):

        instante = (
            instante
            .dt
            .tz_convert(
                "America/New_York"
            )
        )


    if modo.endswith(
        "hora_inteira"
    ):

        hora = (
            instante.dt.hour
            .astype(float)
        )


    elif modo.endswith(
        "decimal_min"
    ):

        hora = (

            instante.dt.hour

            +

            instante.dt.minute
            / 60.0
        )


    elif modo.endswith(
        "decimal_completo"
    ):

        hora = (

            instante.dt.hour

            +

            instante.dt.minute
            / 60.0

            +

            instante.dt.second
            / 3600.0
        )


    else:

        raise ValueError(
            f"Modo de hora inválido: {modo}"
        )


    angulo = (

        2.0
        * np.pi
        * hora
        / 24.0
    )


    df[
        "hora_local_sin"
    ] = np.sin(
        angulo
    )


    df[
        "hora_local_cos"
    ] = np.cos(
        angulo
    )


    return df


# ============================================================
# COMPARAÇÃO NUMÉRICA
# ============================================================

def comparar_colunas(
    calculado,
    referencia,
    colunas
):

    chave = [
        "participante",
        "noite",
        "epoca"
    ]


    ref = referencia[
        chave
        + colunas
    ].copy()


    calc = calculado[
        chave
        + colunas
    ].copy()


    comparacao = ref.merge(

        calc,

        on=chave,

        how="inner",

        suffixes=(
            "_ref",
            "_calc"
        )
    )


    if len(comparacao) != len(ref):

        return {
            "mismatches":
                10**12,

            "max_diff":
                float("inf")
        }


    mismatches = 0

    max_diff = 0.0


    for coluna in colunas:

        a = (
            comparacao[
                coluna
                + "_ref"
            ]
            .to_numpy(
                dtype=float
            )
        )


        b = (
            comparacao[
                coluna
                + "_calc"
            ]
            .to_numpy(
                dtype=float
            )
        )


        proximos = np.isclose(
            a,
            b,
            rtol=1e-7,
            atol=1e-9,
            equal_nan=True
        )


        mismatches += int(
            (~proximos).sum()
        )


        diferenca = np.abs(
            a
            - b
        )


        diferenca = diferenca[
            np.isfinite(
                diferenca
            )
        ]


        if len(diferenca) > 0:

            max_diff = max(
                max_diff,
                float(
                    diferenca.max()
                )
            )


    return {
        "mismatches":
            mismatches,

        "max_diff":
            max_diff
    }


# ============================================================
# AUTODESCOBRIR CONFIGURAÇÃO ESTRUTURAL
# ============================================================

def descobrir_configuracao_estrutural(
    base,
    referencia
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AUTOVERIFICAÇÃO DAS FEATURES TEMPORAIS"
    )

    print(
        "=" * 100
    )


    colunas_estruturais = [

        "hr_mean_delta1",
        "motion_mean_delta1",

        "hr_mean_roll5_mean",
        "hr_mean_roll5_std",
        "hr_std_roll5_mean",
        "motion_mean_roll5_mean",
        "motion_mean_roll5_std",
        "motion_max_roll5_mean",

        "hr_mean_roll10_mean",
        "hr_mean_roll10_std",
        "hr_std_roll10_mean",
        "motion_mean_roll10_mean",
        "motion_mean_roll10_std",
        "motion_max_roll10_mean",

        "hr_mean_roll20_mean",
        "hr_mean_roll20_std",
        "hr_std_roll20_mean",
        "motion_mean_roll20_mean",
        "motion_mean_roll20_std",
        "motion_max_roll20_mean"
    ]


    melhor = None


    for usar_segmento in [
        True,
        False
    ]:

        for center in [
            False,
            True
        ]:

            for ddof in [
                1,
                0
            ]:

                calculado = (
                    criar_features_estruturais(
                        base,
                        usar_segmento=
                            usar_segmento,
                        center=
                            center,
                        ddof=
                            ddof
                    )
                )


                resultado = (
                    comparar_colunas(
                        calculado,
                        referencia,
                        colunas_estruturais
                    )
                )


                candidato = {
                    "usar_segmento":
                        usar_segmento,

                    "center":
                        center,

                    "ddof":
                        ddof,

                    "mismatches":
                        resultado[
                            "mismatches"
                        ],

                    "max_diff":
                        resultado[
                            "max_diff"
                        ]
                }


                print(
                    f"segmento={usar_segmento} | "
                    f"center={center} | "
                    f"ddof={ddof} | "
                    f"diferenças={candidato['mismatches']}"
                )


                if (
                    melhor is None
                    or candidato[
                        "mismatches"
                    ]
                    < melhor[
                        "mismatches"
                    ]
                ):

                    melhor = candidato


    if (
        melhor is None
        or melhor[
            "mismatches"
        ] != 0
    ):

        print(
            "\n❌ Nenhuma configuração reproduziu "
            "exatamente as features estruturais antigas."
        )


        print(
            "Melhor tentativa:"
        )


        print(
            melhor
        )


        raise RuntimeError(
            "AUTOVERIFICAÇÃO DAS FEATURES FALHOU."
        )


    print(
        "\n✅ Configuração estrutural encontrada:"
    )


    print(
        melhor
    )


    return melhor


# ============================================================
# AUTODESCOBRIR TEMPO
# ============================================================

def descobrir_modo_tempo(
    base_estrutural,
    referencia
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AUTOVERIFICAÇÃO DO TEMPO DESDE INÍCIO"
    )

    print(
        "=" * 100
    )


    melhor = None


    for modo in [
        "epoca",
        "primeira_epoca_valida",
        "inicio_segmento"
    ]:

        calculado = (
            aplicar_tempo_desde_inicio(
                base_estrutural,
                modo
            )
        )


        resultado = (
            comparar_colunas(
                calculado,
                referencia,
                [
                    "tempo_desde_inicio_min"
                ]
            )
        )


        candidato = {
            "modo":
                modo,

            "mismatches":
                resultado[
                    "mismatches"
                ],

            "max_diff":
                resultado[
                    "max_diff"
                ]
        }


        print(
            f"{modo}: "
            f"diferenças="
            f"{candidato['mismatches']}"
        )


        if (
            melhor is None
            or candidato[
                "mismatches"
            ]
            < melhor[
                "mismatches"
            ]
        ):

            melhor = candidato


    if (
        melhor is None
        or melhor[
            "mismatches"
        ] != 0
    ):

        print(
            "\n❌ O tempo desde início "
            "não foi reproduzido exatamente."
        )


        print(
            "Melhor tentativa:"
        )


        print(
            melhor
        )


        raise RuntimeError(
            "AUTOVERIFICAÇÃO DE TEMPO FALHOU."
        )


    print(
        "\n✅ Modo de tempo encontrado:"
    )


    print(
        melhor
    )


    return melhor[
        "modo"
    ]


# ============================================================
# AUTODESCOBRIR HORA CÍCLICA
# ============================================================

def descobrir_modo_hora(
    base,
    referencia
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AUTOVERIFICAÇÃO DA HORA CÍCLICA"
    )

    print(
        "=" * 100
    )


    modos = [

        "local_hora_inteira",
        "local_decimal_min",
        "local_decimal_completo",

        "utc_hora_inteira",
        "utc_decimal_min",
        "utc_decimal_completo"
    ]


    melhor = None


    for modo in modos:

        calculado = (
            aplicar_hora_ciclica(
                base,
                modo
            )
        )


        resultado = (
            comparar_colunas(
                calculado,
                referencia,
                [
                    "hora_local_sin",
                    "hora_local_cos"
                ]
            )
        )


        candidato = {
            "modo":
                modo,

            "mismatches":
                resultado[
                    "mismatches"
                ],

            "max_diff":
                resultado[
                    "max_diff"
                ]
        }


        print(
            f"{modo}: "
            f"diferenças="
            f"{candidato['mismatches']}"
        )


        if (
            melhor is None
            or candidato[
                "mismatches"
            ]
            < melhor[
                "mismatches"
            ]
        ):

            melhor = candidato


    if (
        melhor is None
        or melhor[
            "mismatches"
        ] != 0
    ):

        print(
            "\n❌ A representação cíclica "
            "não foi reproduzida exatamente."
        )


        print(
            "Melhor tentativa:"
        )


        print(
            melhor
        )


        raise RuntimeError(
            "AUTOVERIFICAÇÃO DE HORA FALHOU."
        )


    print(
        "\n✅ Modo de hora encontrado:"
    )


    print(
        melhor
    )


    return melhor[
        "modo"
    ]


# ============================================================
# CRIAR TODAS AS FEATURES
# ============================================================

def criar_todas_features(
    df,
    config_estrutural,
    modo_tempo,
    modo_hora
):

    resultado = (
        criar_features_estruturais(

            df,

            usar_segmento=
                config_estrutural[
                    "usar_segmento"
                ],

            center=
                config_estrutural[
                    "center"
                ],

            ddof=
                config_estrutural[
                    "ddof"
                ]
        )
    )


    resultado = (
        aplicar_tempo_desde_inicio(
            resultado,
            modo_tempo
        )
    )


    resultado = (
        aplicar_hora_ciclica(
            resultado,
            modo_hora
        )
    )


    return resultado


# ============================================================
# EXECUÇÃO
# ============================================================

print(
    "=" * 100
)

print(
    "CRIAÇÃO DAS FEATURES - TESTE FINAL 4 CLASSES"
)

print(
    "=" * 100
)


# ============================================================
# VALIDAR ARQUIVOS
# ============================================================

for arquivo in [

    ARQUIVO_BASE_DESENVOLVIMENTO,
    ARQUIVO_FEATURES_REFERENCIA,
    ARQUIVO_TESTE_QC

]:

    if not arquivo.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n"
            f"{arquivo}"
        )


# ============================================================
# CARREGAR DESENVOLVIMENTO ANTIGO
# ============================================================

print(
    "\nCarregando dados antigos "
    "para autoverificação..."
)


base_dev = pd.read_csv(
    ARQUIVO_BASE_DESENVOLVIMENTO
)


ref_dev = pd.read_csv(
    ARQUIVO_FEATURES_REFERENCIA
)


base_dev = preparar_chaves(
    base_dev
)


ref_dev = preparar_chaves(
    ref_dev
)


print(
    f"Base antiga: "
    f"{len(base_dev)} épocas"
)


print(
    f"Features antigas: "
    f"{len(ref_dev)} épocas"
)


if len(base_dev) != len(ref_dev):

    raise RuntimeError(
        "Os datasets antigos possuem "
        "quantidades diferentes."
    )


# ============================================================
# ENCONTRAR RECEITA EXATA
# ============================================================

config_estrutural = (
    descobrir_configuracao_estrutural(
        base_dev,
        ref_dev
    )
)


base_dev_estrutural = (
    criar_features_estruturais(

        base_dev,

        usar_segmento=
            config_estrutural[
                "usar_segmento"
            ],

        center=
            config_estrutural[
                "center"
            ],

        ddof=
            config_estrutural[
                "ddof"
            ]
    )
)


modo_tempo = (
    descobrir_modo_tempo(
        base_dev_estrutural,
        ref_dev
    )
)


modo_hora = (
    descobrir_modo_hora(
        base_dev_estrutural,
        ref_dev
    )
)


# ============================================================
# VERIFICAÇÃO FINAL DAS 30 FEATURES
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "VERIFICAÇÃO FINAL DAS 30 FEATURES"
)

print(
    "=" * 100
)


dev_reconstruido = (
    criar_todas_features(
        base_dev,
        config_estrutural,
        modo_tempo,
        modo_hora
    )
)


resultado_final = (
    comparar_colunas(
        dev_reconstruido,
        ref_dev,
        FEATURES
    )
)


if (
    resultado_final[
        "mismatches"
    ]
    != 0
):

    raise RuntimeError(
        f"As 30 features não coincidiram. "
        f"Diferenças: "
        f"{resultado_final['mismatches']}"
    )


print(
    "✅ As 30 features foram reproduzidas "
    "exatamente no desenvolvimento antigo."
)


# ============================================================
# CRIAR FEATURES DO TESTE FINAL
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "CRIANDO FEATURES DO TESTE FINAL"
)

print(
    "=" * 100
)


teste = pd.read_csv(
    ARQUIVO_TESTE_QC
)


teste = preparar_chaves(
    teste
)


teste_features = (
    criar_todas_features(
        teste,
        config_estrutural,
        modo_tempo,
        modo_hora
    )
)


# ============================================================
# MAPEAR PARA 4 CLASSES
# ============================================================

mapa_4_classes = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 3
}


nomes_4_classes = {
    0: "Wake",
    1: "Light",
    2: "Deep",
    3: "REM"
}


teste_features[
    "label_5_classes"
] = (
    teste_features[
        "label"
    ]
    .astype(int)
)


teste_features[
    "label_4_classes"
] = (

    teste_features[
        "label_5_classes"
    ]
    .map(
        mapa_4_classes
    )
)


if (
    teste_features[
        "label_4_classes"
    ]
    .isna()
    .any()
):

    raise RuntimeError(
        "Foi encontrado algum label "
        "fora de 0,1,2,3,4."
    )


teste_features[
    "label_4_classes"
] = (

    teste_features[
        "label_4_classes"
    ]
    .astype(int)
)


teste_features[
    "label"
] = (
    teste_features[
        "label_4_classes"
    ]
)


teste_features[
    "classe_4"
] = (

    teste_features[
        "label_4_classes"
    ]
    .map(
        nomes_4_classes
    )
)


# ============================================================
# VALIDAR FEATURES
# ============================================================

ausentes = (

    teste_features[
        FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


if ausentes != 0:

    raise RuntimeError(
        f"Foram encontrados "
        f"{ausentes} valores ausentes."
    )


valores = (

    teste_features[
        FEATURES
    ]
    .to_numpy(
        dtype=float
    )
)


if not np.isfinite(
    valores
).all():

    raise RuntimeError(
        "Existem valores infinitos "
        "ou inválidos nas features."
    )


if len(
    teste_features
) != len(
    teste
):

    raise RuntimeError(
        "A quantidade de épocas mudou "
        "durante a criação das features."
    )


# ============================================================
# ORGANIZAR COLUNAS
# ============================================================

colunas_metadata = [
    "participante",
    "noite",
    "segmento",
    "epoca"
]


coluna_inicio = (
    detectar_coluna_inicio(
        teste_features
    )
)


if coluna_inicio not in (
    colunas_metadata
):

    colunas_metadata.append(
        coluna_inicio
    )


colunas_saida = (

    colunas_metadata

    + [
        "label_5_classes",
        "label_4_classes",
        "label",
        "classe_4"
    ]

    + FEATURES
)


teste_features = (
    teste_features[
        colunas_saida
    ]
    .copy()
)


# ============================================================
# SALVAR
# ============================================================

teste_features.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


# ============================================================
# RESUMO
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FEATURES DO TESTE FINAL CRIADAS COM SUCESSO"
)

print(
    "=" * 100
)


print(
    f"\nÉpocas: "
    f"{len(teste_features)}"
)


print(
    f"Participantes: "
    f"{teste_features['participante'].nunique()}"
)


print(
    f"Noites: "
    f"{teste_features[['participante','noite']].drop_duplicates().shape[0]}"
)


print(
    f"Features oficiais: "
    f"{len(FEATURES)}"
)


print(
    f"Valores ausentes nas features: "
    f"{teste_features[FEATURES].isna().sum().sum()}"
)


print(
    "\nArquivo criado:"
)

print(
    ARQUIVO_SAIDA
)


print(
    "\n✅ Nenhuma métrica do modelo foi calculada."
)

print(
    "✅ Nenhuma configuração foi alterada."
)

print(
    "✅ O teste final continua independente."
)


print(
    "\nPróxima etapa:"
)

print(
    "Treinar o modelo final de 4 classes "
    "nos 15 participantes de desenvolvimento."
)