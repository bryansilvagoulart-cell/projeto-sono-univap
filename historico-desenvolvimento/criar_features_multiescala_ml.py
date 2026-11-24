from pathlib import Path

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_ENTRADA = Path(
    "data/processed/dataset_ml_features_temporais.csv"
)

ARQUIVO_SAIDA = Path(
    "data/processed/dataset_ml_features_multiescala.csv"
)


# ============================================================
# INÍCIO
# ============================================================

print("=" * 75)
print("CRIAÇÃO DAS FEATURES TEMPORAIS MULTIESCALA")
print("=" * 75)


# ============================================================
# 1. CARREGAR DATASET TEMPORAL
# ============================================================

print(
    "\nCarregando dataset temporal..."
)

dataset = pd.read_csv(
    ARQUIVO_ENTRADA
)

print(
    f"Épocas carregadas: {len(dataset)}"
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


# ============================================================
# 2. VERIFICAR COLUNAS NECESSÁRIAS
# ============================================================

colunas_obrigatorias = [
    "id_noite",
    "segmento",
    "epoca",

    "hr_mean",
    "hr_std",

    "motion_mean",
    "motion_max"
]


faltando = [
    coluna
    for coluna in colunas_obrigatorias
    if coluna not in dataset.columns
]


if faltando:

    raise ValueError(
        "Estão faltando as colunas: "
        + ", ".join(faltando)
    )


# ============================================================
# 3. GARANTIR ORDEM TEMPORAL
# ============================================================

dataset = dataset.sort_values(
    by=[
        "id_noite",
        "segmento",
        "epoca"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 4. AGRUPAR APENAS TRECHOS CONTÍNUOS
# ============================================================

grupos = dataset.groupby(
    [
        "id_noite",
        "segmento"
    ],
    sort=False
)


# ============================================================
# FUNÇÃO PARA CRIAR UMA ESCALA TEMPORAL
# ============================================================

def criar_features_janela(
    dataframe,
    grupos,
    janela
):

    # --------------------------------------------------------
    # Frequência cardíaca média
    # --------------------------------------------------------

    dataframe[
        f"hr_mean_roll{janela}_mean"
    ] = (

        grupos["hr_mean"]

        .transform(
            lambda serie:
                serie
                .rolling(
                    window=janela,
                    min_periods=1
                )
                .mean()
        )
    )


    # --------------------------------------------------------
    # Variação da frequência cardíaca média
    # --------------------------------------------------------

    dataframe[
        f"hr_mean_roll{janela}_std"
    ] = (

        grupos["hr_mean"]

        .transform(
            lambda serie:
                serie
                .rolling(
                    window=janela,
                    min_periods=1
                )
                .std(
                    ddof=0
                )
        )
    )


    # --------------------------------------------------------
    # Média da variabilidade interna da FC
    # --------------------------------------------------------

    dataframe[
        f"hr_std_roll{janela}_mean"
    ] = (

        grupos["hr_std"]

        .transform(
            lambda serie:
                serie
                .rolling(
                    window=janela,
                    min_periods=1
                )
                .mean()
        )
    )


    # --------------------------------------------------------
    # Movimento médio
    # --------------------------------------------------------

    dataframe[
        f"motion_mean_roll{janela}_mean"
    ] = (

        grupos["motion_mean"]

        .transform(
            lambda serie:
                serie
                .rolling(
                    window=janela,
                    min_periods=1
                )
                .mean()
        )
    )


    # --------------------------------------------------------
    # Variação do movimento
    # --------------------------------------------------------

    dataframe[
        f"motion_mean_roll{janela}_std"
    ] = (

        grupos["motion_mean"]

        .transform(
            lambda serie:
                serie
                .rolling(
                    window=janela,
                    min_periods=1
                )
                .std(
                    ddof=0
                )
        )
    )


    # --------------------------------------------------------
    # Movimento máximo médio
    # --------------------------------------------------------

    dataframe[
        f"motion_max_roll{janela}_mean"
    ] = (

        grupos["motion_max"]

        .transform(
            lambda serie:
                serie
                .rolling(
                    window=janela,
                    min_periods=1
                )
                .mean()
        )
    )


# ============================================================
# 5. JANELA DE 5 MINUTOS
# ============================================================

print(
    "\nCriando contexto de 5 minutos..."
)

criar_features_janela(
    dataframe=dataset,
    grupos=grupos,
    janela=10
)


# ============================================================
# 6. JANELA DE 10 MINUTOS
# ============================================================

print(
    "Criando contexto de 10 minutos..."
)

criar_features_janela(
    dataframe=dataset,
    grupos=grupos,
    janela=20
)


# ============================================================
# 7. LISTA DAS NOVAS FEATURES
# ============================================================

novas_features = []

for janela in [
    10,
    20
]:

    novas_features.extend([
        f"hr_mean_roll{janela}_mean",
        f"hr_mean_roll{janela}_std",
        f"hr_std_roll{janela}_mean",

        f"motion_mean_roll{janela}_mean",
        f"motion_mean_roll{janela}_std",
        f"motion_max_roll{janela}_mean"
    ])


# ============================================================
# 8. VALIDAR
# ============================================================

print(
    "\nValidando novas features..."
)


ausentes = (
    dataset[
        novas_features
    ]

    .isna()

    .sum()

    .sum()
)


print(
    f"Linhas finais: {len(dataset)}"
)

print(
    f"Novas features: {len(novas_features)}"
)

print(
    f"Valores ausentes: {ausentes}"
)


if ausentes != 0:

    raise ValueError(
        "Foram encontrados valores ausentes "
        "nas features multiescala."
    )


# ============================================================
# 9. SALVAR
# ============================================================

ARQUIVO_SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)


dataset.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


# ============================================================
# RESULTADO
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "DATASET MULTIESCALA CRIADO"
)

print(
    "=" * 75
)


print(
    f"\nArquivo:"
)

print(
    ARQUIVO_SAIDA
)


print(
    "\nFeatures adicionadas:"
)


for feature in novas_features:

    print(
        f"  - {feature}"
    )


print(
    "\n✅ FEATURES DE 5 E 10 MINUTOS "
    "CRIADAS COM SUCESSO"
)