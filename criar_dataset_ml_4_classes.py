from pathlib import Path

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_ENTRADA = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
)

ARQUIVO_SAIDA = Path(
    "data/processed/dataset_ml_4_classes.csv"
)


# ============================================================
# MAPEAMENTO
# ============================================================
#
# ORIGINAL:
#
# 0 = Wake
# 1 = N1
# 2 = N2
# 3 = N3
# 4 = REM
#
#
# NOVO:
#
# 0 = Wake
# 1 = Light  (N1 + N2)
# 2 = Deep   (N3)
# 3 = REM
#
# ============================================================

MAPA_4_CLASSES = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 3
}


NOMES_CLASSES = {
    0: "Wake",
    1: "Light",
    2: "Deep",
    3: "REM"
}


# ============================================================
# INÍCIO
# ============================================================

print("=" * 95)
print("CRIAÇÃO DO DATASET OFICIAL DE ML - 4 CLASSES")
print("=" * 95)


# ============================================================
# VALIDAR ARQUIVO
# ============================================================

if not ARQUIVO_ENTRADA.exists():

    raise FileNotFoundError(
        f"Dataset não encontrado:\n"
        f"{ARQUIVO_ENTRADA}"
    )


# ============================================================
# CARREGAR
# ============================================================

print(
    "\nCarregando dataset original..."
)


df = pd.read_csv(
    ARQUIVO_ENTRADA
)


print(
    f"Épocas encontradas: "
    f"{len(df)}"
)


# ============================================================
# VALIDAR LABEL
# ============================================================

if "label" not in df.columns:

    raise ValueError(
        "A coluna 'label' não foi encontrada."
    )


if df[
    "label"
].isna().any():

    raise ValueError(
        "Existem labels ausentes."
    )


df[
    "label"
] = df[
    "label"
].astype(
    int
)


classes_originais = set(
    int(x)
    for x in df[
        "label"
    ].unique()
)


esperadas = {
    0,
    1,
    2,
    3,
    4
}


if classes_originais != esperadas:

    raise ValueError(
        f"Classes encontradas: "
        f"{classes_originais}\n"
        f"Esperávamos: {esperadas}"
    )


print(
    "✅ Classes originais validadas."
)


# ============================================================
# PRESERVAR LABEL ORIGINAL
# ============================================================

df[
    "label_5_classes"
] = df[
    "label"
].copy()


# ============================================================
# CRIAR LABEL DE 4 CLASSES
# ============================================================

df[
    "label_4_classes"
] = df[
    "label"
].map(
    MAPA_4_CLASSES
)


if df[
    "label_4_classes"
].isna().any():

    raise RuntimeError(
        "Algum label não pôde ser convertido."
    )


df[
    "label_4_classes"
] = df[
    "label_4_classes"
].astype(
    int
)


# ============================================================
# O LABEL PRINCIPAL PASSA A SER O DE 4 CLASSES
# ============================================================

df[
    "label"
] = df[
    "label_4_classes"
]


# ============================================================
# NOME DA CLASSE
# ============================================================

df[
    "classe_4"
] = df[
    "label"
].map(
    NOMES_CLASSES
)


# ============================================================
# VALIDAÇÕES
# ============================================================

if len(df) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(df)}."
    )


if df[
    "label"
].isna().any():

    raise RuntimeError(
        "Existem labels inválidos após conversão."
    )


if not set(
    df[
        "label"
    ].unique()
).issubset(
    {
        0,
        1,
        2,
        3
    }
):

    raise RuntimeError(
        "Existem classes fora de 0, 1, 2 e 3."
    )


# ============================================================
# DISTRIBUIÇÃO ORIGINAL
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "DISTRIBUIÇÃO ORIGINAL - 5 CLASSES"
)

print(
    "=" * 95
)


distribuicao_5 = (
    df[
        "label_5_classes"
    ]
    .value_counts()
    .sort_index()
)


nomes_5 = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}


for classe, quantidade in (
    distribuicao_5.items()
):

    percentual = (
        quantidade
        / len(df)
        * 100
    )

    print(
        f"{classe} "
        f"{nomes_5[classe]:<5} "
        f"-> "
        f"{quantidade:6d} "
        f"({percentual:6.2f}%)"
    )


# ============================================================
# DISTRIBUIÇÃO NOVA
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "DISTRIBUIÇÃO NOVA - 4 CLASSES"
)

print(
    "=" * 95
)


distribuicao_4 = (
    df[
        "label"
    ]
    .value_counts()
    .sort_index()
)


for classe, quantidade in (
    distribuicao_4.items()
):

    percentual = (
        quantidade
        / len(df)
        * 100
    )

    print(
        f"{classe} "
        f"{NOMES_CLASSES[classe]:<6} "
        f"-> "
        f"{quantidade:6d} "
        f"({percentual:6.2f}%)"
    )


# ============================================================
# CONFERIR N1 + N2
# ============================================================

n1 = int(
    (
        df[
            "label_5_classes"
        ] == 1
    ).sum()
)


n2 = int(
    (
        df[
            "label_5_classes"
        ] == 2
    ).sum()
)


light = int(
    (
        df[
            "label"
        ] == 1
    ).sum()
)


print(
    "\nValidação de Light:"
)

print(
    f"N1 original: "
    f"{n1}"
)

print(
    f"N2 original: "
    f"{n2}"
)

print(
    f"N1 + N2: "
    f"{n1 + n2}"
)

print(
    f"Light novo: "
    f"{light}"
)


if light != (
    n1 + n2
):

    raise RuntimeError(
        "A classe Light não corresponde "
        "exatamente a N1 + N2."
    )


print(
    "✅ Light = N1 + N2 confirmado."
)


# ============================================================
# PARTICIPANTES E NOITES
# ============================================================

quantidade_participantes = (
    df[
        "participante"
    ].nunique()
)


quantidade_noites = (
    df[
        "id_noite"
    ].nunique()
)


print(
    "\nEstrutura:"
)

print(
    f"Participantes: "
    f"{quantidade_participantes}"
)

print(
    f"Noites: "
    f"{quantidade_noites}"
)

print(
    f"Épocas: "
    f"{len(df)}"
)


# ============================================================
# SALVAR
# ============================================================

df.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


print(
    "\n"
    + "=" * 95
)

print(
    "DATASET DE 4 CLASSES CRIADO COM SUCESSO"
)

print(
    "=" * 95
)


print(
    f"\nArquivo:"
)

print(
    ARQUIVO_SAIDA
)


print(
    "\nMapeamento final:"
)

print(
    "0 -> Wake"
)

print(
    "1 -> Light (N1 + N2)"
)

print(
    "2 -> Deep (N3)"
)

print(
    "3 -> REM"
)


print(
    "\n✅ Nenhuma época removida."
)

print(
    "✅ Os dados fisiológicos não foram alterados."
)

print(
    "✅ Apenas o alvo de classificação foi remapeado."
)