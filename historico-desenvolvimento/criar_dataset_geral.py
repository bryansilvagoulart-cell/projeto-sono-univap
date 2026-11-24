from pathlib import Path

import numpy as np
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


PASTA_PROCESSADOS = Path(
    "data/processed/participantes"
)

ARQUIVO_QUALIDADE = Path(
    "data/processed/"
    "qualidade_temporal_epocas.csv"
)

ARQUIVO_DATASET = Path(
    "data/processed/"
    "dataset_geral_epocas.csv"
)

ARQUIVO_EXCLUIDAS = Path(
    "data/processed/"
    "epocas_excluidas_qualidade.csv"
)


# ============================================================
# REGRAS DE QUALIDADE
# ============================================================

COBERTURA_MINIMA = 95.0

MAIOR_GAP_MAXIMO = 1.0

HR_SAMPLES_MINIMO = 4


# ============================================================
# 1. CARREGAR CONTROLE DE QUALIDADE
# ============================================================

print("=" * 75)
print("CRIAÇÃO DO DATASET GERAL DE ÉPOCAS")
print("=" * 75)


print(
    "\nCarregando informações "
    "de qualidade temporal..."
)


qualidade = pd.read_csv(
    ARQUIVO_QUALIDADE
)


print(
    f"Épocas no controle de qualidade: "
    f"{len(qualidade)}"
)


# ============================================================
# 2. CARREGAR AS 75 NOITES PROCESSADAS
# ============================================================

datasets = []

contador = 0

total_noites = (
    len(PARTICIPANTES)
    * len(NOITES)
)


print(
    "\nCarregando os "
    "75 datasets processados..."
)


for participante in PARTICIPANTES:

    for noite in NOITES:

        contador += 1

        arquivo = (
            PASTA_PROCESSADOS
            / participante
            / f"noite_{noite}.csv"
        )


        df = pd.read_csv(
            arquivo
        )


        datasets.append(
            df
        )


        print(
            f"[{contador:02d}/{total_noites}] "
            f"{participante} / noite {noite} "
            f"-> {len(df)} épocas"
        )


# ============================================================
# 3. JUNTAR TODAS AS NOITES
# ============================================================

dataset = pd.concat(
    datasets,
    ignore_index=True
)


print(
    f"\nTotal antes do controle "
    f"de qualidade: "
    f"{len(dataset)}"
)


# ============================================================
# 4. VERIFICAR DUPLICIDADES
# ============================================================

chave = [
    "participante",
    "noite",
    "epoca"
]


duplicadas = (
    dataset
    .duplicated(
        subset=chave
    )
    .sum()
)


print(
    f"Épocas duplicadas: "
    f"{duplicadas}"
)


if duplicadas > 0:

    raise ValueError(
        "Foram encontradas épocas "
        "duplicadas no dataset."
    )


# ============================================================
# 5. JUNTAR COM QUALIDADE TEMPORAL
# ============================================================

# O arquivo de qualidade possui hr_samples
# e motion_samples também.
#
# Como essas colunas já existem no dataset,
# vamos trazer somente as novas métricas
# de qualidade temporal.

qualidade_reduzida = qualidade[
    [
        "participante",
        "noite",
        "epoca",
        "duracao_coberta",
        "cobertura_temporal_pct",
        "maior_gap"
    ]
].copy()


dataset = dataset.merge(
    qualidade_reduzida,
    on=[
        "participante",
        "noite",
        "epoca"
    ],
    how="left",
    validate="one_to_one"
)


# ============================================================
# 6. VERIFICAR SE TODAS AS ÉPOCAS ENCONTRARAM QUALIDADE
# ============================================================

sem_qualidade = (
    dataset[
        "cobertura_temporal_pct"
    ]
    .isna()
    .sum()
)


print(
    f"Épocas sem informação "
    f"de cobertura temporal: "
    f"{sem_qualidade}"
)


# Uma época com apenas uma amostra
# pode possuir maior_gap = NaN.
#
# Ela não deve passar no filtro.

dataset[
    "maior_gap"
] = dataset[
    "maior_gap"
].fillna(
    np.inf
)


# ============================================================
# 7. CRIAR AS TRÊS REGRAS
# ============================================================

dataset[
    "qualidade_cobertura_ok"
] = (

    dataset[
        "cobertura_temporal_pct"
    ]

    >= COBERTURA_MINIMA
)


dataset[
    "qualidade_gap_ok"
] = (

    dataset[
        "maior_gap"
    ]

    <= MAIOR_GAP_MAXIMO
)


dataset[
    "qualidade_hr_ok"
] = (

    dataset[
        "hr_samples"
    ]

    >= HR_SAMPLES_MINIMO
)


# ============================================================
# 8. QUALIDADE FINAL
# ============================================================

dataset[
    "qualidade_ok"
] = (

    dataset[
        "qualidade_cobertura_ok"
    ]

    &

    dataset[
        "qualidade_gap_ok"
    ]

    &

    dataset[
        "qualidade_hr_ok"
    ]
)


# ============================================================
# 9. ESTATÍSTICAS DAS REGRAS
# ============================================================

total = len(
    dataset
)


falha_cobertura = int(
    (
        ~dataset[
            "qualidade_cobertura_ok"
        ]
    ).sum()
)


falha_gap = int(
    (
        ~dataset[
            "qualidade_gap_ok"
        ]
    ).sum()
)


falha_hr = int(
    (
        ~dataset[
            "qualidade_hr_ok"
        ]
    ).sum()
)


total_validas = int(
    dataset[
        "qualidade_ok"
    ].sum()
)


total_excluidas = (
    total
    - total_validas
)


# ============================================================
# 10. MOSTRAR O RESULTADO DO FILTRO
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "RESULTADO DO CONTROLE DE QUALIDADE"
)

print(
    "=" * 75
)


print(
    f"\nÉpocas originais: "
    f"{total}"
)


print(
    f"\nFalha por cobertura < "
    f"{COBERTURA_MINIMA}%: "
    f"{falha_cobertura}"
)


print(
    f"Falha por gap > "
    f"{MAIOR_GAP_MAXIMO} s: "
    f"{falha_gap}"
)


print(
    f"Falha por HR < "
    f"{HR_SAMPLES_MINIMO} amostras: "
    f"{falha_hr}"
)


print(
    f"\nÉpocas aprovadas: "
    f"{total_validas}"
)


print(
    f"Épocas excluídas: "
    f"{total_excluidas}"
)


percentual_mantido = (
    total_validas
    / total
    * 100
)


print(
    f"Percentual mantido: "
    f"{percentual_mantido:.2f}%"
)


# ============================================================
# 11. GUARDAR ÉPOCAS EXCLUÍDAS
# ============================================================

excluidas = dataset[
    ~dataset[
        "qualidade_ok"
    ]
].copy()


excluidas.to_csv(
    ARQUIVO_EXCLUIDAS,
    index=False
)


# ============================================================
# 12. CRIAR DATASET DEFINITIVO
# ============================================================

dataset_final = dataset[
    dataset[
        "qualidade_ok"
    ]
].copy()


# ============================================================
# 13. ORDENAR
# ============================================================

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


# ============================================================
# 14. VERIFICAR LABELS
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "DISTRIBUIÇÃO DOS ESTÁGIOS"
)

print(
    "=" * 75
)


distribuicao = (
    dataset_final[
        "label"
    ]
    .value_counts()
    .sort_index()
)


nomes = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}


for label, quantidade in distribuicao.items():

    percentual = (
        quantidade
        / len(dataset_final)
        * 100
    )


    print(
        f"{label} - "
        f"{nomes.get(label, 'Desconhecido')}: "
        f"{quantidade} "
        f"({percentual:.2f}%)"
    )


# ============================================================
# 15. VERIFICAR VALORES AUSENTES
# ============================================================

total_nan = int(
    dataset_final
    .isna()
    .sum()
    .sum()
)


print(
    f"\nValores ausentes: "
    f"{total_nan}"
)


# ============================================================
# 16. PARTICIPANTES E NOITES
# ============================================================

quantidade_participantes = (
    dataset_final[
        "participante"
    ]
    .nunique()
)


quantidade_noites = (
    dataset_final[
        [
            "participante",
            "noite"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


print(
    f"Participantes presentes: "
    f"{quantidade_participantes}"
)


print(
    f"Noites presentes: "
    f"{quantidade_noites}"
)


# ============================================================
# 17. SALVAR DATASET GERAL
# ============================================================

dataset_final.to_csv(
    ARQUIVO_DATASET,
    index=False
)


# ============================================================
# 18. RESULTADO FINAL
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "DATASET GERAL CRIADO"
)

print(
    "=" * 75
)


print(
    f"\nArquivo principal:"
)

print(
    ARQUIVO_DATASET
)


print(
    "\nArquivo com épocas excluídas:"
)

print(
    ARQUIVO_EXCLUIDAS
)


print(
    f"\nLinhas finais: "
    f"{len(dataset_final)}"
)


print(
    f"Colunas finais: "
    f"{len(dataset_final.columns)}"
)


print(
    "\n✅ DATASET PRONTO PARA "
    "A PRÓXIMA ETAPA"
)