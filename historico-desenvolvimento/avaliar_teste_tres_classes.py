from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO = Path(
    "data/processed/resultados_ml/teste_final_predicoes.csv"
)

PASTA_RESULTADOS = Path(
    "data/processed/resultados_ml"
)

PASTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
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
# TRÊS CLASSES:
#
# 0 = Wake
# 1 = NREM  (N1 + N2 + N3)
# 2 = REM
#
# ============================================================

MAPA = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2
}


NOMES = {
    0: "Wake",
    1: "NREM",
    2: "REM"
}


CLASSES = [
    0,
    1,
    2
]


# ============================================================
# CARREGAR
# ============================================================

print("=" * 95)
print("ANÁLISE SECUNDÁRIA DO TESTE FINAL - 3 CLASSES")
print("=" * 95)


if not ARQUIVO.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado:\n{ARQUIVO}"
    )


df = pd.read_csv(
    ARQUIVO
)


print(
    f"\nÉpocas encontradas: {len(df)}"
)


if len(df) != 12061:

    raise ValueError(
        "Esperávamos exatamente 12061 épocas."
    )


if (
    df[
        [
            "label",
            "predicao"
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "Existem valores ausentes."
    )


df["label"] = df["label"].astype(int)
df["predicao"] = df["predicao"].astype(int)


# ============================================================
# CINCO CLASSES
# ============================================================

y5 = df[
    "label"
].to_numpy()


p5 = df[
    "predicao"
].to_numpy()


accuracy_5 = accuracy_score(
    y5,
    p5
)


balanced_5 = balanced_accuracy_score(
    y5,
    p5
)


macro_5 = f1_score(
    y5,
    p5,
    average="macro"
)


# ============================================================
# TRÊS CLASSES
# ============================================================

df[
    "label_3"
] = df[
    "label"
].map(
    MAPA
)


df[
    "predicao_3"
] = df[
    "predicao"
].map(
    MAPA
)


y3 = df[
    "label_3"
].to_numpy()


p3 = df[
    "predicao_3"
].to_numpy()


accuracy_3 = accuracy_score(
    y3,
    p3
)


balanced_3 = balanced_accuracy_score(
    y3,
    p3
)


macro_3 = f1_score(
    y3,
    p3,
    average="macro"
)


weighted_3 = f1_score(
    y3,
    p3,
    average="weighted"
)


# ============================================================
# RESULTADO GERAL
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "RESULTADO"
)

print(
    "=" * 95
)


print(
    "\n5 CLASSES"
)

print(
    f"Accuracy:          "
    f"{accuracy_5:.4f} "
    f"({accuracy_5 * 100:.2f}%)"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_5:.4f}"
)

print(
    f"Macro F1:          "
    f"{macro_5:.4f}"
)


print(
    "\n3 CLASSES"
)

print(
    f"Accuracy:          "
    f"{accuracy_3:.4f} "
    f"({accuracy_3 * 100:.2f}%)"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_3:.4f} "
    f"({balanced_3 * 100:.2f}%)"
)

print(
    f"Macro F1:          "
    f"{macro_3:.4f}"
)

print(
    f"Weighted F1:       "
    f"{weighted_3:.4f}"
)


print(
    "\nGANHO:"
)

print(
    f"Accuracy: "
    f"{(accuracy_3 - accuracy_5) * 100:+.2f} "
    f"pontos percentuais"
)


# ============================================================
# POR CLASSE
# ============================================================

(
    precision,
    recall,
    f1_classes,
    suporte

) = precision_recall_fscore_support(

    y3,
    p3,
    labels=CLASSES,
    zero_division=0
)


print(
    "\n"
    + "=" * 95
)

print(
    "DESEMPENHO POR CLASSE"
)

print(
    "=" * 95
)


resultados_classes = []


for i, classe in enumerate(
    CLASSES
):

    print(
        f"\n{NOMES[classe]}"
    )

    print(
        f"  Precision: "
        f"{precision[i]:.4f}"
    )

    print(
        f"  Recall:    "
        f"{recall[i]:.4f}"
    )

    print(
        f"  F1:        "
        f"{f1_classes[i]:.4f}"
    )

    print(
        f"  Support:   "
        f"{int(suporte[i])}"
    )


    resultados_classes.append({

        "classe":
            classe,

        "nome":
            NOMES[
                classe
            ],

        "precision":
            precision[i],

        "recall":
            recall[i],

        "f1":
            f1_classes[i],

        "support":
            int(
                suporte[i]
            )
    })


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

matriz = confusion_matrix(

    y3,
    p3,
    labels=CLASSES
)


print(
    "\n"
    + "=" * 95
)

print(
    "MATRIZ DE CONFUSÃO"
)

print(
    "=" * 95
)


print(
    "\nOrdem:"
)

print(
    "Wake | NREM | REM"
)


print(
    "\n"
    + str(
        matriz
    )
)


# ============================================================
# POR PARTICIPANTE
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "RESULTADO POR PARTICIPANTE"
)

print(
    "=" * 95
)


resultados_participantes = []


for participante, grupo in df.groupby(
    "participante"
):

    real = grupo[
        "label_3"
    ].to_numpy()

    previsto = grupo[
        "predicao_3"
    ].to_numpy()


    acc = accuracy_score(
        real,
        previsto
    )


    bal = balanced_accuracy_score(
        real,
        previsto
    )


    macro = f1_score(
        real,
        previsto,
        average="macro"
    )


    print(
        f"\n{participante}"
    )


    print(
        f"  Accuracy: "
        f"{acc:.4f} "
        f"({acc * 100:.2f}%)"
    )


    print(
        f"  Balanced Accuracy: "
        f"{bal:.4f}"
    )


    print(
        f"  Macro F1: "
        f"{macro:.4f}"
    )


    resultados_participantes.append({

        "participante":
            participante,

        "epocas":
            len(
                grupo
            ),

        "accuracy":
            acc,

        "balanced_accuracy":
            bal,

        "macro_f1":
            macro
    })


# ============================================================
# SALVAR
# ============================================================

pd.DataFrame(
    resultados_classes
).to_csv(

    PASTA_RESULTADOS
    / "teste_3_classes_por_classe.csv",

    index=False
)


pd.DataFrame(

    matriz,

    index=[
        NOMES[c]
        for c in CLASSES
    ],

    columns=[
        NOMES[c]
        for c in CLASSES
    ]

).to_csv(

    PASTA_RESULTADOS
    / "teste_3_classes_matriz_confusao.csv"
)


pd.DataFrame(
    resultados_participantes
).to_csv(

    PASTA_RESULTADOS
    / "teste_3_classes_participantes.csv",

    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "ANÁLISE CONCLUÍDA"
)

print(
    "=" * 95
)


print(
    f"\nAccuracy final em 3 classes: "
    f"{accuracy_3 * 100:.2f}%"
)


print(
    f"Balanced Accuracy: "
    f"{balanced_3 * 100:.2f}%"
)


print(
    f"Macro F1: "
    f"{macro_3:.4f}"
)


print(
    "\nNenhum modelo foi retreinado."
)

print(
    "Foram apenas reagrupadas as classes "
    "das previsões finais já existentes."
)