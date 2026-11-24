from pathlib import Path

import numpy as np
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

ARQUIVO_PREDICOES = Path(
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
# CONFIGURAÇÃO
# ============================================================
#
# 5 CLASSES ORIGINAIS:
#
# 0 = Wake
# 1 = N1
# 2 = N2
# 3 = N3
# 4 = REM
#
#
# 4 CLASSES:
#
# 0 = Wake
# 1 = Light   (N1 + N2)
# 2 = Deep    (N3)
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


NOMES_4_CLASSES = {
    0: "Wake",
    1: "Light (N1 + N2)",
    2: "Deep (N3)",
    3: "REM"
}


CLASSES_4 = [
    0,
    1,
    2,
    3
]


QUANTIDADE_ESPERADA = 12061


# ============================================================
# CABEÇALHO
# ============================================================

print("=" * 95)
print("ANÁLISE SECUNDÁRIA DO TESTE FINAL - 4 CLASSES")
print("=" * 95)

print(
    "\nMapeamento:"
)

print(
    "  Wake       -> Wake"
)

print(
    "  N1 + N2    -> Light"
)

print(
    "  N3         -> Deep"
)

print(
    "  REM        -> REM"
)


# ============================================================
# CARREGAR ARQUIVO
# ============================================================

if not ARQUIVO_PREDICOES.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado:\n"
        f"{ARQUIVO_PREDICOES}"
    )


print(
    "\nCarregando teste final..."
)


df = pd.read_csv(
    ARQUIVO_PREDICOES
)


print(
    f"Épocas encontradas: "
    f"{len(df)}"
)


if len(df) != QUANTIDADE_ESPERADA:

    raise ValueError(
        f"Esperávamos {QUANTIDADE_ESPERADA} épocas, "
        f"mas encontramos {len(df)}."
    )


# ============================================================
# VALIDAR COLUNAS
# ============================================================

COLUNAS_NECESSARIAS = [
    "participante",
    "label",
    "predicao"
]


faltando = [
    coluna
    for coluna in COLUNAS_NECESSARIAS
    if coluna not in df.columns
]


if faltando:

    raise ValueError(
        "Colunas ausentes:\n"
        + "\n".join(
            faltando
        )
    )


if df[
    [
        "label",
        "predicao"
    ]
].isna().any().any():

    raise ValueError(
        "Existem valores ausentes "
        "em label ou predicao."
    )


df[
    "label"
] = df[
    "label"
].astype(
    int
)


df[
    "predicao"
] = df[
    "predicao"
].astype(
    int
)


classes_encontradas = set(

    df[
        "label"
    ].unique()
).union(

    set(
        df[
            "predicao"
        ].unique()
    )
)


if not classes_encontradas.issubset(
    {
        0,
        1,
        2,
        3,
        4
    }
):

    raise ValueError(
        f"Classes inesperadas encontradas: "
        f"{classes_encontradas}"
    )


print(
    "✅ Arquivo validado."
)


# ============================================================
# RESULTADOS ORIGINAIS EM 5 CLASSES
# ============================================================

y_real_5 = df[
    "label"
].to_numpy()


y_pred_5 = df[
    "predicao"
].to_numpy()


accuracy_5 = accuracy_score(
    y_real_5,
    y_pred_5
)


balanced_5 = balanced_accuracy_score(
    y_real_5,
    y_pred_5
)


macro_5 = f1_score(
    y_real_5,
    y_pred_5,
    average="macro"
)


weighted_5 = f1_score(
    y_real_5,
    y_pred_5,
    average="weighted"
)


# ============================================================
# CONVERTER PARA 4 CLASSES
# ============================================================

df[
    "label_4"
] = df[
    "label"
].map(
    MAPA_4_CLASSES
)


df[
    "predicao_4"
] = df[
    "predicao"
].map(
    MAPA_4_CLASSES
)


if df[
    [
        "label_4",
        "predicao_4"
    ]
].isna().any().any():

    raise RuntimeError(
        "Erro no mapeamento das quatro classes."
    )


df[
    "label_4"
] = df[
    "label_4"
].astype(
    int
)


df[
    "predicao_4"
] = df[
    "predicao_4"
].astype(
    int
)


y_real_4 = df[
    "label_4"
].to_numpy()


y_pred_4 = df[
    "predicao_4"
].to_numpy()


# ============================================================
# MÉTRICAS EM 4 CLASSES
# ============================================================

accuracy_4 = accuracy_score(
    y_real_4,
    y_pred_4
)


balanced_4 = balanced_accuracy_score(
    y_real_4,
    y_pred_4
)


macro_4 = f1_score(
    y_real_4,
    y_pred_4,
    average="macro"
)


weighted_4 = f1_score(
    y_real_4,
    y_pred_4,
    average="weighted"
)


# ============================================================
# COMPARAÇÃO PRINCIPAL
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "COMPARAÇÃO 5 CLASSES x 4 CLASSES"
)

print(
    "=" * 95
)


print(
    "\n5 CLASSES:"
)

print(
    f"  Accuracy:          "
    f"{accuracy_5:.4f} "
    f"({accuracy_5 * 100:.2f}%)"
)

print(
    f"  Balanced Accuracy: "
    f"{balanced_5:.4f} "
    f"({balanced_5 * 100:.2f}%)"
)

print(
    f"  Macro F1:          "
    f"{macro_5:.4f}"
)

print(
    f"  Weighted F1:       "
    f"{weighted_5:.4f}"
)


print(
    "\n4 CLASSES:"
)

print(
    f"  Accuracy:          "
    f"{accuracy_4:.4f} "
    f"({accuracy_4 * 100:.2f}%)"
)

print(
    f"  Balanced Accuracy: "
    f"{balanced_4:.4f} "
    f"({balanced_4 * 100:.2f}%)"
)

print(
    f"  Macro F1:          "
    f"{macro_4:.4f}"
)

print(
    f"  Weighted F1:       "
    f"{weighted_4:.4f}"
)


print(
    "\nGANHO COM O AGRUPAMENTO:"
)

print(
    f"  Accuracy: "
    f"{(accuracy_4 - accuracy_5) * 100:+.2f} "
    f"pontos percentuais"
)

print(
    f"  Balanced Accuracy: "
    f"{(balanced_4 - balanced_5) * 100:+.2f} "
    f"pontos percentuais"
)

print(
    f"  Macro F1: "
    f"{macro_4 - macro_5:+.4f}"
)


# ============================================================
# ERROS N1 <-> N2 QUE DEIXAM DE SER ERROS
# ============================================================

erro_n1_para_n2 = (

    (
        df[
            "label"
        ] == 1
    )
    &
    (
        df[
            "predicao"
        ] == 2
    )
).sum()


erro_n2_para_n1 = (

    (
        df[
            "label"
        ] == 2
    )
    &
    (
        df[
            "predicao"
        ] == 1
    )
).sum()


correcoes_light = int(
    erro_n1_para_n2
    + erro_n2_para_n1
)


print(
    "\n"
    + "=" * 95
)

print(
    "IMPACTO DO AGRUPAMENTO N1 + N2"
)

print(
    "=" * 95
)


print(
    f"\nN1 real -> N2 previsto: "
    f"{int(erro_n1_para_n2)}"
)


print(
    f"N2 real -> N1 previsto: "
    f"{int(erro_n2_para_n1)}"
)


print(
    f"\nErros de 5 classes que passam "
    f"a ser acertos em Light:"
)

print(
    f"  {correcoes_light} épocas"
)


print(
    f"  equivalente a "
    f"{correcoes_light / len(df) * 100:.2f}% "
    f"de todo o teste"
)


# ============================================================
# RESULTADO POR CLASSE
# ============================================================

(
    precision,
    recall,
    f1_classes,
    suporte

) = precision_recall_fscore_support(

    y_real_4,
    y_pred_4,
    labels=CLASSES_4,
    zero_division=0
)


print(
    "\n"
    + "=" * 95
)

print(
    "RESULTADO POR CLASSE - 4 CLASSES"
)

print(
    "=" * 95
)


dados_classes = []


for i, classe in enumerate(
    CLASSES_4
):

    nome = NOMES_4_CLASSES[
        classe
    ]


    print(
        f"\n{nome}"
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


    dados_classes.append({

        "classe":
            classe,

        "nome_classe":
            nome,

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

    y_real_4,

    y_pred_4,

    labels=CLASSES_4
)


print(
    "\n"
    + "=" * 95
)

print(
    "MATRIZ DE CONFUSÃO - 4 CLASSES"
)

print(
    "=" * 95
)


print(
    "\nOrdem:"
)

print(
    "Wake | Light | Deep | REM"
)


print(
    "\n"
    + str(
        matriz
    )
)


# ============================================================
# RESULTADO POR PARTICIPANTE
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "RESULTADO POR PARTICIPANTE - 4 CLASSES"
)

print(
    "=" * 95
)


dados_participantes = []


for participante, grupo in df.groupby(
    "participante"
):

    y_real_p = grupo[
        "label_4"
    ].to_numpy()


    y_pred_p = grupo[
        "predicao_4"
    ].to_numpy()


    acc_p = accuracy_score(
        y_real_p,
        y_pred_p
    )


    bal_p = balanced_accuracy_score(
        y_real_p,
        y_pred_p
    )


    macro_p = f1_score(
        y_real_p,
        y_pred_p,
        average="macro"
    )


    weighted_p = f1_score(
        y_real_p,
        y_pred_p,
        average="weighted"
    )


    print(
        f"\n{participante}"
    )


    print(
        f"  Accuracy:          "
        f"{acc_p:.4f} "
        f"({acc_p * 100:.2f}%)"
    )


    print(
        f"  Balanced Accuracy: "
        f"{bal_p:.4f}"
    )


    print(
        f"  Macro F1:          "
        f"{macro_p:.4f}"
    )


    print(
        f"  Weighted F1:       "
        f"{weighted_p:.4f}"
    )


    dados_participantes.append({

        "participante":
            participante,

        "epocas":
            len(
                grupo
            ),

        "accuracy":
            acc_p,

        "balanced_accuracy":
            bal_p,

        "macro_f1":
            macro_p,

        "weighted_f1":
            weighted_p
    })


# ============================================================
# SALVAR RESULTADOS
# ============================================================

resumo = pd.DataFrame(
    [
        {
            "configuracao":
                "5_classes",

            "accuracy":
                accuracy_5,

            "balanced_accuracy":
                balanced_5,

            "macro_f1":
                macro_5,

            "weighted_f1":
                weighted_5
        },

        {
            "configuracao":
                "4_classes_N1_N2_light",

            "accuracy":
                accuracy_4,

            "balanced_accuracy":
                balanced_4,

            "macro_f1":
                macro_4,

            "weighted_f1":
                weighted_4
        }
    ]
)


resumo.to_csv(

    PASTA_RESULTADOS
    / "comparacao_teste_5_vs_4_classes.csv",

    index=False
)


pd.DataFrame(
    dados_classes
).to_csv(

    PASTA_RESULTADOS
    / "teste_4_classes_por_classe.csv",

    index=False
)


pd.DataFrame(

    matriz,

    index=[
        NOMES_4_CLASSES[
            classe
        ]
        for classe in CLASSES_4
    ],

    columns=[
        NOMES_4_CLASSES[
            classe
        ]
        for classe in CLASSES_4
    ]

).to_csv(

    PASTA_RESULTADOS
    / "teste_4_classes_matriz_confusao.csv"
)


pd.DataFrame(
    dados_participantes
).to_csv(

    PASTA_RESULTADOS
    / "teste_4_classes_participantes.csv",

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
    "ANÁLISE SECUNDÁRIA CONCLUÍDA"
)

print(
    "=" * 95
)


print(
    f"\nRESULTADO PRINCIPAL - 5 CLASSES:"
)

print(
    f"Accuracy = "
    f"{accuracy_5 * 100:.2f}%"
)


print(
    f"\nRESULTADO SECUNDÁRIO - 4 CLASSES:"
)

print(
    f"Accuracy = "
    f"{accuracy_4 * 100:.2f}%"
)


print(
    f"Macro F1 = "
    f"{macro_4:.4f}"
)


print(
    "\nArquivos salvos em:"
)

print(
    PASTA_RESULTADOS
)


print(
    "\nIMPORTANTE:"
)

print(
    "Esta é uma análise secundária das mesmas "
    "predições finais."
)

print(
    "O modelo NÃO foi retreinado ou ajustado "
    "com base no teste."
)