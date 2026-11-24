from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_multiescala.csv"
)

ARQUIVO_DIVISAO = Path(
    "data/processed/divisao_participantes_ml.csv"
)

PASTA_RESULTADOS = Path(
    "data/processed/resultados_ml"
)


PASTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES DA ÉPOCA ATUAL
# ============================================================

FEATURES_ORIGINAIS = [
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",

    "motion_mean",
    "motion_std",
    "motion_max"
]


# ============================================================
# FEATURES DE 2,5 MINUTOS
# ============================================================

FEATURES_2_5_MIN = [
    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean"
]


# ============================================================
# FEATURES DE 5 MINUTOS
# ============================================================

FEATURES_5_MIN = [
    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",

    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean"
]


# ============================================================
# FEATURES DE 10 MINUTOS
# ============================================================

FEATURES_10_MIN = [
    "hr_mean_roll20_mean",
    "hr_mean_roll20_std",
    "hr_std_roll20_mean",

    "motion_mean_roll20_mean",
    "motion_mean_roll20_std",
    "motion_max_roll20_mean"
]


# ============================================================
# TODAS AS 27 FEATURES
# ============================================================

FEATURES = (
    FEATURES_ORIGINAIS
    + FEATURES_2_5_MIN
    + FEATURES_5_MIN
    + FEATURES_10_MIN
)


NOMES_ESTAGIOS = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM"
]


# ============================================================
# INÍCIO
# ============================================================

print("=" * 80)
print("VALIDAÇÃO CRUZADA POR PARTICIPANTE")
print("RANDOM FOREST MULTIESCALA")
print("=" * 80)


print(
    f"\nQuantidade de features: "
    f"{len(FEATURES)}"
)


print(
    "Número de folds: 6"
)


print(
    "\nCada participante ficará inteiramente "
    "em treino OU validação dentro de cada rodada."
)


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

print(
    "\nCarregando dataset multiescala..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


print(
    f"Épocas totais carregadas: "
    f"{len(dataset)}"
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


# ============================================================
# 2. CARREGAR DIVISÃO ORIGINAL
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


# ============================================================
# 3. IDENTIFICAR PARTICIPANTES DE TESTE
# ============================================================
#
# Esses três participantes NÃO entram
# na validação cruzada.
#
# Eles continuam guardados para o teste final.
#
# ============================================================

participantes_teste = (
    divisao[
        divisao["grupo"] == "teste"
    ]["participante"]
    .tolist()
)


print(
    "\nParticipantes reservados para teste final:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# 4. SEPARAR BASE DE DESENVOLVIMENTO
# ============================================================
#
# Desenvolvimento =
#
# treino original
# +
# validação original
#
# Portanto:
#
# 10 + 2 = 12 participantes
#
# ============================================================

dataset_dev = dataset[
    ~dataset[
        "participante"
    ].isin(
        participantes_teste
    )
].copy()


dataset_teste = dataset[
    dataset[
        "participante"
    ].isin(
        participantes_teste
    )
].copy()


participantes_dev = sorted(
    dataset_dev[
        "participante"
    ].unique()
)


print(
    "\n"
    + "=" * 80
)

print(
    "BASE DE DESENVOLVIMENTO"
)

print(
    "=" * 80
)


print(
    f"\nParticipantes: "
    f"{len(participantes_dev)}"
)


print(
    f"Épocas de desenvolvimento: "
    f"{len(dataset_dev)}"
)


print(
    f"Épocas reservadas para teste: "
    f"{len(dataset_teste)}"
)


if len(participantes_dev) != 12:

    raise ValueError(
        f"Esperávamos 12 participantes "
        f"de desenvolvimento, "
        f"mas encontramos "
        f"{len(participantes_dev)}."
    )


if len(dataset_teste) != 12061:

    raise ValueError(
        f"Esperávamos 12061 épocas "
        f"reservadas para teste, "
        f"mas encontramos "
        f"{len(dataset_teste)}."
    )


# ============================================================
# 5. VALIDAR FEATURES
# ============================================================

features_faltando = [
    feature
    for feature in FEATURES
    if feature not in dataset_dev.columns
]


if features_faltando:

    raise ValueError(
        "Features ausentes:\n"
        + "\n".join(
            features_faltando
        )
    )


ausentes = (
    dataset_dev[
        FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


if ausentes != 0:

    raise ValueError(
        f"Existem {ausentes} valores "
        f"ausentes nas features."
    )


print(
    "\n✅ Features validadas."
)


# ============================================================
# 6. DEFINIR X, Y E GRUPOS
# ============================================================
#
# X:
# características utilizadas pelo modelo
#
# y:
# estágio correto
#
# grupos:
# participante
#
# O participante é usado SOMENTE para
# controlar a separação dos folds.
#
# Ele NÃO é uma feature do modelo.
#
# ============================================================

X = dataset_dev[
    FEATURES
]


y = dataset_dev[
    "label"
]


grupos = dataset_dev[
    "participante"
]


# ============================================================
# 7. CRIAR VALIDAÇÃO CRUZADA
# ============================================================
#
# StratifiedGroupKFold:
#
# GROUP:
# impede que um participante apareça
# simultaneamente em treino e validação.
#
# STRATIFIED:
# tenta manter a distribuição das classes
# razoavelmente equilibrada entre os folds.
#
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=42
)


# ============================================================
# 8. LISTAS PARA GUARDAR RESULTADOS
# ============================================================

resultados_folds = []

resultados_classes = []


# ============================================================
# 9. EXECUTAR OS 6 FOLDS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "INICIANDO VALIDAÇÃO CRUZADA"
)

print(
    "=" * 80
)


for fold, (
    indices_treino,
    indices_validacao
) in enumerate(
    cv.split(
        X,
        y,
        groups=grupos
    ),
    start=1
):


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"FOLD {fold}/6"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # DADOS DO FOLD
    # ========================================================

    X_treino = X.iloc[
        indices_treino
    ]


    y_treino = y.iloc[
        indices_treino
    ]


    X_validacao = X.iloc[
        indices_validacao
    ]


    y_validacao = y.iloc[
        indices_validacao
    ]


    grupos_treino = grupos.iloc[
        indices_treino
    ]


    grupos_validacao = grupos.iloc[
        indices_validacao
    ]


    participantes_fold_treino = sorted(
        grupos_treino.unique()
    )


    participantes_fold_validacao = sorted(
        grupos_validacao.unique()
    )


    # ========================================================
    # GARANTIR AUSÊNCIA DE VAZAMENTO
    # ========================================================

    intersecao = (
        set(participantes_fold_treino)
        &
        set(participantes_fold_validacao)
    )


    if intersecao:

        raise ValueError(
            f"Vazamento de participante "
            f"detectado no fold {fold}: "
            f"{intersecao}"
        )


    print(
        "\nParticipantes de validação:"
    )


    for participante in (
        participantes_fold_validacao
    ):

        print(
            f"  - {participante}"
        )


    print(
        f"\nÉpocas de treino: "
        f"{len(X_treino)}"
    )


    print(
        f"Épocas de validação: "
        f"{len(X_validacao)}"
    )


    # ========================================================
    # CRIAR MODELO
    # ========================================================
    #
    # Estamos usando exatamente a configuração
    # do nosso melhor Random Forest multiescala.
    #
    # ========================================================

    modelo = RandomForestClassifier(

        n_estimators=300,

        max_depth=12,

        min_samples_split=20,

        min_samples_leaf=10,

        max_features="sqrt",

        class_weight="balanced_subsample",

        random_state=42,

        n_jobs=-1
    )


    # ========================================================
    # TREINAR
    # ========================================================

    print(
        "\nTreinando..."
    )


    modelo.fit(
        X_treino,
        y_treino
    )


    # ========================================================
    # PREVER
    # ========================================================

    pred_validacao = modelo.predict(
        X_validacao
    )


    # ========================================================
    # MÉTRICAS GERAIS
    # ========================================================

    accuracy = accuracy_score(
        y_validacao,
        pred_validacao
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_validacao,
            pred_validacao
        )
    )


    macro_f1 = f1_score(
        y_validacao,
        pred_validacao,
        average="macro"
    )


    weighted_f1 = f1_score(
        y_validacao,
        pred_validacao,
        average="weighted"
    )


    print(
        "\nResultados:"
    )


    print(
        f"Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )


    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )


    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )


    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )


    # ========================================================
    # GUARDAR RESULTADO DO FOLD
    # ========================================================

    resultados_folds.append({

        "fold":
            fold,

        "participantes_validacao":
            ", ".join(
                participantes_fold_validacao
            ),

        "epocas_treino":
            len(X_treino),

        "epocas_validacao":
            len(X_validacao),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1
    })


    # ========================================================
    # RESULTADO POR CLASSE
    # ========================================================

    relatorio = classification_report(

        y_validacao,
        pred_validacao,

        labels=[
            0,
            1,
            2,
            3,
            4
        ],

        target_names=
            NOMES_ESTAGIOS,

        output_dict=True,

        zero_division=0
    )


    for nome_estagio in NOMES_ESTAGIOS:

        resultados_classes.append({

            "fold":
                fold,

            "estagio":
                nome_estagio,

            "precision":
                relatorio[
                    nome_estagio
                ]["precision"],

            "recall":
                relatorio[
                    nome_estagio
                ]["recall"],

            "f1":
                relatorio[
                    nome_estagio
                ]["f1-score"],

            "support":
                relatorio[
                    nome_estagio
                ]["support"]
        })


# ============================================================
# 10. TRANSFORMAR RESULTADOS EM DATAFRAME
# ============================================================

df_folds = pd.DataFrame(
    resultados_folds
)


df_classes = pd.DataFrame(
    resultados_classes
)


# ============================================================
# 11. RESULTADO FINAL DOS 6 FOLDS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RESULTADO FINAL DA VALIDAÇÃO CRUZADA"
)

print(
    "=" * 80
)


print(
    "\nResultados por fold:"
)


print(
    df_folds[
        [
            "fold",
            "participantes_validacao",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. MÉDIA E DESVIO PADRÃO
# ============================================================

metricas = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1"
]


print(
    "\n"
    + "=" * 80
)

print(
    "MÉDIA DOS 6 FOLDS"
)

print(
    "=" * 80
)


for metrica in metricas:

    media = df_folds[
        metrica
    ].mean()


    desvio = df_folds[
        metrica
    ].std()


    print(
        f"\n{metrica}:"
    )


    print(
        f"  Média = "
        f"{media:.4f}"
    )


    print(
        f"  Desvio padrão = "
        f"{desvio:.4f}"
    )


# ============================================================
# 13. MÉDIA POR ESTÁGIO
# ============================================================

media_classes = (
    df_classes

    .groupby(
        "estagio"
    )[
        [
            "precision",
            "recall",
            "f1"
        ]
    ]

    .mean()

    .reindex(
        NOMES_ESTAGIOS
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "DESEMPENHO MÉDIO POR ESTÁGIO"
)

print(
    "=" * 80
)


print(
    "\n"
    + media_classes
    .round(4)
    .to_string()
)


# ============================================================
# 14. IDENTIFICAR MELHOR E PIOR FOLD
# ============================================================

indice_melhor = (
    df_folds[
        "macro_f1"
    ]
    .idxmax()
)


indice_pior = (
    df_folds[
        "macro_f1"
    ]
    .idxmin()
)


melhor_fold = df_folds.loc[
    indice_melhor
]


pior_fold = df_folds.loc[
    indice_pior
]


print(
    "\n"
    + "=" * 80
)

print(
    "VARIAÇÃO ENTRE PARTICIPANTES"
)

print(
    "=" * 80
)


print(
    f"\nMelhor fold: "
    f"{int(melhor_fold['fold'])}"
)


print(
    f"Participantes: "
    f"{melhor_fold['participantes_validacao']}"
)


print(
    f"Macro F1: "
    f"{melhor_fold['macro_f1']:.4f}"
)


print(
    f"\nPior fold: "
    f"{int(pior_fold['fold'])}"
)


print(
    f"Participantes: "
    f"{pior_fold['participantes_validacao']}"
)


print(
    f"Macro F1: "
    f"{pior_fold['macro_f1']:.4f}"
)


# ============================================================
# 15. SALVAR RESULTADOS
# ============================================================

ARQUIVO_FOLDS = (
    PASTA_RESULTADOS
    / "validacao_cruzada_rf_multiescala_folds.csv"
)


ARQUIVO_CLASSES = (
    PASTA_RESULTADOS
    / "validacao_cruzada_rf_multiescala_classes.csv"
)


ARQUIVO_MEDIA_CLASSES = (
    PASTA_RESULTADOS
    / "validacao_cruzada_rf_multiescala_media_classes.csv"
)


df_folds.to_csv(
    ARQUIVO_FOLDS,
    index=False
)


df_classes.to_csv(
    ARQUIVO_CLASSES,
    index=False
)


media_classes.to_csv(
    ARQUIVO_MEDIA_CLASSES
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "VALIDAÇÃO CRUZADA FINALIZADA"
)

print(
    "=" * 80
)


print(
    "\nArquivos criados:"
)


print(
    ARQUIVO_FOLDS
)


print(
    ARQUIVO_CLASSES
)


print(
    ARQUIVO_MEDIA_CLASSES
)


print(
    "\n⚠️ Os participantes de teste "
    "NÃO foram utilizados."
)


print(
    "\n✅ EXPERIMENTO CONCLUÍDO"
)