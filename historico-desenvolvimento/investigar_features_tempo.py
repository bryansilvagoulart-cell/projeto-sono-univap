from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
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
# 27 FEATURES BASE
# ============================================================

FEATURES_BASE = [

    # ÉPOCA ATUAL
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",

    "motion_mean",
    "motion_std",
    "motion_max",

    # 2,5 MINUTOS
    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean",

    # 5 MINUTOS
    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",

    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean",

    # 10 MINUTOS
    "hr_mean_roll20_mean",
    "hr_mean_roll20_std",
    "hr_std_roll20_mean",

    "motion_mean_roll20_mean",
    "motion_mean_roll20_std",
    "motion_max_roll20_mean"
]


# ============================================================
# FEATURES DE TEMPO
# ============================================================

FEATURE_TEMPO_DECORRIDO = [
    "tempo_desde_inicio_min"
]


FEATURES_RELOGIO = [
    "hora_local_sin",
    "hora_local_cos"
]


# ============================================================
# EXPERIMENTOS
# ============================================================

EXPERIMENTOS = {

    "A_BASE":
        FEATURES_BASE,

    "B_BASE_TEMPO_DECORRIDO":
        FEATURES_BASE
        + FEATURE_TEMPO_DECORRIDO,

    "C_BASE_HORARIO_RELOGIO":
        FEATURES_BASE
        + FEATURES_RELOGIO,

    "D_BASE_TODAS_TEMPO":
        FEATURES_BASE
        + FEATURE_TEMPO_DECORRIDO
        + FEATURES_RELOGIO
}


# ============================================================
# INÍCIO
# ============================================================

print("=" * 85)
print("INVESTIGAÇÃO DAS FEATURES TEMPORAIS")
print("=" * 85)


for nome, features in EXPERIMENTOS.items():

    print(
        f"{nome}: "
        f"{len(features)} features"
    )


# ============================================================
# CARREGAR DATASET
# ============================================================

print("\nCarregando dataset...")


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


print(
    f"Épocas totais: {len(dataset)}"
)


# ============================================================
# CARREGAR DIVISÃO
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


participantes_teste = (

    divisao[
        divisao["grupo"] == "teste"
    ]

    ["participante"]

    .tolist()
)


print(
    "\nTeste final continua reservado:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# DESENVOLVIMENTO
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


print(
    f"\nDesenvolvimento: "
    f"{len(dataset_dev)} épocas"
)


print(
    f"Teste reservado: "
    f"{len(dataset_teste)} épocas"
)


if len(dataset_teste) != 12061:

    raise ValueError(
        "Quantidade inesperada "
        "no teste final."
    )


# ============================================================
# VALIDAR FEATURES
# ============================================================

todas_features = list(
    dict.fromkeys(
        FEATURES_BASE
        + FEATURE_TEMPO_DECORRIDO
        + FEATURES_RELOGIO
    )
)


faltando = [
    feature
    for feature in todas_features
    if feature not in dataset_dev.columns
]


if faltando:

    raise ValueError(
        "Features ausentes:\n"
        + "\n".join(faltando)
    )


ausentes = (
    dataset_dev[
        todas_features
    ]

    .isna()

    .sum()

    .sum()
)


if ausentes != 0:

    raise ValueError(
        f"Existem {ausentes} "
        f"valores ausentes."
    )


print(
    "✅ Dados validados."
)


# ============================================================
# LABEL E GRUPOS
# ============================================================

y = dataset_dev[
    "label"
]


grupos = dataset_dev[
    "participante"
]


# ============================================================
# MESMOS 6 FOLDS
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=42
)


folds = list(
    cv.split(
        dataset_dev[
            FEATURES_BASE
        ],
        y,
        groups=grupos
    )
)


# ============================================================
# RESULTADOS
# ============================================================

resultados = []


# ============================================================
# EXECUTAR EXPERIMENTOS
# ============================================================

for numero, (
    nome,
    features
) in enumerate(
    EXPERIMENTOS.items(),
    start=1
):


    print(
        "\n"
        + "=" * 85
    )

    print(
        f"EXPERIMENTO "
        f"{numero}/"
        f"{len(EXPERIMENTOS)}"
    )

    print(nome)

    print(
        f"{len(features)} FEATURES"
    )

    print(
        "=" * 85
    )


    X = dataset_dev[
        features
    ]


    for fold, (
        idx_treino,
        idx_validacao
    ) in enumerate(
        folds,
        start=1
    ):


        X_treino = X.iloc[
            idx_treino
        ]


        y_treino = y.iloc[
            idx_treino
        ]


        X_validacao = X.iloc[
            idx_validacao
        ]


        y_validacao = y.iloc[
            idx_validacao
        ]


        participantes_validacao = sorted(

            grupos.iloc[
                idx_validacao
            ]

            .unique()
        )


        print(
            f"\nFold {fold}/6"
        )


        print(
            "Validação: "
            + ", ".join(
                participantes_validacao
            )
        )


        # ====================================================
        # MODELO
        # ====================================================

        modelo = RandomForestClassifier(

            n_estimators=300,

            max_depth=16,

            min_samples_split=10,

            min_samples_leaf=5,

            max_features="sqrt",

            class_weight="balanced_subsample",

            random_state=42,

            n_jobs=-1
        )


        modelo.fit(
            X_treino,
            y_treino
        )


        pred = modelo.predict(
            X_validacao
        )


        accuracy = accuracy_score(
            y_validacao,
            pred
        )


        balanced = balanced_accuracy_score(
            y_validacao,
            pred
        )


        macro_f1 = f1_score(
            y_validacao,
            pred,
            average="macro"
        )


        weighted_f1 = f1_score(
            y_validacao,
            pred,
            average="weighted"
        )


        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )


        print(
            f"Balanced Accuracy: "
            f"{balanced:.4f}"
        )


        print(
            f"Macro F1: "
            f"{macro_f1:.4f}"
        )


        resultados.append({

            "experimento":
                nome,

            "quantidade_features":
                len(features),

            "fold":
                fold,

            "participantes_validacao":
                ", ".join(
                    participantes_validacao
                ),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced,

            "macro_f1":
                macro_f1,

            "weighted_f1":
                weighted_f1
        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    resultados
)


# ============================================================
# RESUMO
# ============================================================

resumo = (

    df

    .groupby(
        [
            "experimento",
            "quantidade_features"
        ]
    )

    .agg(

        accuracy_media=(
            "accuracy",
            "mean"
        ),

        balanced_accuracy_media=(
            "balanced_accuracy",
            "mean"
        ),

        macro_f1_media=(
            "macro_f1",
            "mean"
        ),

        macro_f1_desvio=(
            "macro_f1",
            "std"
        ),

        weighted_f1_media=(
            "weighted_f1",
            "mean"
        )
    )

    .reset_index()

    .sort_values(
        "macro_f1_media",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# RESULTADO
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "RESULTADO FINAL"
)

print(
    "=" * 85
)


print(

    "\n"

    + resumo[
        [
            "experimento",
            "quantidade_features",
            "accuracy_media",
            "balanced_accuracy_media",
            "macro_f1_media",
            "macro_f1_desvio"
        ]
    ]

    .round(4)

    .to_string(
        index=False
    )
)


# ============================================================
# GANHO CONTRA BASE
# ============================================================

base = resumo[
    resumo[
        "experimento"
    ] == "A_BASE"
].iloc[0]


print(
    "\n"
    + "=" * 85
)

print(
    "GANHO CONTRA A BASE"
)

print(
    "=" * 85
)


for _, linha in resumo.iterrows():

    ganho_accuracy = (
        linha[
            "accuracy_media"
        ]
        - base[
            "accuracy_media"
        ]
    )


    ganho_macro = (
        linha[
            "macro_f1_media"
        ]
        - base[
            "macro_f1_media"
        ]
    )


    print(
        f"\n{linha['experimento']}"
    )


    print(
        f"  Accuracy: "
        f"{ganho_accuracy:+.4f}"
    )


    print(
        f"  Macro F1: "
        f"{ganho_macro:+.4f}"
    )


# ============================================================
# MELHOR
# ============================================================

melhor = resumo.iloc[0]


print(
    "\n"
    + "=" * 85
)

print(
    "MELHOR CONFIGURAÇÃO TEMPORAL"
)

print(
    "=" * 85
)


print(
    f"\n{melhor['experimento']}"
)


print(
    f"Accuracy média: "
    f"{melhor['accuracy_media']:.4f}"
)


print(
    f"Balanced Accuracy média: "
    f"{melhor['balanced_accuracy_media']:.4f}"
)


print(
    f"Macro F1 médio: "
    f"{melhor['macro_f1_media']:.4f}"
)


# ============================================================
# SALVAR
# ============================================================

ARQUIVO_FOLDS = (
    PASTA_RESULTADOS
    / "investigacao_features_tempo_folds.csv"
)


ARQUIVO_RESUMO = (
    PASTA_RESULTADOS
    / "investigacao_features_tempo_resumo.csv"
)


df.to_csv(
    ARQUIVO_FOLDS,
    index=False
)


resumo.to_csv(
    ARQUIVO_RESUMO,
    index=False
)


# ============================================================
# FIM
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "INVESTIGAÇÃO FINALIZADA"
)

print(
    "=" * 85
)


print(
    "\n⚠️ O teste final continua intocado."
)


print(
    "\n✅ EXPERIMENTO CONCLUÍDO"
)