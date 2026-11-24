from pathlib import Path
import time

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

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
# FEATURES ORIGINAIS
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
# CONTEXTO DE 2,5 MINUTOS
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
# CONTEXTO DE 5 MINUTOS
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
# CONTEXTO DE 10 MINUTOS
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
# FREQUÊNCIA DE AMOSTRAGEM
# ============================================================

FEATURES_FREQ = [
    "hr_sampling_hz",
    "hr_sampling_delta1",

    "hr_sampling_roll5_mean",
    "hr_sampling_roll5_std",

    "hr_sampling_roll10_mean",
    "hr_sampling_roll10_std",

    "hr_sampling_roll20_mean",
    "hr_sampling_roll20_std"
]


# ============================================================
# TEMPO
# ============================================================
#
# Usaremos tempo em minutos.
#
# Não precisamos colocar também a versão em horas,
# pois ela representa exatamente a mesma informação
# dividida por 60.
#
# ============================================================

FEATURES_TEMPO = [
    "tempo_desde_inicio_min",
    "hora_local_sin",
    "hora_local_cos"
]


# ============================================================
# TODAS AS FEATURES
# ============================================================

FEATURES = (
    FEATURES_ORIGINAIS
    + FEATURES_2_5_MIN
    + FEATURES_5_MIN
    + FEATURES_10_MIN
    + FEATURES_FREQ
    + FEATURES_TEMPO
)


# ============================================================
# INÍCIO
# ============================================================

print("=" * 80)
print("COMPARAÇÃO DE MODELOS DE MACHINE LEARNING")
print("=" * 80)


print(
    f"\nFeatures utilizadas: {len(FEATURES)}"
)


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

print(
    "\nCarregando dataset..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


print(
    f"Épocas encontradas: {len(dataset)}"
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


# ============================================================
# 2. CARREGAR DIVISÃO
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


participantes_teste = (
    divisao[
        divisao["grupo"] == "teste"
    ]["participante"]
    .tolist()
)


print(
    "\nParticipantes reservados para teste:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# 3. CRIAR BASE DE DESENVOLVIMENTO
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
    f"\nÉpocas de desenvolvimento: "
    f"{len(dataset_dev)}"
)


print(
    f"Épocas de teste reservadas: "
    f"{len(dataset_teste)}"
)


if len(dataset_teste) != 12061:

    raise ValueError(
        "O conjunto reservado de teste "
        "não possui 12061 épocas."
    )


# ============================================================
# 4. VALIDAR FEATURES
# ============================================================

faltando = [
    feature
    for feature in FEATURES
    if feature not in dataset_dev.columns
]


if faltando:

    raise ValueError(
        "Features ausentes:\n"
        + "\n".join(
            faltando
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
        f"Existem {ausentes} valores ausentes."
    )


print(
    "✅ Dados validados."
)


# ============================================================
# 5. X, Y E GRUPOS
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
# 6. CRIAR OS 6 FOLDS
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=42
)


folds = list(
    cv.split(
        X,
        y,
        groups=grupos
    )
)


# ============================================================
# 7. FÁBRICA DOS MODELOS
# ============================================================
#
# Criamos uma função porque precisamos de
# um modelo NOVO para cada fold.
#
# ============================================================

def criar_modelo(nome):

    if nome == "RandomForest":

        return RandomForestClassifier(

            n_estimators=300,

            # Melhor configuração encontrada
            # na nossa busca anterior.
            max_depth=16,

            min_samples_split=10,

            min_samples_leaf=5,

            max_features="sqrt",

            class_weight="balanced_subsample",

            random_state=42,

            n_jobs=-1
        )


    if nome == "ExtraTrees":

        return ExtraTreesClassifier(

            n_estimators=400,

            max_depth=16,

            min_samples_split=10,

            min_samples_leaf=5,

            max_features="sqrt",

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        )


    if nome == "HistGradientBoosting":

        return HistGradientBoostingClassifier(

            learning_rate=0.08,

            max_iter=250,

            max_leaf_nodes=31,

            min_samples_leaf=20,

            l2_regularization=1.0,

            class_weight="balanced",

            random_state=42
        )


    raise ValueError(
        f"Modelo desconhecido: {nome}"
    )


# ============================================================
# 8. MODELOS QUE VÃO COMPETIR
# ============================================================

NOMES_MODELOS = [
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting"
]


resultados = []


# ============================================================
# 9. TESTAR CADA MODELO
# ============================================================

for numero_modelo, nome_modelo in enumerate(
    NOMES_MODELOS,
    start=1
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"MODELO {numero_modelo}/"
        f"{len(NOMES_MODELOS)}"
    )

    print(
        nome_modelo
    )

    print(
        "=" * 80
    )


    for fold, (
        indices_treino,
        indices_validacao
    ) in enumerate(
        folds,
        start=1
    ):


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


        participantes_validacao = sorted(

            grupos.iloc[
                indices_validacao
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
        # CRIAR MODELO
        # ====================================================

        modelo = criar_modelo(
            nome_modelo
        )


        # ====================================================
        # TREINAMENTO
        # ====================================================

        inicio = time.perf_counter()


        modelo.fit(
            X_treino,
            y_treino
        )


        tempo_treino = (
            time.perf_counter()
            - inicio
        )


        # ====================================================
        # PREDIÇÃO
        # ====================================================

        pred = modelo.predict(
            X_validacao
        )


        # ====================================================
        # MÉTRICAS
        # ====================================================

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


        print(
            f"Tempo treino: "
            f"{tempo_treino:.1f}s"
        )


        resultados.append({

            "modelo":
                nome_modelo,

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
                weighted_f1,

            "tempo_treino_seg":
                tempo_treino
        })


# ============================================================
# 10. DATAFRAME
# ============================================================

df = pd.DataFrame(
    resultados
)


# ============================================================
# 11. RESUMO DOS MODELOS
# ============================================================

resumo = (

    df

    .groupby(
        "modelo"
    )

    .agg(

        accuracy_media=(
            "accuracy",
            "mean"
        ),

        accuracy_desvio=(
            "accuracy",
            "std"
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
        ),

        tempo_medio_seg=(
            "tempo_treino_seg",
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
# 12. RANKING
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RANKING DOS MODELOS"
)

print(
    "=" * 80
)


print(
    "\n"
    + resumo[
        [
            "modelo",
            "accuracy_media",
            "balanced_accuracy_media",
            "macro_f1_media",
            "macro_f1_desvio",
            "weighted_f1_media",
            "tempo_medio_seg"
        ]
    ]

    .round(4)

    .to_string(
        index=False
    )
)


# ============================================================
# 13. MELHOR MODELO
# ============================================================

melhor = resumo.iloc[0]


print(
    "\n"
    + "=" * 80
)

print(
    "MELHOR MODELO"
)

print(
    "=" * 80
)


print(
    f"\nModelo: "
    f"{melhor['modelo']}"
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


print(
    f"Desvio Macro F1: "
    f"{melhor['macro_f1_desvio']:.4f}"
)


# ============================================================
# 14. SALVAR
# ============================================================

ARQUIVO_COMPLETO = (
    PASTA_RESULTADOS
    / "comparacao_modelos_ml_folds.csv"
)


ARQUIVO_RANKING = (
    PASTA_RESULTADOS
    / "comparacao_modelos_ml_ranking.csv"
)


df.to_csv(
    ARQUIVO_COMPLETO,
    index=False
)


resumo.to_csv(
    ARQUIVO_RANKING,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "COMPARAÇÃO FINALIZADA"
)

print(
    "=" * 80
)


print(
    "\nArquivos:"
)


print(
    ARQUIVO_COMPLETO
)


print(
    ARQUIVO_RANKING
)


print(
    "\n⚠️ O conjunto de teste "
    "continua completamente intocado."
)


print(
    "\n✅ COMPARAÇÃO CONCLUÍDA"
)