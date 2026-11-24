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
# 1. FEATURES BASE
# ============================================================
#
# Estas são as 27 features que já tínhamos antes
# de adicionar frequência de amostragem e tempo.
#
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
# 2. FEATURES DE FREQUÊNCIA DE AMOSTRAGEM
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
# 3. FEATURES DE TEMPO
# ============================================================
#
# Não usamos tempo_desde_inicio_horas porque ele é
# exatamente a mesma informação de minutos / 60.
#
# ============================================================

FEATURES_TEMPO = [

    "tempo_desde_inicio_min",

    "hora_local_sin",
    "hora_local_cos"
]


# ============================================================
# 4. GRUPOS QUE SERÃO COMPARADOS
# ============================================================

EXPERIMENTOS = {

    "A_BASE_27":
        FEATURES_BASE,

    "B_BASE_MAIS_FREQ":
        FEATURES_BASE
        + FEATURES_FREQ,

    "C_BASE_MAIS_TEMPO":
        FEATURES_BASE
        + FEATURES_TEMPO,

    "D_BASE_FREQ_TEMPO":
        FEATURES_BASE
        + FEATURES_FREQ
        + FEATURES_TEMPO
}


# ============================================================
# INÍCIO
# ============================================================

print("=" * 85)
print("ESTUDO DE ABLAÇÃO DAS FEATURES")
print("=" * 85)


print(
    "\nExperimentos:"
)


for nome, features in EXPERIMENTOS.items():

    print(
        f"  {nome}: "
        f"{len(features)} features"
    )


print(
    "\nCada experimento será avaliado "
    "com os mesmos 6 folds por participante."
)


# ============================================================
# 5. CARREGAR DATASET
# ============================================================

print(
    "\nCarregando dataset..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


print(
    f"Épocas encontradas: "
    f"{len(dataset)}"
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


# ============================================================
# 6. CARREGAR DIVISÃO
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
    "\nParticipantes reservados para teste final:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# 7. RETIRAR TESTE
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
        "O conjunto de teste deveria possuir "
        "12061 épocas."
    )


# ============================================================
# 8. VALIDAR TODAS AS FEATURES
# ============================================================

todas_features = list(
    dict.fromkeys(
        FEATURES_BASE
        + FEATURES_FREQ
        + FEATURES_TEMPO
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
        + "\n".join(
            faltando
        )
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
        f"Foram encontrados {ausentes} "
        f"valores ausentes."
    )


print(
    "✅ Dados validados."
)


# ============================================================
# 9. LABEL E PARTICIPANTES
# ============================================================

y = dataset_dev[
    "label"
]


grupos = dataset_dev[
    "participante"
]


# ============================================================
# 10. CRIAR OS MESMOS FOLDS PARA TODOS
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=42
)


# Para criar os índices, qualquer conjunto de features serve.
X_referencia = dataset_dev[
    FEATURES_BASE
]


folds = list(
    cv.split(
        X_referencia,
        y,
        groups=grupos
    )
)


# ============================================================
# 11. RESULTADOS
# ============================================================

resultados = []


# ============================================================
# 12. EXECUTAR CADA EXPERIMENTO
# ============================================================

for numero, (
    nome_experimento,
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
        f"EXPERIMENTO {numero}/"
        f"{len(EXPERIMENTOS)}"
    )

    print(
        nome_experimento
    )

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
            ].unique()
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
        # RANDOM FOREST
        # ====================================================
        #
        # Usamos a melhor configuração encontrada
        # na busca de hiperparâmetros.
        #
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


        # ====================================================
        # TREINAR
        # ====================================================

        modelo.fit(
            X_treino,
            y_treino
        )


        # ====================================================
        # PREVER
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


        resultados.append({

            "experimento":
                nome_experimento,

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
# 13. DATAFRAME DE RESULTADOS
# ============================================================

df = pd.DataFrame(
    resultados
)


# ============================================================
# 14. CRIAR RESUMO
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

        accuracy_desvio=(
            "accuracy",
            "std"
        ),

        balanced_accuracy_media=(
            "balanced_accuracy",
            "mean"
        ),

        balanced_accuracy_desvio=(
            "balanced_accuracy",
            "std"
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
# 15. RESULTADO FINAL
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "RESULTADO DO ESTUDO DE ABLAÇÃO"
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
# 16. CALCULAR GANHO EM RELAÇÃO À BASE
# ============================================================

linha_base = resumo[
    resumo[
        "experimento"
    ] == "A_BASE_27"
].iloc[0]


macro_base = linha_base[
    "macro_f1_media"
]


accuracy_base = linha_base[
    "accuracy_media"
]


print(
    "\n"
    + "=" * 85
)

print(
    "GANHO EM RELAÇÃO À BASE DE 27 FEATURES"
)

print(
    "=" * 85
)


for _, linha in resumo.iterrows():

    ganho_macro = (
        linha[
            "macro_f1_media"
        ]
        - macro_base
    )


    ganho_accuracy = (
        linha[
            "accuracy_media"
        ]
        - accuracy_base
    )


    print(
        f"\n{linha['experimento']}"
    )


    print(
        f"  Ganho Macro F1: "
        f"{ganho_macro:+.4f}"
    )


    print(
        f"  Ganho Accuracy: "
        f"{ganho_accuracy:+.4f}"
    )


# ============================================================
# 17. MELHOR CONJUNTO
# ============================================================

melhor = resumo.iloc[0]


print(
    "\n"
    + "=" * 85
)

print(
    "MELHOR CONJUNTO DE FEATURES"
)

print(
    "=" * 85
)


print(
    f"\nExperimento: "
    f"{melhor['experimento']}"
)


print(
    f"Quantidade de features: "
    f"{int(melhor['quantidade_features'])}"
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
# 18. SALVAR
# ============================================================

ARQUIVO_FOLDS = (
    PASTA_RESULTADOS
    / "ablacao_features_folds.csv"
)


ARQUIVO_RESUMO = (
    PASTA_RESULTADOS
    / "ablacao_features_resumo.csv"
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
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "ESTUDO DE ABLAÇÃO FINALIZADO"
)

print(
    "=" * 85
)


print(
    "\nArquivos criados:"
)


print(
    ARQUIVO_FOLDS
)


print(
    ARQUIVO_RESUMO
)


print(
    "\n⚠️ Os 3 participantes de teste "
    "continuam sem ser utilizados."
)


print(
    "\n✅ EXPERIMENTO CONCLUÍDO"
)