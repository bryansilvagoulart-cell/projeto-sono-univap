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
# 27 FEATURES DO NOSSO MODELO MULTIESCALA
# ============================================================

FEATURES = [

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
# CONFIGURAÇÕES QUE SERÃO TESTADAS
# ============================================================
#
# A configuração 1 é praticamente o nosso modelo atual.
#
# As outras alteram profundidade e quantidade mínima
# de exemplos necessária para criar regras.
#
# O balanceamento será mantido em todas elas.
#
# ============================================================

CONFIGURACOES = [

    {
        "nome": "RF_01_ATUAL",
        "max_depth": 12,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt"
    },

    {
        "nome": "RF_02_DEPTH10",
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt"
    },

    {
        "nome": "RF_03_DEPTH14",
        "max_depth": 14,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt"
    },

    {
        "nome": "RF_04_DEPTH16",
        "max_depth": 16,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt"
    },

    {
        "nome": "RF_05_FOLHA5",
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    },

    {
        "nome": "RF_06_FOLHA20",
        "max_depth": 12,
        "min_samples_split": 40,
        "min_samples_leaf": 20,
        "max_features": "sqrt"
    },

    {
        "nome": "RF_07_DEPTH16_FOLHA5",
        "max_depth": 16,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    },

    {
        "nome": "RF_08_FEATURES_70",
        "max_depth": 12,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": 0.7
    }
]


# ============================================================
# INÍCIO
# ============================================================

print("=" * 80)
print("BUSCA DE HIPERPARÂMETROS")
print("RANDOM FOREST MULTIESCALA")
print("=" * 80)


print(
    f"\nConfigurações que serão testadas: "
    f"{len(CONFIGURACOES)}"
)

print(
    "Folds por configuração: 6"
)

print(
    f"Total de treinamentos: "
    f"{len(CONFIGURACOES) * 6}"
)

print(
    "\n⚠️ Isso pode demorar."
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
    f"Épocas totais: {len(dataset)}"
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
    "\nParticipantes que continuarão reservados:"
)

for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# 3. REMOVER TESTE DA BUSCA
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
        "A quantidade do conjunto de teste "
        "não corresponde às 12061 épocas esperadas."
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
        + "\n".join(faltando)
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
# 6. CRIAR OS MESMOS 6 FOLDS PARA TODAS AS CONFIGURAÇÕES
# ============================================================
#
# Isso é importante.
#
# Todas as configurações serão testadas exatamente
# nas mesmas divisões.
#
# Assim a comparação fica justa.
#
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
# 7. RESULTADOS
# ============================================================

resultados_configuracoes = []

resultados_folds = []


# ============================================================
# 8. TESTAR CADA CONFIGURAÇÃO
# ============================================================

for numero_config, config in enumerate(
    CONFIGURACOES,
    start=1
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"CONFIGURAÇÃO {numero_config}/"
        f"{len(CONFIGURACOES)}"
    )

    print(
        config["nome"]
    )

    print(
        "=" * 80
    )


    print(
        f"\nmax_depth = "
        f"{config['max_depth']}"
    )

    print(
        f"min_samples_split = "
        f"{config['min_samples_split']}"
    )

    print(
        f"min_samples_leaf = "
        f"{config['min_samples_leaf']}"
    )

    print(
        f"max_features = "
        f"{config['max_features']}"
    )


    metricas_config = []


    # ========================================================
    # FOLDS DA CONFIGURAÇÃO
    # ========================================================

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
            f"\nFold {fold}/6 "
            f"-> "
            f"{', '.join(participantes_validacao)}"
        )


        # ====================================================
        # MODELO
        # ====================================================

        modelo = RandomForestClassifier(

            # Mantemos 300 árvores
            # igual aos experimentos anteriores.
            n_estimators=300,

            max_depth=
                config["max_depth"],

            min_samples_split=
                config["min_samples_split"],

            min_samples_leaf=
                config["min_samples_leaf"],

            max_features=
                config["max_features"],

            class_weight=
                "balanced_subsample",

            random_state=42,

            n_jobs=-1
        )


        # ====================================================
        # TREINO
        # ====================================================

        modelo.fit(
            X_treino,
            y_treino
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


        balanced_accuracy = (
            balanced_accuracy_score(
                y_validacao,
                pred
            )
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
            f"  Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"  Balanced Accuracy: "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"  Macro F1: "
            f"{macro_f1:.4f}"
        )


        resultado_fold = {

            "configuracao":
                config["nome"],

            "fold":
                fold,

            "participantes_validacao":
                ", ".join(
                    participantes_validacao
                ),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_f1":
                macro_f1,

            "weighted_f1":
                weighted_f1
        }


        metricas_config.append(
            resultado_fold
        )


        resultados_folds.append(
            resultado_fold
        )


    # ========================================================
    # MÉDIA DA CONFIGURAÇÃO
    # ========================================================

    df_config = pd.DataFrame(
        metricas_config
    )


    accuracy_media = (
        df_config[
            "accuracy"
        ].mean()
    )


    balanced_media = (
        df_config[
            "balanced_accuracy"
        ].mean()
    )


    macro_media = (
        df_config[
            "macro_f1"
        ].mean()
    )


    macro_desvio = (
        df_config[
            "macro_f1"
        ].std()
    )


    weighted_media = (
        df_config[
            "weighted_f1"
        ].mean()
    )


    print(
        "\nResultado médio:"
    )

    print(
        f"  Accuracy = "
        f"{accuracy_media:.4f}"
    )

    print(
        f"  Balanced Accuracy = "
        f"{balanced_media:.4f}"
    )

    print(
        f"  Macro F1 = "
        f"{macro_media:.4f}"
    )

    print(
        f"  Desvio Macro F1 = "
        f"{macro_desvio:.4f}"
    )


    resultados_configuracoes.append({

        "configuracao":
            config["nome"],

        "max_depth":
            config["max_depth"],

        "min_samples_split":
            config["min_samples_split"],

        "min_samples_leaf":
            config["min_samples_leaf"],

        "max_features":
            config["max_features"],

        "accuracy_media":
            accuracy_media,

        "balanced_accuracy_media":
            balanced_media,

        "macro_f1_media":
            macro_media,

        "macro_f1_desvio":
            macro_desvio,

        "weighted_f1_media":
            weighted_media
    })


# ============================================================
# 9. CRIAR TABELA FINAL
# ============================================================

df_resultados = pd.DataFrame(
    resultados_configuracoes
)


df_resultados = (
    df_resultados

    .sort_values(
        by="macro_f1_media",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 10. RANKING
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RANKING FINAL"
)

print(
    "=" * 80
)


print(
    "\n"
    + df_resultados[
        [
            "configuracao",
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
# 11. MELHOR CONFIGURAÇÃO
# ============================================================

melhor = df_resultados.iloc[0]


print(
    "\n"
    + "=" * 80
)

print(
    "MELHOR CONFIGURAÇÃO"
)

print(
    "=" * 80
)


print(
    f"\nNome: "
    f"{melhor['configuracao']}"
)


print(
    f"max_depth: "
    f"{melhor['max_depth']}"
)


print(
    f"min_samples_split: "
    f"{melhor['min_samples_split']}"
)


print(
    f"min_samples_leaf: "
    f"{melhor['min_samples_leaf']}"
)


print(
    f"max_features: "
    f"{melhor['max_features']}"
)


print(
    f"\nMacro F1 médio: "
    f"{melhor['macro_f1_media']:.4f}"
)


print(
    f"Balanced Accuracy média: "
    f"{melhor['balanced_accuracy_media']:.4f}"
)


print(
    f"Accuracy média: "
    f"{melhor['accuracy_media']:.4f}"
)


# ============================================================
# 12. SALVAR RESULTADOS
# ============================================================

ARQUIVO_RANKING = (
    PASTA_RESULTADOS
    / "busca_hiperparametros_rf_ranking.csv"
)


ARQUIVO_FOLDS = (
    PASTA_RESULTADOS
    / "busca_hiperparametros_rf_folds.csv"
)


df_resultados.to_csv(
    ARQUIVO_RANKING,
    index=False
)


pd.DataFrame(
    resultados_folds
).to_csv(
    ARQUIVO_FOLDS,
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
    "BUSCA FINALIZADA"
)

print(
    "=" * 80
)


print(
    "\nArquivos criados:"
)


print(
    ARQUIVO_RANKING
)


print(
    ARQUIVO_FOLDS
)


print(
    "\n⚠️ Os 3 participantes de teste "
    "continuam completamente intocados."
)


print(
    "\n✅ BUSCA DE HIPERPARÂMETROS CONCLUÍDA"
)