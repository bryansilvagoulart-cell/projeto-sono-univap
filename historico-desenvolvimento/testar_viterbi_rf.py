from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support
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
# FEATURES DEFINITIVAS DESTA ETAPA
# ============================================================
#
# 27 features anteriores
# +
# tempo decorrido
# +
# horário local seno/cosseno
#
# Total = 30
#
# Não utilizaremos as features de frequência de amostragem,
# pois o estudo de ablação mostrou que elas praticamente
# não acrescentaram desempenho.
#
# ============================================================

FEATURES = [

    # --------------------------------------------------------
    # ÉPOCA ATUAL
    # --------------------------------------------------------

    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",

    "motion_mean",
    "motion_std",
    "motion_max",


    # --------------------------------------------------------
    # CONTEXTO 2,5 MIN
    # --------------------------------------------------------

    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean",


    # --------------------------------------------------------
    # CONTEXTO 5 MIN
    # --------------------------------------------------------

    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",

    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean",


    # --------------------------------------------------------
    # CONTEXTO 10 MIN
    # --------------------------------------------------------

    "hr_mean_roll20_mean",
    "hr_mean_roll20_std",
    "hr_std_roll20_mean",

    "motion_mean_roll20_mean",
    "motion_mean_roll20_std",
    "motion_max_roll20_mean",


    # --------------------------------------------------------
    # TEMPO
    # --------------------------------------------------------

    "tempo_desde_inicio_min",
    "hora_local_sin",
    "hora_local_cos"
]


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    0,
    1,
    2,
    3,
    4
]


NOMES_CLASSES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}


# ============================================================
# PESOS DO VITERBI
# ============================================================
#
# 0.00:
# ignora completamente a matriz de transição.
# Deve produzir praticamente o mesmo resultado do RF.
#
# Valores maiores:
# dão cada vez mais importância à sequência.
#
# ============================================================

PESOS_TRANSICAO = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00
]


# ============================================================
# FUNÇÃO: MATRIZ DE TRANSIÇÃO
# ============================================================

def calcular_transicoes(dataset_treino):

    # --------------------------------------------------------
    # Começamos com 1 em todas as posições.
    #
    # Isso é uma suavização de Laplace.
    # Evita probabilidades exatamente iguais a zero.
    # --------------------------------------------------------

    contagens = np.ones(
        (
            len(CLASSES),
            len(CLASSES)
        ),
        dtype=float
    )


    dataset_ordenado = (
        dataset_treino
        .sort_values(
            [
                "id_noite",
                "segmento",
                "epoca"
            ]
        )
    )


    grupos = dataset_ordenado.groupby(
        [
            "id_noite",
            "segmento"
        ],
        sort=False
    )


    for _, grupo in grupos:

        labels = grupo[
            "label"
        ].to_numpy()


        epocas = grupo[
            "epoca"
        ].to_numpy()


        if len(labels) < 2:

            continue


        for i in range(
            1,
            len(labels)
        ):

            # ------------------------------------------------
            # Só contamos a transição se as épocas forem
            # realmente consecutivas.
            # ------------------------------------------------

            if (
                epocas[i]
                - epocas[i - 1]
                != 1
            ):

                continue


            anterior = int(
                labels[i - 1]
            )


            atual = int(
                labels[i]
            )


            contagens[
                anterior,
                atual
            ] += 1


    # --------------------------------------------------------
    # Transformar contagens em probabilidades
    # --------------------------------------------------------

    matriz = (
        contagens
        / contagens.sum(
            axis=1,
            keepdims=True
        )
    )


    return matriz


# ============================================================
# FUNÇÃO: PROBABILIDADE INICIAL DOS ESTÁGIOS
# ============================================================

def calcular_probabilidade_inicial(
    dataset_treino
):

    contagens = np.ones(
        len(CLASSES),
        dtype=float
    )


    valores = (
        dataset_treino[
            "label"
        ]
        .value_counts()
    )


    for classe in CLASSES:

        if classe in valores.index:

            contagens[
                classe
            ] += valores[
                classe
            ]


    probabilidades = (
        contagens
        / contagens.sum()
    )


    return probabilidades


# ============================================================
# FUNÇÃO VITERBI
# ============================================================

def viterbi(
    probabilidades_emissao,
    matriz_transicao,
    probabilidade_inicial,
    peso_transicao
):

    quantidade_epocas = (
        probabilidades_emissao.shape[0]
    )


    quantidade_classes = (
        probabilidades_emissao.shape[1]
    )


    # --------------------------------------------------------
    # Evitar log(0)
    # --------------------------------------------------------

    epsilon = 1e-12


    log_emissao = np.log(
        probabilidades_emissao
        + epsilon
    )


    log_transicao = np.log(
        matriz_transicao
        + epsilon
    )


    log_inicial = np.log(
        probabilidade_inicial
        + epsilon
    )


    # --------------------------------------------------------
    # Tabelas do algoritmo
    # --------------------------------------------------------

    pontuacao = np.zeros(
        (
            quantidade_epocas,
            quantidade_classes
        )
    )


    caminho = np.zeros(
        (
            quantidade_epocas,
            quantidade_classes
        ),
        dtype=int
    )


    # --------------------------------------------------------
    # PRIMEIRA ÉPOCA
    # --------------------------------------------------------

    pontuacao[
        0
    ] = (

        log_emissao[
            0
        ]

        +

        peso_transicao
        * log_inicial
    )


    # --------------------------------------------------------
    # DEMAIS ÉPOCAS
    # --------------------------------------------------------

    for t in range(
        1,
        quantidade_epocas
    ):


        for classe_atual in range(
            quantidade_classes
        ):


            candidatos = (

                pontuacao[
                    t - 1
                ]

                +

                peso_transicao
                * log_transicao[
                    :,
                    classe_atual
                ]
            )


            melhor_anterior = int(
                np.argmax(
                    candidatos
                )
            )


            pontuacao[
                t,
                classe_atual
            ] = (

                candidatos[
                    melhor_anterior
                ]

                +

                log_emissao[
                    t,
                    classe_atual
                ]
            )


            caminho[
                t,
                classe_atual
            ] = melhor_anterior


    # --------------------------------------------------------
    # RECUPERAR MELHOR CAMINHO
    # --------------------------------------------------------

    resultado = np.zeros(
        quantidade_epocas,
        dtype=int
    )


    resultado[
        -1
    ] = int(
        np.argmax(
            pontuacao[
                -1
            ]
        )
    )


    for t in range(
        quantidade_epocas - 2,
        -1,
        -1
    ):

        resultado[
            t
        ] = caminho[
            t + 1,
            resultado[
                t + 1
            ]
        ]


    return resultado


# ============================================================
# FUNÇÃO: APLICAR VITERBI EM TODAS AS SEQUÊNCIAS
# ============================================================

def aplicar_viterbi_dataset(
    metadata,
    probabilidades,
    matriz_transicao,
    probabilidade_inicial,
    peso_transicao
):

    # --------------------------------------------------------
    # Array onde colocaremos as predições.
    #
    # A posição será a mesma posição original do conjunto
    # de validação.
    # --------------------------------------------------------

    predicoes = np.zeros(
        len(metadata),
        dtype=int
    )


    metadata = metadata.copy()


    metadata[
        "_posicao"
    ] = np.arange(
        len(metadata)
    )


    grupos = metadata.groupby(
        [
            "id_noite",
            "segmento"
        ],
        sort=False
    )


    for _, grupo in grupos:


        grupo = grupo.sort_values(
            "epoca"
        )


        posicoes = grupo[
            "_posicao"
        ].to_numpy()


        probabilidades_grupo = (
            probabilidades[
                posicoes
            ]
        )


        predicao_grupo = viterbi(

            probabilidades_grupo,

            matriz_transicao,

            probabilidade_inicial,

            peso_transicao
        )


        predicoes[
            posicoes
        ] = predicao_grupo


    return predicoes


# ============================================================
# INÍCIO
# ============================================================

print("=" * 90)
print("RANDOM FOREST + MODELAGEM SEQUENCIAL VITERBI")
print("=" * 90)


print(
    f"\nQuantidade de features: "
    f"{len(FEATURES)}"
)


print(
    "\nPesos de transição:"
)


for peso in PESOS_TRANSICAO:

    print(
        f"  {peso:.2f}"
    )


# ============================================================
# CARREGAR DATASET
# ============================================================

print(
    "\nCarregando dataset..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


print(
    f"Épocas: {len(dataset)}"
)


# ============================================================
# CARREGAR DIVISÃO
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


participantes_teste = (

    divisao[
        divisao[
            "grupo"
        ] == "teste"
    ]

    ["participante"]

    .tolist()
)


print(
    "\nTeste final continua fechado:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# REMOVER TESTE
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
        "no conjunto de teste."
    )


# ============================================================
# VALIDAÇÃO
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


if dataset_dev[
    FEATURES
].isna().sum().sum() != 0:

    raise ValueError(
        "Existem valores ausentes "
        "nas features."
    )


print(
    "✅ Dados validados."
)


# ============================================================
# X, Y E GRUPOS
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
# CRIAR 6 FOLDS
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
# RESULTADOS
# ============================================================

resultados = []

resultados_classes = []


# ============================================================
# EXECUTAR FOLDS
# ============================================================

for fold, (
    indices_treino,
    indices_validacao
) in enumerate(
    folds,
    start=1
):


    print(
        "\n"
        + "=" * 90
    )


    print(
        f"FOLD {fold}/6"
    )


    print(
        "=" * 90
    )


    # --------------------------------------------------------
    # DADOS
    # --------------------------------------------------------

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


    dataset_treino_fold = (
        dataset_dev.iloc[
            indices_treino
        ]
        .copy()
    )


    metadata_validacao = (
        dataset_dev.iloc[
            indices_validacao
        ][
            [
                "id_noite",
                "segmento",
                "epoca",
                "participante",
                "label"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    participantes_validacao = sorted(

        metadata_validacao[
            "participante"
        ]

        .unique()
    )


    print(
        "Participantes validação: "
        + ", ".join(
            participantes_validacao
        )
    )


    # ========================================================
    # APRENDER AS TRANSIÇÕES SOMENTE COM O TREINO
    # ========================================================

    matriz_transicao = (
        calcular_transicoes(
            dataset_treino_fold
        )
    )


    probabilidade_inicial = (
        calcular_probabilidade_inicial(
            dataset_treino_fold
        )
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

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


    print(
        "\nTreinando Random Forest..."
    )


    modelo.fit(
        X_treino,
        y_treino
    )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    probabilidades_originais = (
        modelo.predict_proba(
            X_validacao
        )
    )


    # --------------------------------------------------------
    # Garantir ordem das classes 0,1,2,3,4
    # --------------------------------------------------------

    probabilidades = np.zeros(
        (
            len(X_validacao),
            len(CLASSES)
        )
    )


    for indice_modelo, classe in enumerate(
        modelo.classes_
    ):

        probabilidades[
            :,
            int(classe)
        ] = (
            probabilidades_originais[
                :,
                indice_modelo
            ]
        )


    # ========================================================
    # TESTAR PESOS DE TRANSIÇÃO
    # ========================================================

    for peso in PESOS_TRANSICAO:


        pred = aplicar_viterbi_dataset(

            metadata_validacao,

            probabilidades,

            matriz_transicao,

            probabilidade_inicial,

            peso
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
            f"\nPeso {peso:.2f}"
        )


        print(
            f"  Accuracy: "
            f"{accuracy:.4f}"
        )


        print(
            f"  Balanced Accuracy: "
            f"{balanced:.4f}"
        )


        print(
            f"  Macro F1: "
            f"{macro_f1:.4f}"
        )


        resultados.append({

            "fold":
                fold,

            "participantes_validacao":
                ", ".join(
                    participantes_validacao
                ),

            "peso_transicao":
                peso,

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced,

            "macro_f1":
                macro_f1,

            "weighted_f1":
                weighted_f1
        })


        # ====================================================
        # MÉTRICAS POR CLASSE
        # ====================================================

        precisao, recall, f1_classes, suporte = (
            precision_recall_fscore_support(

                y_validacao,

                pred,

                labels=CLASSES,

                zero_division=0
            )
        )


        for i, classe in enumerate(
            CLASSES
        ):

            resultados_classes.append({

                "fold":
                    fold,

                "peso_transicao":
                    peso,

                "classe":
                    classe,

                "nome_classe":
                    NOMES_CLASSES[
                        classe
                    ],

                "precision":
                    precisao[
                        i
                    ],

                "recall":
                    recall[
                        i
                    ],

                "f1":
                    f1_classes[
                        i
                    ],

                "support":
                    suporte[
                        i
                    ]
            })


# ============================================================
# DATAFRAMES
# ============================================================

df = pd.DataFrame(
    resultados
)


df_classes = pd.DataFrame(
    resultados_classes
)


# ============================================================
# RANKING
# ============================================================

ranking = (

    df

    .groupby(
        "peso_transicao"
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
# MOSTRAR RANKING
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "RANKING VITERBI"
)

print(
    "=" * 90
)


print(

    "\n"

    + ranking.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# MELHOR PESO
# ============================================================

melhor = ranking.iloc[
    0
]


melhor_peso = float(
    melhor[
        "peso_transicao"
    ]
)


print(
    "\n"
    + "=" * 90
)

print(
    "MELHOR CONFIGURAÇÃO"
)

print(
    "=" * 90
)


print(
    f"\nPeso de transição: "
    f"{melhor_peso:.2f}"
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
# RESULTADO POR CLASSE DO MELHOR PESO
# ============================================================

classes_melhor = (

    df_classes[
        df_classes[
            "peso_transicao"
        ] == melhor_peso
    ]

    .groupby(
        [
            "classe",
            "nome_classe"
        ]
    )

    .agg(

        precision_media=(
            "precision",
            "mean"
        ),

        recall_medio=(
            "recall",
            "mean"
        ),

        f1_medio=(
            "f1",
            "mean"
        )
    )

    .reset_index()
)


print(
    "\n"
    + "=" * 90
)

print(
    "DESEMPENHO POR CLASSE"
)

print(
    "=" * 90
)


print(

    "\n"

    + classes_melhor.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# SALVAR
# ============================================================

df.to_csv(

    PASTA_RESULTADOS
    / "viterbi_rf_folds.csv",

    index=False
)


ranking.to_csv(

    PASTA_RESULTADOS
    / "viterbi_rf_ranking.csv",

    index=False
)


df_classes.to_csv(

    PASTA_RESULTADOS
    / "viterbi_rf_classes.csv",

    index=False
)


classes_melhor.to_csv(

    PASTA_RESULTADOS
    / "viterbi_rf_melhor_classes.csv",

    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "EXPERIMENTO VITERBI FINALIZADO"
)

print(
    "=" * 90
)


print(
    "\n⚠️ O conjunto de teste final "
    "continua completamente intocado."
)


print(
    "\n✅ CONCLUÍDO"
)