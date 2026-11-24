from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 256
EPOCAS_TREINO = 8
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001

RAIO_JANELA = 10
TAMANHO_JANELA = 21


# Peso da BiGRU no ensemble.
#
# 0.00 = somente Random Forest
# 1.00 = somente BiGRU
#
PESOS_BIGRU = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


# Intensidade das transições do Viterbi.
#
PESOS_VITERBI = [
    0.00,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60
]


# ============================================================
# REFERÊNCIAS ATUAIS
# ============================================================

RF_VITERBI_MACRO = 0.5117
RF_VITERBI_ACCURACY = 0.5632

BIGRU_MACRO = 0.5174
BIGRU_ACCURACY = 0.5456


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
# FEATURES
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


    # 2,5 MIN
    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean",


    # 5 MIN
    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",

    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean",


    # 10 MIN
    "hr_mean_roll20_mean",
    "hr_mean_roll20_std",
    "hr_std_roll20_mean",

    "motion_mean_roll20_mean",
    "motion_mean_roll20_std",
    "motion_max_roll20_mean",


    # TEMPO
    "tempo_desde_inicio_min",
    "hora_local_sin",
    "hora_local_cos"
]


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
# SEED
# ============================================================

def definir_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# JANELAS TEMPORAIS
# ============================================================

def construir_indices_janelas(
    dataset,
    raio
):

    tamanho = (
        raio * 2
        + 1
    )

    indices = np.full(
        (
            len(dataset),
            tamanho
        ),
        -1,
        dtype=np.int64
    )


    auxiliar = (
        dataset
        .reset_index()
        .sort_values(
            [
                "id_noite",
                "segmento",
                "epoca"
            ]
        )
    )


    grupos = auxiliar.groupby(
        [
            "id_noite",
            "segmento"
        ],
        sort=False
    )


    for _, grupo in grupos:


        posicoes = grupo[
            "index"
        ].to_numpy()


        quantidade = len(
            posicoes
        )


        for pos_local in range(
            quantidade
        ):


            inicio = max(
                0,
                pos_local - raio
            )


            fim = min(
                quantidade,
                pos_local + raio + 1
            )


            esquerda = (
                pos_local - inicio
            )


            destino_inicio = (
                raio - esquerda
            )


            destino_fim = (
                destino_inicio
                + (fim - inicio)
            )


            linha_original = posicoes[
                pos_local
            ]


            indices[
                linha_original,
                destino_inicio:destino_fim
            ] = posicoes[
                inicio:fim
            ]


    return indices


# ============================================================
# DATASET PYTORCH
# ============================================================

class DatasetSequencial(Dataset):


    def __init__(
        self,
        indices_linhas,
        X_escalado,
        labels,
        indices_janelas
    ):


        self.indices_linhas = np.asarray(
            indices_linhas,
            dtype=np.int64
        )


        self.X = X_escalado

        self.labels = labels

        self.indices_janelas = (
            indices_janelas
        )


    def __len__(self):

        return len(
            self.indices_linhas
        )


    def __getitem__(
        self,
        indice
    ):


        linha = self.indices_linhas[
            indice
        ]


        janela = self.indices_janelas[
            linha
        ]


        validos = (
            janela >= 0
        )


        seguros = janela.copy()

        seguros[
            ~validos
        ] = 0


        sequencia = (
            self.X[
                seguros
            ]
            .copy()
        )


        sequencia[
            ~validos
        ] = 0.0


        mascara = (
            validos
            .astype(
                np.float32
            )
            .reshape(
                -1,
                1
            )
        )


        sequencia = np.concatenate(
            [
                sequencia,
                mascara
            ],
            axis=1
        )


        label = int(
            self.labels[
                linha
            ]
        )


        return (
            torch.tensor(
                sequencia,
                dtype=torch.float32
            ),

            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# BiGRU + ATTENTION
# ============================================================

class BiGRUAttention(nn.Module):


    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes
    ):


        super().__init__()


        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )


        saida_size = (
            hidden_size * 2
        )


        self.attention = nn.Sequential(

            nn.Linear(
                saida_size,
                64
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                1
            )
        )


        self.classificador = nn.Sequential(

            nn.Linear(
                saida_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):


        mascara = (
            x[
                :,
                :,
                -1
            ] > 0.5
        )


        saida, _ = self.gru(
            x
        )


        scores = (
            self.attention(
                saida
            )
            .squeeze(
                -1
            )
        )


        scores = scores.masked_fill(
            ~mascara,
            -1e9
        )


        pesos = torch.softmax(
            scores,
            dim=1
        )


        contexto = torch.sum(
            saida
            * pesos.unsqueeze(
                -1
            ),
            dim=1
        )


        return self.classificador(
            contexto
        )


# ============================================================
# PROBABILIDADES BiGRU
# ============================================================

def prever_probabilidades_bigru(
    modelo,
    loader
):


    modelo.eval()


    probabilidades = []

    labels = []


    with torch.no_grad():


        for X_batch, y_batch in loader:


            X_batch = X_batch.to(
                DEVICE
            )


            logits = modelo(
                X_batch
            )


            probs = torch.softmax(
                logits,
                dim=1
            )


            probabilidades.append(
                probs
                .cpu()
                .numpy()
            )


            labels.append(
                y_batch
                .numpy()
            )


    return (
        np.concatenate(
            probabilidades,
            axis=0
        ),

        np.concatenate(
            labels,
            axis=0
        )
    )


# ============================================================
# MATRIZ DE TRANSIÇÃO
# ============================================================

def calcular_transicoes(
    dataset_treino
):


    contagens = np.ones(
        (
            len(CLASSES),
            len(CLASSES)
        ),
        dtype=np.float64
    )


    ordenado = (
        dataset_treino
        .sort_values(
            [
                "id_noite",
                "segmento",
                "epoca"
            ]
        )
    )


    grupos = ordenado.groupby(
        [
            "id_noite",
            "segmento"
        ],
        sort=False
    )


    for _, grupo in grupos:


        labels = grupo[
            "label"
        ].to_numpy(
            dtype=int
        )


        epocas = grupo[
            "epoca"
        ].to_numpy()


        for i in range(
            1,
            len(grupo)
        ):


            if (
                epocas[i]
                - epocas[i - 1]
                != 1
            ):

                continue


            anterior = labels[
                i - 1
            ]


            atual = labels[
                i
            ]


            contagens[
                anterior,
                atual
            ] += 1.0


    return (
        contagens
        / contagens.sum(
            axis=1,
            keepdims=True
        )
    )


# ============================================================
# PROBABILIDADE INICIAL
# ============================================================

def calcular_probabilidade_inicial(
    dataset_treino
):


    contagens = np.ones(
        len(CLASSES),
        dtype=np.float64
    )


    ordenado = (
        dataset_treino
        .sort_values(
            [
                "id_noite",
                "segmento",
                "epoca"
            ]
        )
    )


    primeiros = (
        ordenado
        .groupby(
            [
                "id_noite",
                "segmento"
            ],
            sort=False
        )
        .first()
    )


    for label in primeiros[
        "label"
    ].astype(
        int
    ):

        contagens[
            label
        ] += 1.0


    return (
        contagens
        / contagens.sum()
    )


# ============================================================
# VITERBI
# ============================================================

def viterbi(
    probabilidades,
    transicoes,
    prob_inicial,
    peso
):


    if peso == 0:

        return np.argmax(
            probabilidades,
            axis=1
        )


    epsilon = 1e-12


    log_emissao = np.log(
        probabilidades
        + epsilon
    )


    log_transicao = np.log(
        transicoes
        + epsilon
    )


    log_inicial = np.log(
        prob_inicial
        + epsilon
    )


    T = probabilidades.shape[
        0
    ]


    K = probabilidades.shape[
        1
    ]


    score = np.zeros(
        (
            T,
            K
        )
    )


    caminho = np.zeros(
        (
            T,
            K
        ),
        dtype=np.int64
    )


    score[0] = (
        log_emissao[0]
        + peso * log_inicial
    )


    for t in range(
        1,
        T
    ):


        for atual in range(
            K
        ):


            candidatos = (
                score[
                    t - 1
                ]
                + peso
                * log_transicao[
                    :,
                    atual
                ]
            )


            melhor_anterior = int(
                np.argmax(
                    candidatos
                )
            )


            score[
                t,
                atual
            ] = (
                candidatos[
                    melhor_anterior
                ]
                + log_emissao[
                    t,
                    atual
                ]
            )


            caminho[
                t,
                atual
            ] = melhor_anterior


    resultado = np.zeros(
        T,
        dtype=np.int64
    )


    resultado[-1] = int(
        np.argmax(
            score[-1]
        )
    )


    for t in range(
        T - 2,
        -1,
        -1
    ):


        resultado[t] = caminho[
            t + 1,
            resultado[
                t + 1
            ]
        ]


    return resultado


# ============================================================
# APLICAR VITERBI POR SEGMENTO
# ============================================================

def aplicar_viterbi_dataset(
    metadata,
    probabilidades,
    transicoes,
    prob_inicial,
    peso
):


    if peso == 0:

        return np.argmax(
            probabilidades,
            axis=1
        )


    metadata = metadata.copy()


    metadata[
        "_posicao"
    ] = np.arange(
        len(metadata)
    )


    predicoes = np.zeros(
        len(metadata),
        dtype=np.int64
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


        probs_grupo = probabilidades[
            posicoes
        ]


        pred_grupo = viterbi(
            probs_grupo,
            transicoes,
            prob_inicial,
            peso
        )


        predicoes[
            posicoes
        ] = pred_grupo


    return predicoes


# ============================================================
# INÍCIO
# ============================================================

def main():


    definir_seed(
        SEED
    )


    print(
        "=" * 95
    )

    print(
        "ENSEMBLE RANDOM FOREST + BiGRU + VITERBI"
    )

    print(
        "=" * 95
    )


    print(
        f"\nDispositivo PyTorch: "
        f"{DEVICE}"
    )


    print(
        f"Janela BiGRU: "
        f"{TAMANHO_JANELA} épocas "
        f"({TAMANHO_JANELA * 0.5:.1f} min)"
    )


    print(
        f"\nCombinações:"
    )


    print(
        f"{len(PESOS_BIGRU)} pesos de ensemble "
        f"× "
        f"{len(PESOS_VITERBI)} pesos Viterbi "
        f"= "
        f"{len(PESOS_BIGRU) * len(PESOS_VITERBI)}"
    )


    # ========================================================
    # DADOS
    # ========================================================

    print(
        "\nCarregando dataset..."
    )


    dataset = pd.read_csv(
        ARQUIVO_DATASET
    )


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
        "\nTESTE FINAL CONTINUA FECHADO:"
    )


    for participante in participantes_teste:

        print(
            f"  - {participante}"
        )


    dataset_dev = (

        dataset[
            ~dataset[
                "participante"
            ].isin(
                participantes_teste
            )
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )


    if len(dataset_dev) != 48104:

        raise ValueError(
            "Esperávamos 48104 épocas."
        )


    print(
        f"\nÉpocas desenvolvimento: "
        f"{len(dataset_dev)}"
    )


    # ========================================================
    # VALIDAR FEATURES
    # ========================================================

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


    if (
        dataset_dev[
            FEATURES
        ]
        .isna()
        .sum()
        .sum()
        != 0
    ):

        raise ValueError(
            "Existem valores ausentes."
        )


    # ========================================================
    # JANELAS
    # ========================================================

    print(
        "\nConstruindo janelas temporais..."
    )


    indices_janelas = (
        construir_indices_janelas(
            dataset_dev,
            RAIO_JANELA
        )
    )


    print(
        "✅ Janelas construídas."
    )


    labels = (
        dataset_dev[
            "label"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    grupos = dataset_dev[
        "participante"
    ]


    # ========================================================
    # FOLDS
    # ========================================================

    cv = StratifiedGroupKFold(
        n_splits=6,
        shuffle=True,
        random_state=42
    )


    folds = list(
        cv.split(
            dataset_dev[
                FEATURES
            ],
            labels,
            groups=grupos
        )
    )


    resultados = []

    resultados_classes = []


    # ========================================================
    # 6 FOLDS
    # ========================================================

    for fold, (
        idx_treino,
        idx_validacao
    ) in enumerate(
        folds,
        start=1
    ):


        print(
            "\n"
            + "=" * 95
        )


        print(
            f"FOLD {fold}/6"
        )


        print(
            "=" * 95
        )


        definir_seed(
            SEED + fold
        )


        participantes_val = sorted(

            dataset_dev.iloc[
                idx_validacao
            ][
                "participante"
            ]
            .unique()
        )


        print(
            "\nValidação:"
        )


        for participante in participantes_val:

            print(
                f"  - {participante}"
            )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        print(
            "\nTreinando Random Forest..."
        )


        X_treino_rf = dataset_dev.iloc[
            idx_treino
        ][
            FEATURES
        ]


        y_treino = labels[
            idx_treino
        ]


        X_val_rf = dataset_dev.iloc[
            idx_validacao
        ][
            FEATURES
        ]


        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )


        rf.fit(
            X_treino_rf,
            y_treino
        )


        probs_rf_original = rf.predict_proba(
            X_val_rf
        )


        probs_rf = np.zeros(
            (
                len(idx_validacao),
                len(CLASSES)
            ),
            dtype=np.float64
        )


        for i, classe in enumerate(
            rf.classes_
        ):

            probs_rf[
                :,
                int(classe)
            ] = probs_rf_original[
                :,
                i
            ]


        print(
            "✅ Random Forest concluído."
        )


        # ====================================================
        # SCALER PARA BiGRU
        # ====================================================

        scaler = StandardScaler()


        scaler.fit(
            dataset_dev.iloc[
                idx_treino
            ][
                FEATURES
            ]
        )


        X_escalado = scaler.transform(
            dataset_dev[
                FEATURES
            ]
        ).astype(
            np.float32
        )


        # ====================================================
        # DATASET PYTORCH
        # ====================================================

        treino_torch = DatasetSequencial(
            idx_treino,
            X_escalado,
            labels,
            indices_janelas
        )


        val_torch = DatasetSequencial(
            idx_validacao,
            X_escalado,
            labels,
            indices_janelas
        )


        loader_treino = DataLoader(
            treino_torch,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )


        loader_val = DataLoader(
            val_torch,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )


        # ====================================================
        # PESOS DAS CLASSES
        # ====================================================

        contagens = np.bincount(
            y_treino,
            minlength=len(
                CLASSES
            )
        )


        pesos_classes = (
            len(y_treino)
            /
            (
                len(CLASSES)
                * contagens
            )
        )


        pesos_tensor = torch.tensor(
            pesos_classes,
            dtype=torch.float32,
            device=DEVICE
        )


        # ====================================================
        # BiGRU
        # ====================================================

        modelo = BiGRUAttention(
            input_size=len(FEATURES) + 1,
            hidden_size=HIDDEN_SIZE,
            num_classes=len(CLASSES)
        ).to(
            DEVICE
        )


        criterio = nn.CrossEntropyLoss(
            weight=pesos_tensor,
            label_smoothing=0.05
        )


        otimizador = torch.optim.AdamW(
            modelo.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )


        print(
            "\nTreinando BiGRU..."
        )


        for epoca in range(
            1,
            EPOCAS_TREINO + 1
        ):


            modelo.train()


            soma_loss = 0.0
            quantidade = 0


            for X_batch, y_batch in loader_treino:


                X_batch = X_batch.to(
                    DEVICE
                )


                y_batch = y_batch.to(
                    DEVICE
                )


                otimizador.zero_grad()


                logits = modelo(
                    X_batch
                )


                loss = criterio(
                    logits,
                    y_batch
                )


                loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    modelo.parameters(),
                    max_norm=5.0
                )


                otimizador.step()


                soma_loss += (
                    loss.item()
                    * len(y_batch)
                )


                quantidade += len(
                    y_batch
                )


            print(
                f"Época "
                f"{epoca:02d}/"
                f"{EPOCAS_TREINO}"
                f" | Loss "
                f"{soma_loss / quantidade:.4f}"
            )


        # ====================================================
        # PROBABILIDADES BiGRU
        # ====================================================

        probs_bigru, y_val = (
            prever_probabilidades_bigru(
                modelo,
                loader_val
            )
        )


        # ====================================================
        # MATRIZ DE TRANSIÇÃO
        # ====================================================

        dataset_treino_fold = (
            dataset_dev.iloc[
                idx_treino
            ]
            .copy()
        )


        matriz_transicao = (
            calcular_transicoes(
                dataset_treino_fold
            )
        )


        prob_inicial = (
            calcular_probabilidade_inicial(
                dataset_treino_fold
            )
        )


        metadata_val = (
            dataset_dev.iloc[
                idx_validacao
            ][
                [
                    "id_noite",
                    "segmento",
                    "epoca"
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        # ====================================================
        # TESTAR ENSEMBLES
        # ====================================================

        print(
            "\nTestando combinações..."
        )


        for peso_bigru in PESOS_BIGRU:


            # ------------------------------------------------
            # BLEND
            #
            # Ex:
            #
            # peso 0.75:
            #
            # 75% BiGRU
            # 25% RF
            #
            # ------------------------------------------------

            probs_ensemble = (

                peso_bigru
                * probs_bigru

                +

                (
                    1.0
                    - peso_bigru
                )
                * probs_rf
            )


            # Normalizar por segurança.

            probs_ensemble = (

                probs_ensemble

                / probs_ensemble.sum(
                    axis=1,
                    keepdims=True
                )
            )


            for peso_viterbi in PESOS_VITERBI:


                pred = aplicar_viterbi_dataset(

                    metadata_val,

                    probs_ensemble,

                    matriz_transicao,

                    prob_inicial,

                    peso_viterbi
                )


                accuracy = accuracy_score(
                    y_val,
                    pred
                )


                balanced = balanced_accuracy_score(
                    y_val,
                    pred
                )


                macro = f1_score(
                    y_val,
                    pred,
                    average="macro"
                )


                weighted = f1_score(
                    y_val,
                    pred,
                    average="weighted"
                )


                resultados.append({

                    "fold":
                        fold,

                    "participantes_validacao":
                        ", ".join(
                            participantes_val
                        ),

                    "peso_bigru":
                        peso_bigru,

                    "peso_rf":
                        1.0
                        - peso_bigru,

                    "peso_viterbi":
                        peso_viterbi,

                    "accuracy":
                        accuracy,

                    "balanced_accuracy":
                        balanced,

                    "macro_f1":
                        macro,

                    "weighted_f1":
                        weighted
                })


                (
                    precisao,
                    recall,
                    f1_classes,
                    suporte

                ) = (
                    precision_recall_fscore_support(

                        y_val,

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

                        "peso_bigru":
                            peso_bigru,

                        "peso_viterbi":
                            peso_viterbi,

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


        print(
            "✅ Fold concluído."
        )


        del modelo
        del rf
        del X_escalado


    # ========================================================
    # RESULTADOS
    # ========================================================

    df = pd.DataFrame(
        resultados
    )


    df_classes = pd.DataFrame(
        resultados_classes
    )


    ranking = (

        df

        .groupby(
            [
                "peso_bigru",
                "peso_rf",
                "peso_viterbi"
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


    # ========================================================
    # RANKING
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "TOP 15 CONFIGURAÇÕES"
    )

    print(
        "=" * 95
    )


    print(

        "\n"

        + ranking.head(
            15
        ).round(
            4
        ).to_string(
            index=False
        )
    )


    # ========================================================
    # MELHOR
    # ========================================================

    melhor = ranking.iloc[
        0
    ]


    melhor_bigru = float(
        melhor[
            "peso_bigru"
        ]
    )


    melhor_viterbi = float(
        melhor[
            "peso_viterbi"
        ]
    )


    print(
        "\n"
        + "=" * 95
    )

    print(
        "MELHOR ENSEMBLE"
    )

    print(
        "=" * 95
    )


    print(
        f"\nPeso BiGRU: "
        f"{melhor_bigru:.2f}"
    )


    print(
        f"Peso Random Forest: "
        f"{1.0 - melhor_bigru:.2f}"
    )


    print(
        f"Peso Viterbi: "
        f"{melhor_viterbi:.2f}"
    )


    print(
        f"\nAccuracy média: "
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


    # ========================================================
    # CLASSES DA MELHOR CONFIGURAÇÃO
    # ========================================================

    melhor_classes = (

        df_classes[
            (
                df_classes[
                    "peso_bigru"
                ] == melhor_bigru
            )
            &
            (
                df_classes[
                    "peso_viterbi"
                ] == melhor_viterbi
            )
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
        + "=" * 95
    )

    print(
        "DESEMPENHO POR CLASSE DO MELHOR ENSEMBLE"
    )

    print(
        "=" * 95
    )


    print(

        "\n"

        + melhor_classes.round(
            4
        ).to_string(
            index=False
        )
    )


    # ========================================================
    # COMPARAÇÃO
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "COMPARAÇÃO COM OS CAMPEÕES ANTERIORES"
    )

    print(
        "=" * 95
    )


    print(
        "\nRF + Viterbi:"
    )

    print(
        f"  Accuracy: "
        f"{RF_VITERBI_ACCURACY:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{RF_VITERBI_MACRO:.4f}"
    )


    print(
        "\nBiGRU:"
    )

    print(
        f"  Accuracy: "
        f"{BIGRU_ACCURACY:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{BIGRU_MACRO:.4f}"
    )


    print(
        "\nEnsemble:"
    )

    print(
        f"  Accuracy: "
        f"{melhor['accuracy_media']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{melhor['macro_f1_media']:.4f}"
    )


    # ========================================================
    # SALVAR
    # ========================================================

    df.to_csv(

        PASTA_RESULTADOS
        / "ensemble_rf_bigru_viterbi_folds.csv",

        index=False
    )


    ranking.to_csv(

        PASTA_RESULTADOS
        / "ensemble_rf_bigru_viterbi_ranking.csv",

        index=False
    )


    df_classes.to_csv(

        PASTA_RESULTADOS
        / "ensemble_rf_bigru_viterbi_classes.csv",

        index=False
    )


    melhor_classes.to_csv(

        PASTA_RESULTADOS
        / "ensemble_rf_bigru_viterbi_melhor_classes.csv",

        index=False
    )


    print(
        "\n"
        + "=" * 95
    )

    print(
        "EXPERIMENTO CONCLUÍDO"
    )

    print(
        "=" * 95
    )


    print(
        "\n⚠️ Bidslab16, Bidslab22 e "
        "Bidslab32 continuam completamente fechados."
    )


    print(
        "\n✅ TESTE FINAL NÃO FOI UTILIZADO"
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()