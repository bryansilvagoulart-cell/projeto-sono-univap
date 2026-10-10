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
# CONFIGURAÇÕES GERAIS
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 256

EPOCAS_BIGRU = 8

HIDDEN_SIZE = 64

LEARNING_RATE = 0.001

RAIO_JANELA = 10

TAMANHO_JANELA = (
    RAIO_JANELA * 2 + 1
)


# ============================================================
# PESOS QUE SERÃO TESTADOS
# ============================================================
#
# peso_bigru = 0.00
# significa:
# 100% RF
#
# peso_bigru = 1.00
# significa:
# 100% BiGRU
#
# ============================================================

PESOS_BIGRU = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


PESOS_VITERBI = [
    0.00,
    0.20,
    0.40,
    0.60,
    0.80,
    1.00
]


# ============================================================
# ARQUIVOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_4_classes.csv"
)

PASTA_RESULTADOS = Path(
    "data/processed/resultados_ml_4_classes"
)


PASTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 30 FEATURES
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


    # CONTEXTO CURTO
    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean",


    # CONTEXTO MÉDIO
    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",

    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean",


    # CONTEXTO MAIOR
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


# ============================================================
# 4 CLASSES
# ============================================================

CLASSES = [
    0,
    1,
    2,
    3
]


NOMES_CLASSES = {
    0: "Wake",
    1: "Light",
    2: "Deep",
    3: "REM"
}


# ============================================================
# SEED
# ============================================================

def definir_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# CONSTRUIR JANELAS TEMPORAIS
# ============================================================

def construir_indices_janelas(
    dataset,
    raio
):

    tamanho = (
        raio * 2 + 1
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


        sequencia = self.X[
            seguros
        ].copy()


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
# PREVER PROBABILIDADES DA BiGRU
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
        + peso
        * log_inicial
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

                +

                peso
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

                +

                log_emissao[
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
# MAIN
# ============================================================

def main():

    definir_seed(
        SEED
    )


    print("=" * 100)
    print("VALIDAÇÃO CRUZADA - MODELO OFICIAL DE 4 CLASSES")
    print("=" * 100)


    print(
        f"\nPyTorch: "
        f"{torch.__version__}"
    )


    print(
        f"Dispositivo: "
        f"{DEVICE}"
    )


    print(
        f"Janela BiGRU: "
        f"{TAMANHO_JANELA} épocas "
        f"({TAMANHO_JANELA * 0.5:.1f} min)"
    )


    # ========================================================
    # CARREGAR
    # ========================================================

    print(
        "\nCarregando dataset..."
    )


    if not ARQUIVO_DATASET.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n"
            f"{ARQUIVO_DATASET}"
        )


    dataset = pd.read_csv(
        ARQUIVO_DATASET
    )


    print(
        f"Épocas: "
        f"{len(dataset)}"
    )


    print(
        f"Participantes: "
        f"{dataset['participante'].nunique()}"
    )


    print(
        f"Noites: "
        f"{dataset['id_noite'].nunique()}"
    )


    if len(dataset) != 60165:

        raise ValueError(
            "Esperávamos 60165 épocas."
        )


    if (
        dataset[
            "participante"
        ].nunique()
        != 15
    ):

        raise ValueError(
            "Esperávamos 15 participantes."
        )


    # ========================================================
    # VALIDAR FEATURES
    # ========================================================

    faltando = [
        feature
        for feature in FEATURES
        if feature not in dataset.columns
    ]


    if faltando:

        raise ValueError(
            "Features ausentes:\n"
            + "\n".join(
                faltando
            )
        )


    if (
        dataset[
            FEATURES
        ]
        .isna()
        .sum()
        .sum()
        != 0
    ):

        raise ValueError(
            "Existem valores ausentes nas features."
        )


    labels = (
        dataset[
            "label"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    if set(
        np.unique(
            labels
        )
    ) != {
        0,
        1,
        2,
        3
    }:

        raise ValueError(
            "As quatro classes esperadas "
            "não foram encontradas."
        )


    # ========================================================
    # DISTRIBUIÇÃO
    # ========================================================

    print(
        "\nDistribuição:"
    )


    for classe in CLASSES:

        quantidade = int(
            (
                labels
                == classe
            ).sum()
        )

        percentual = (
            quantidade
            / len(labels)
            * 100
        )

        print(
            f"  {NOMES_CLASSES[classe]:<6} "
            f"{quantidade:6d} "
            f"({percentual:6.2f}%)"
        )


    # ========================================================
    # JANELAS
    # ========================================================

    print(
        "\nConstruindo janelas temporais..."
    )


    indices_janelas = (
        construir_indices_janelas(
            dataset,
            RAIO_JANELA
        )
    )


    print(
        "✅ Janelas construídas."
    )


    # ========================================================
    # 5-FOLD POR PARTICIPANTE
    # ========================================================

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )


    folds = list(
        cv.split(
            dataset[
                FEATURES
            ],
            labels,
            groups=dataset[
                "participante"
            ]
        )
    )


    resultados = []

    resultados_classes = []


    # ========================================================
    # FOLDS
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
            + "=" * 100
        )

        print(
            f"FOLD {fold}/5"
        )

        print(
            "=" * 100
        )


        definir_seed(
            SEED + fold
        )


        participantes_treino = sorted(

            dataset.iloc[
                idx_treino
            ][
                "participante"
            ]
            .unique()
        )


        participantes_val = sorted(

            dataset.iloc[
                idx_validacao
            ][
                "participante"
            ]
            .unique()
        )


        print(
            f"\nTreino: "
            f"{len(participantes_treino)} participantes "
            f"| {len(idx_treino)} épocas"
        )


        print(
            "Validação:"
        )


        for participante in participantes_val:

            print(
                f"  - {participante}"
            )


        print(
            f"Épocas de validação: "
            f"{len(idx_validacao)}"
        )


        y_treino = labels[
            idx_treino
        ]


        y_validacao = labels[
            idx_validacao
        ]


        # ====================================================
        # CHECAR CLASSES
        # ====================================================

        if set(
            np.unique(
                y_treino
            )
        ) != {
            0,
            1,
            2,
            3
        }:

            raise RuntimeError(
                f"Fold {fold}: treino sem todas as classes."
            )


        if set(
            np.unique(
                y_validacao
            )
        ) != {
            0,
            1,
            2,
            3
        }:

            raise RuntimeError(
                f"Fold {fold}: validação sem todas as classes."
            )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        print(
            "\nTreinando Random Forest..."
        )


        rf = RandomForestClassifier(

            n_estimators=300,

            max_depth=16,

            min_samples_split=10,

            min_samples_leaf=5,

            max_features="sqrt",

            class_weight="balanced_subsample",

            random_state=SEED,

            n_jobs=-1
        )


        rf.fit(

            dataset.iloc[
                idx_treino
            ][
                FEATURES
            ],

            y_treino
        )


        probs_rf_original = rf.predict_proba(

            dataset.iloc[
                idx_validacao
            ][
                FEATURES
            ]
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


        pred_rf = np.argmax(
            probs_rf,
            axis=1
        )


        print(
            "✅ Random Forest treinado."
        )


        print(
            f"RF Accuracy: "
            f"{accuracy_score(y_validacao, pred_rf):.4f}"
        )


        print(
            f"RF Macro F1: "
            f"{f1_score(y_validacao, pred_rf, average='macro'):.4f}"
        )


        # ====================================================
        # NORMALIZAÇÃO DA BiGRU
        # ====================================================

        scaler = StandardScaler()


        scaler.fit(
            dataset.iloc[
                idx_treino
            ][
                FEATURES
            ]
        )


        X_escalado = scaler.transform(
            dataset[
                FEATURES
            ]
        ).astype(
            np.float32
        )


        # ====================================================
        # PYTORCH
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
        # PESOS DE CLASSE
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


        print(
            "\nPesos das classes:"
        )


        for classe in CLASSES:

            print(
                f"  {NOMES_CLASSES[classe]:<6}: "
                f"{pesos_classes[classe]:.4f}"
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
            EPOCAS_BIGRU + 1
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
                f"{EPOCAS_BIGRU}"
                f" | Loss "
                f"{soma_loss / quantidade:.4f}"
            )


        # ====================================================
        # PROBABILIDADES BiGRU
        # ====================================================

        probs_bigru, y_val_loader = (
            prever_probabilidades_bigru(
                modelo,
                loader_val
            )
        )


        if not np.array_equal(
            y_val_loader,
            y_validacao
        ):

            raise RuntimeError(
                "A ordem dos labels da validação "
                "não corresponde ao DataLoader."
            )


        pred_bigru = np.argmax(
            probs_bigru,
            axis=1
        )


        print(
            "\nBiGRU:"
        )


        print(
            f"Accuracy: "
            f"{accuracy_score(y_validacao, pred_bigru):.4f}"
        )


        print(
            f"Macro F1: "
            f"{f1_score(y_validacao, pred_bigru, average='macro'):.4f}"
        )


        # ====================================================
        # TRANSIÇÕES
        # ====================================================

        dataset_treino_fold = (
            dataset.iloc[
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

            dataset.iloc[
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
            "\nTestando ensembles + Viterbi..."
        )


        for peso_bigru in PESOS_BIGRU:

            peso_rf = (
                1.0
                - peso_bigru
            )


            probs_ensemble = (

                peso_rf
                * probs_rf

                +

                peso_bigru
                * probs_bigru
            )


            probs_ensemble = (

                probs_ensemble

                /

                probs_ensemble.sum(
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
                    y_validacao,
                    pred
                )


                balanced = balanced_accuracy_score(
                    y_validacao,
                    pred
                )


                macro = f1_score(
                    y_validacao,
                    pred,
                    average="macro"
                )


                weighted = f1_score(
                    y_validacao,
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

                    "peso_rf":
                        peso_rf,

                    "peso_bigru":
                        peso_bigru,

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

                ) = precision_recall_fscore_support(

                    y_validacao,

                    pred,

                    labels=CLASSES,

                    zero_division=0
                )


                for i, classe in enumerate(
                    CLASSES
                ):

                    resultados_classes.append({

                        "fold":
                            fold,

                        "peso_rf":
                            peso_rf,

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
                            int(
                                suporte[
                                    i
                                ]
                            )
                    })


        print(
            "✅ Fold concluído."
        )


        del rf
        del modelo
        del X_escalado


    # ========================================================
    # DATAFRAMES
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados
    )


    df_classes = pd.DataFrame(
        resultados_classes
    )


    # ========================================================
    # RANKING
    # ========================================================

    ranking = (

        df_resultados

        .groupby(
            [
                "peso_rf",
                "peso_bigru",
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
            [
                "macro_f1_media",
                "accuracy_media"
            ],
            ascending=[
                False,
                False
            ]
        )

        .reset_index(
            drop=True
        )
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "TOP 15 CONFIGURAÇÕES - 4 CLASSES"
    )

    print(
        "=" * 100
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
    # MELHOR CONFIGURAÇÃO
    # ========================================================

    melhor = ranking.iloc[
        0
    ]


    melhor_rf = float(
        melhor[
            "peso_rf"
        ]
    )


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
        + "=" * 100
    )

    print(
        "MELHOR CONFIGURAÇÃO"
    )

    print(
        "=" * 100
    )


    print(
        f"\nPeso Random Forest: "
        f"{melhor_rf:.2f}"
    )


    print(
        f"Peso BiGRU: "
        f"{melhor_bigru:.2f}"
    )


    print(
        f"Peso Viterbi: "
        f"{melhor_viterbi:.2f}"
    )


    print(
        f"\nAccuracy média: "
        f"{melhor['accuracy_media']:.4f} "
        f"({melhor['accuracy_media'] * 100:.2f}%)"
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
        f"Weighted F1 médio: "
        f"{melhor['weighted_f1_media']:.4f}"
    )


    print(
        f"Desvio Macro F1: "
        f"{melhor['macro_f1_desvio']:.4f}"
    )


    # ========================================================
    # RESULTADO POR CLASSE
    # ========================================================

    filtro_melhor = (

        np.isclose(
            df_classes[
                "peso_rf"
            ],
            melhor_rf
        )

        &

        np.isclose(
            df_classes[
                "peso_bigru"
            ],
            melhor_bigru
        )

        &

        np.isclose(
            df_classes[
                "peso_viterbi"
            ],
            melhor_viterbi
        )
    )


    melhor_classes = (

        df_classes[
            filtro_melhor
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
        + "=" * 100
    )

    print(
        "DESEMPENHO MÉDIO POR CLASSE"
    )

    print(
        "=" * 100
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
    # RESULTADO DA MELHOR CONFIGURAÇÃO POR FOLD
    # ========================================================

    filtro_resultados_melhor = (

        np.isclose(
            df_resultados[
                "peso_rf"
            ],
            melhor_rf
        )

        &

        np.isclose(
            df_resultados[
                "peso_bigru"
            ],
            melhor_bigru
        )

        &

        np.isclose(
            df_resultados[
                "peso_viterbi"
            ],
            melhor_viterbi
        )
    )


    melhor_folds = df_resultados[
        filtro_resultados_melhor
    ].copy()


    print(
        "\n"
        + "=" * 100
    )

    print(
        "MELHOR CONFIGURAÇÃO POR FOLD"
    )

    print(
        "=" * 100
    )


    print(

        "\n"

        + melhor_folds[
            [
                "fold",
                "participantes_validacao",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1"
            ]
        ]

        .round(
            4
        )

        .to_string(
            index=False
        )
    )


    # ========================================================
    # SALVAR
    # ========================================================

    df_resultados.to_csv(

        PASTA_RESULTADOS
        / "cv_4_classes_todas_configuracoes.csv",

        index=False
    )


    ranking.to_csv(

        PASTA_RESULTADOS
        / "cv_4_classes_ranking.csv",

        index=False
    )


    df_classes.to_csv(

        PASTA_RESULTADOS
        / "cv_4_classes_por_classe.csv",

        index=False
    )


    melhor_classes.to_csv(

        PASTA_RESULTADOS
        / "cv_4_classes_melhor_por_classe.csv",

        index=False
    )


    melhor_folds.to_csv(

        PASTA_RESULTADOS
        / "cv_4_classes_melhor_por_fold.csv",

        index=False
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "VALIDAÇÃO CRUZADA DE 4 CLASSES CONCLUÍDA"
    )

    print(
        "=" * 100
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
        "Os 15 participantes atuais foram utilizados "
        "somente como desenvolvimento/validação."
    )


    print(
        "O teste final da versão de 4 classes "
        "ainda NÃO foi realizado."
    )


    print(
        "\nPróximo passo:"
    )


    print(
        "Selecionar participantes novos e nunca vistos "
        "para o teste final."
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()