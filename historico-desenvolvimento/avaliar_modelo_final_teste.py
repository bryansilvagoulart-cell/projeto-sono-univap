from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURAÇÕES DEFINITIVAS
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# Arquitetura congelada
# ------------------------------------------------------------

RAIO_JANELA = 10

TAMANHO_JANELA = (
    RAIO_JANELA * 2
    + 1
)


HIDDEN_SIZE = 64

BATCH_SIZE = 256

EPOCAS_TREINO_BIGRU = 8

LEARNING_RATE = 0.001


# ------------------------------------------------------------
# Ensemble DEFINITIVO
# ------------------------------------------------------------

PESO_RF = 0.75

PESO_BIGRU = 0.25

PESO_VITERBI = 0.60


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
)

ARQUIVO_DIVISAO = Path(
    "data/processed/divisao_participantes_ml.csv"
)

PASTA_MODELOS = Path(
    "models"
)

PASTA_RESULTADOS = Path(
    "data/processed/resultados_ml"
)


PASTA_MODELOS.mkdir(
    parents=True,
    exist_ok=True
)

PASTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES DEFINITIVAS
# ============================================================

FEATURES = [

    # ÉPOCA
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",

    "motion_mean",
    "motion_std",
    "motion_max",


    # CONTEXTO 2,5 MIN
    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean",


    # CONTEXTO 5 MIN
    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",

    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean",


    # CONTEXTO 10 MIN
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


    T = probabilidades.shape[0]

    K = probabilidades.shape[1]


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
# PROGRAMA PRINCIPAL
# ============================================================

def main():


    definir_seed(
        SEED
    )


    print("=" * 100)
    print("AVALIAÇÃO FINAL - ENSEMBLE RF + BiGRU + VITERBI")
    print("=" * 100)


    print(
        "\nCONFIGURAÇÃO CONGELADA:"
    )


    print(
        f"Random Forest: "
        f"{PESO_RF:.2f}"
    )


    print(
        f"BiGRU: "
        f"{PESO_BIGRU:.2f}"
    )


    print(
        f"Viterbi: "
        f"{PESO_VITERBI:.2f}"
    )


    print(
        f"Janela BiGRU: "
        f"{TAMANHO_JANELA} épocas "
        f"({TAMANHO_JANELA * 0.5:.1f} min)"
    )


    print(
        f"Épocas BiGRU: "
        f"{EPOCAS_TREINO_BIGRU}"
    )


    # ========================================================
    # CARREGAR DADOS
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
        "\nPARTICIPANTES DO TESTE FINAL:"
    )


    for participante in participantes_teste:

        print(
            f"  - {participante}"
        )


    # ========================================================
    # DESENVOLVIMENTO E TESTE
    # ========================================================

    mascara_teste = dataset[
        "participante"
    ].isin(
        participantes_teste
    )


    dataset_dev = (
        dataset[
            ~mascara_teste
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    dataset_teste = (
        dataset[
            mascara_teste
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    if len(dataset_dev) != 48104:

        raise ValueError(
            "Desenvolvimento deveria possuir "
            "48104 épocas."
        )


    if len(dataset_teste) != 12061:

        raise ValueError(
            "Teste deveria possuir "
            "12061 épocas."
        )


    print(
        f"\nDesenvolvimento: "
        f"{len(dataset_dev)} épocas"
    )


    print(
        f"Teste final: "
        f"{len(dataset_teste)} épocas"
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


    # ========================================================
    # CRIAR DATASET TEMPORÁRIO COM DEV + TESTE
    # ========================================================
    #
    # Isso é necessário somente para construir corretamente
    # as janelas de cada conjunto.
    #
    # As janelas nunca atravessam participantes/noites.
    #
    # ========================================================

    dataset_completo = pd.concat(
        [
            dataset_dev,
            dataset_teste
        ],
        ignore_index=True
    )


    labels_completos = (
        dataset_completo[
            "label"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    indices_dev = np.arange(
        len(dataset_dev)
    )


    indices_teste = np.arange(
        len(dataset_dev),
        len(dataset_completo)
    )


    print(
        "\nConstruindo janelas temporais..."
    )


    indices_janelas = (
        construir_indices_janelas(
            dataset_completo,
            RAIO_JANELA
        )
    )


    print(
        "✅ Janelas construídas."
    )


    # ========================================================
    # RANDOM FOREST FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TREINANDO RANDOM FOREST FINAL"
    )

    print(
        "=" * 100
    )


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

        dataset_dev[
            FEATURES
        ],

        dataset_dev[
            "label"
        ]
    )


    probs_rf_original = rf.predict_proba(

        dataset_teste[
            FEATURES
        ]
    )


    probs_rf = np.zeros(
        (
            len(dataset_teste),
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
        "✅ Random Forest final treinado."
    )


    # ========================================================
    # SCALER
    # ========================================================

    print(
        "\nNormalizando features para BiGRU..."
    )


    scaler = StandardScaler()


    scaler.fit(
        dataset_dev[
            FEATURES
        ]
    )


    X_escalado = scaler.transform(
        dataset_completo[
            FEATURES
        ]
    ).astype(
        np.float32
    )


    # ========================================================
    # PYTORCH DATASETS
    # ========================================================

    treino_torch = DatasetSequencial(

        indices_dev,

        X_escalado,

        labels_completos,

        indices_janelas
    )


    teste_torch = DatasetSequencial(

        indices_teste,

        X_escalado,

        labels_completos,

        indices_janelas
    )


    loader_treino = DataLoader(

        treino_torch,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0
    )


    loader_teste = DataLoader(

        teste_torch,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    # ========================================================
    # PESOS DE CLASSE
    # ========================================================

    labels_dev = (
        dataset_dev[
            "label"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    contagens = np.bincount(

        labels_dev,

        minlength=len(
            CLASSES
        )
    )


    pesos_classes = (

        len(labels_dev)

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


    # ========================================================
    # BiGRU FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TREINANDO BiGRU FINAL"
    )

    print(
        "=" * 100
    )


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


    for epoca in range(
        1,
        EPOCAS_TREINO_BIGRU + 1
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
            f"{EPOCAS_TREINO_BIGRU}"

            f" | Loss: "
            f"{soma_loss / quantidade:.4f}"
        )


    # ========================================================
    # PROBABILIDADES DO TESTE
    # ========================================================

    print(
        "\nGerando probabilidades da BiGRU..."
    )


    probs_bigru, y_teste = (
        prever_probabilidades_bigru(

            modelo,

            loader_teste
        )
    )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    print(
        "\nCombinando modelos..."
    )


    probs_ensemble = (

        PESO_RF
        * probs_rf

        +

        PESO_BIGRU
        * probs_bigru
    )


    probs_ensemble = (

        probs_ensemble

        / probs_ensemble.sum(
            axis=1,
            keepdims=True
        )
    )


    # ========================================================
    # TRANSIÇÕES APRENDIDAS SOMENTE NO DESENVOLVIMENTO
    # ========================================================

    matriz_transicao = (
        calcular_transicoes(
            dataset_dev
        )
    )


    prob_inicial = (
        calcular_probabilidade_inicial(
            dataset_dev
        )
    )


    metadata_teste = dataset_teste[
        [
            "id_noite",
            "segmento",
            "epoca"
        ]
    ].copy()


    # ========================================================
    # VITERBI FINAL
    # ========================================================

    print(
        "Aplicando Viterbi..."
    )


    pred_final = aplicar_viterbi_dataset(

        metadata_teste,

        probs_ensemble,

        matriz_transicao,

        prob_inicial,

        PESO_VITERBI
    )


    # ========================================================
    # MÉTRICAS FINAIS
    # ========================================================

    accuracy = accuracy_score(
        y_teste,
        pred_final
    )


    balanced = balanced_accuracy_score(
        y_teste,
        pred_final
    )


    macro = f1_score(
        y_teste,
        pred_final,
        average="macro"
    )


    weighted = f1_score(
        y_teste,
        pred_final,
        average="weighted"
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESULTADO FINAL NO CONJUNTO DE TESTE"
    )

    print(
        "=" * 100
    )


    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )


    print(
        f"Balanced Accuracy: "
        f"{balanced:.4f}"
    )


    print(
        f"Macro F1: "
        f"{macro:.4f}"
    )


    print(
        f"Weighted F1: "
        f"{weighted:.4f}"
    )


    # ========================================================
    # POR CLASSE
    # ========================================================

    (
        precisao,
        recall,
        f1_classes,
        suporte

    ) = precision_recall_fscore_support(

        y_teste,

        pred_final,

        labels=CLASSES,

        zero_division=0
    )


    dados_classes = []


    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESULTADO FINAL POR CLASSE"
    )

    print(
        "=" * 100
    )


    for i, classe in enumerate(
        CLASSES
    ):


        print(
            f"\n{NOMES_CLASSES[classe]}"
        )


        print(
            f"  Precision: "
            f"{precisao[i]:.4f}"
        )


        print(
            f"  Recall: "
            f"{recall[i]:.4f}"
        )


        print(
            f"  F1: "
            f"{f1_classes[i]:.4f}"
        )


        print(
            f"  Support: "
            f"{suporte[i]}"
        )


        dados_classes.append({

            "classe":
                classe,

            "nome_classe":
                NOMES_CLASSES[
                    classe
                ],

            "precision":
                precisao[i],

            "recall":
                recall[i],

            "f1":
                f1_classes[i],

            "support":
                suporte[i]
        })


    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    matriz = confusion_matrix(

        y_teste,

        pred_final,

        labels=CLASSES
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "MATRIZ DE CONFUSÃO FINAL"
    )

    print(
        "=" * 100
    )


    print(
        matriz
    )


    # ========================================================
    # RESULTADO POR PARTICIPANTE
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESULTADO POR PARTICIPANTE DO TESTE"
    )

    print(
        "=" * 100
    )


    resultado_predicoes = (
        dataset_teste[
            [
                "participante",
                "noite",
                "epoca",
                "id_noite",
                "segmento",
                "label"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    resultado_predicoes[
        "predicao"
    ] = pred_final


    resultado_predicoes[
        "confianca"
    ] = probs_ensemble[
        np.arange(
            len(pred_final)
        ),
        pred_final
    ]


    participantes_resultados = []


    for participante, grupo in (
        resultado_predicoes.groupby(
            "participante"
        )
    ):


        y_real_participante = (
            grupo[
                "label"
            ].to_numpy()
        )


        y_pred_participante = (
            grupo[
                "predicao"
            ].to_numpy()
        )


        acc_p = accuracy_score(
            y_real_participante,
            y_pred_participante
        )


        bal_p = balanced_accuracy_score(
            y_real_participante,
            y_pred_participante
        )


        macro_p = f1_score(
            y_real_participante,
            y_pred_participante,
            average="macro"
        )


        print(
            f"\n{participante}"
        )


        print(
            f"  Accuracy: "
            f"{acc_p:.4f}"
        )


        print(
            f"  Balanced Accuracy: "
            f"{bal_p:.4f}"
        )


        print(
            f"  Macro F1: "
            f"{macro_p:.4f}"
        )


        participantes_resultados.append({

            "participante":
                participante,

            "accuracy":
                acc_p,

            "balanced_accuracy":
                bal_p,

            "macro_f1":
                macro_p,

            "epocas":
                len(grupo)
        })


    # ========================================================
    # SALVAR MODELOS
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SALVANDO MODELO FINAL"
    )

    print(
        "=" * 100
    )


    ARQUIVO_RF = (
        PASTA_MODELOS
        / "random_forest_final.joblib"
    )


    ARQUIVO_BIGRU = (
        PASTA_MODELOS
        / "bigru_final.pt"
    )


    ARQUIVO_ENSEMBLE = (
        PASTA_MODELOS
        / "ensemble_final_config.joblib"
    )


    joblib.dump(
        rf,
        ARQUIVO_RF
    )


    torch.save(

        {
            "state_dict":
                modelo.state_dict(),

            "features":
                FEATURES,

            "hidden_size":
                HIDDEN_SIZE,

            "raio_janela":
                RAIO_JANELA,

            "epocas_treinamento":
                EPOCAS_TREINO_BIGRU,

            "scaler_mean":
                scaler.mean_,

            "scaler_scale":
                scaler.scale_
        },

        ARQUIVO_BIGRU
    )


    joblib.dump(

        {
            "peso_rf":
                PESO_RF,

            "peso_bigru":
                PESO_BIGRU,

            "peso_viterbi":
                PESO_VITERBI,

            "matriz_transicao":
                matriz_transicao,

            "probabilidade_inicial":
                prob_inicial,

            "features":
                FEATURES
        },

        ARQUIVO_ENSEMBLE
    )


    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    pd.DataFrame(
        dados_classes
    ).to_csv(

        PASTA_RESULTADOS
        / "teste_final_classes.csv",

        index=False
    )


    pd.DataFrame(
        matriz,

        index=[
            NOMES_CLASSES[
                classe
            ]
            for classe in CLASSES
        ],

        columns=[
            NOMES_CLASSES[
                classe
            ]
            for classe in CLASSES
        ]

    ).to_csv(

        PASTA_RESULTADOS
        / "teste_final_matriz_confusao.csv"
    )


    pd.DataFrame(
        participantes_resultados
    ).to_csv(

        PASTA_RESULTADOS
        / "teste_final_participantes.csv",

        index=False
    )


    resultado_predicoes.to_csv(

        PASTA_RESULTADOS
        / "teste_final_predicoes.csv",

        index=False
    )


    resumo_final = pd.DataFrame(
        [
            {
                "accuracy":
                    accuracy,

                "balanced_accuracy":
                    balanced,

                "macro_f1":
                    macro,

                "weighted_f1":
                    weighted,

                "peso_rf":
                    PESO_RF,

                "peso_bigru":
                    PESO_BIGRU,

                "peso_viterbi":
                    PESO_VITERBI,

                "quantidade_teste":
                    len(
                        y_teste
                    )
            }
        ]
    )


    resumo_final.to_csv(

        PASTA_RESULTADOS
        / "teste_final_resumo.csv",

        index=False
    )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AVALIAÇÃO FINAL CONCLUÍDA"
    )

    print(
        "=" * 100
    )


    print(
        "\nModelos salvos:"
    )


    print(
        ARQUIVO_RF
    )


    print(
        ARQUIVO_BIGRU
    )


    print(
        ARQUIVO_ENSEMBLE
    )


    print(
        "\nResultados salvos em:"
    )


    print(
        PASTA_RESULTADOS
    )


    print(
        "\n🔒 O conjunto de teste agora foi utilizado."
    )


    print(
        "A configuração do modelo deve permanecer "
        "congelada a partir deste ponto."
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()