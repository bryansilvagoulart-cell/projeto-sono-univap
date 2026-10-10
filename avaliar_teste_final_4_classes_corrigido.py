from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 256

PESO_RF_ESPERADO = 0.75
PESO_BIGRU_ESPERADO = 0.25
PESO_VITERBI_ESPERADO = 0.80

PARTICIPANTES_TESTE = [
    "Bidslab01",
    "Bidslab35",
    "Bidslab47"
]

QUANTIDADE_TESTE_ESPERADA = 12252
NOITES_TESTE_ESPERADAS = 15


# ============================================================
# ARQUIVOS
# ============================================================

ARQUIVO_TESTE = Path(
    "data/processed/teste_final_4_classes/"
    "dataset_teste_final_features_4_classes.csv"
)

ARQUIVO_DESENVOLVIMENTO = Path(
    "data/processed/dataset_ml_4_classes.csv"
)

PASTA_MODELOS = Path(
    "models/4_classes"
)

ARQUIVO_RF = (
    PASTA_MODELOS
    / "random_forest_final_4_classes.joblib"
)

ARQUIVO_SCALER = (
    PASTA_MODELOS
    / "scaler_final_4_classes.joblib"
)

ARQUIVO_BIGRU = (
    PASTA_MODELOS
    / "bigru_final_4_classes.pt"
)

ARQUIVO_ENSEMBLE = (
    PASTA_MODELOS
    / "ensemble_final_4_classes.joblib"
)

ARQUIVO_MANIFESTO_MODELO = (
    PASTA_MODELOS
    / "manifesto_modelo_final_4_classes.json"
)


PASTA_RESULTADOS = Path(
    "data/processed/resultados_ml_4_classes_final"
)

ARQUIVO_RESUMO = (
    PASTA_RESULTADOS
    / "teste_final_4_classes_resumo.csv"
)

ARQUIVO_CLASSES = (
    PASTA_RESULTADOS
    / "teste_final_4_classes_por_classe.csv"
)

ARQUIVO_MATRIZ = (
    PASTA_RESULTADOS
    / "teste_final_4_classes_matriz_confusao.csv"
)

ARQUIVO_PARTICIPANTES = (
    PASTA_RESULTADOS
    / "teste_final_4_classes_por_participante.csv"
)

ARQUIVO_PREDICOES = (
    PASTA_RESULTADOS
    / "teste_final_4_classes_predicoes.csv"
)

ARQUIVO_MANIFESTO_TESTE = (
    PASTA_RESULTADOS
    / "manifesto_teste_final_4_classes.json"
)


# ============================================================
# CLASSES
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
# FUNÇÕES AUXILIARES
# ============================================================

def calcular_sha256(caminho):

    sha = hashlib.sha256()

    with open(
        caminho,
        "rb"
    ) as arquivo:

        while True:

            bloco = arquivo.read(
                1024 * 1024
            )

            if not bloco:

                break

            sha.update(
                bloco
            )

    return sha.hexdigest()


def carregar_checkpoint_bigru(
    caminho
):

    try:

        return torch.load(
            caminho,
            map_location=DEVICE,
            weights_only=False
        )

    except TypeError:

        return torch.load(
            caminho,
            map_location=DEVICE
        )


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
# PREVER PROBABILIDADES BiGRU
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

    print(
        "=" * 100
    )

    print(
        "TESTE FINAL INDEPENDENTE - MODELO OFICIAL DE 4 CLASSES"
    )

    print(
        "=" * 100
    )


    print(
        f"\nPyTorch: "
        f"{torch.__version__}"
    )


    print(
        f"Dispositivo: "
        f"{DEVICE}"
    )


    # ========================================================
    # IMPEDIR NOVA AVALIAÇÃO ACIDENTAL
    # ========================================================

    if ARQUIVO_RESUMO.exists():

        raise RuntimeError(
            "O teste final já possui um arquivo de resultado:\n"
            f"{ARQUIVO_RESUMO}\n\n"
            "Este script não executará novamente "
            "para evitar uma nova avaliação acidental."
        )


    # ========================================================
    # VALIDAR ARQUIVOS
    # ========================================================

    arquivos_necessarios = [
        ARQUIVO_TESTE,
        ARQUIVO_DESENVOLVIMENTO,
        ARQUIVO_RF,
        ARQUIVO_SCALER,
        ARQUIVO_BIGRU,
        ARQUIVO_ENSEMBLE,
        ARQUIVO_MANIFESTO_MODELO
    ]


    faltando = [
        str(
            arquivo
        )
        for arquivo in arquivos_necessarios
        if not arquivo.exists()
    ]


    if faltando:

        raise FileNotFoundError(
            "Arquivos necessários ausentes:\n"
            + "\n".join(
                faltando
            )
        )


    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CARREGAR MODELO CONGELADO
    # ========================================================

    print(
        "\nCarregando artefatos congelados..."
    )


    rf = joblib.load(
        ARQUIVO_RF
    )


    scaler = joblib.load(
        ARQUIVO_SCALER
    )


    config = joblib.load(
        ARQUIVO_ENSEMBLE
    )


    checkpoint = (
        carregar_checkpoint_bigru(
            ARQUIVO_BIGRU
        )
    )


    features = list(
        config[
            "features"
        ]
    )


    raio_janela = int(
        checkpoint[
            "raio_janela"
        ]
    )


    hidden_size = int(
        checkpoint[
            "hidden_size"
        ]
    )


    peso_rf = float(
        config[
            "peso_rf"
        ]
    )


    peso_bigru = float(
        config[
            "peso_bigru"
        ]
    )


    peso_viterbi = float(
        config[
            "peso_viterbi"
        ]
    )


    matriz_transicao = np.asarray(
        config[
            "matriz_transicao"
        ],
        dtype=np.float64
    )


    prob_inicial = np.asarray(
        config[
            "probabilidade_inicial"
        ],
        dtype=np.float64
    )


    if not np.isclose(
        peso_rf,
        PESO_RF_ESPERADO
    ):

        raise RuntimeError(
            "Peso do Random Forest diferente "
            "da configuração congelada."
        )


    if not np.isclose(
        peso_bigru,
        PESO_BIGRU_ESPERADO
    ):

        raise RuntimeError(
            "Peso da BiGRU diferente "
            "da configuração congelada."
        )


    if not np.isclose(
        peso_viterbi,
        PESO_VITERBI_ESPERADO
    ):

        raise RuntimeError(
            "Peso do Viterbi diferente "
            "da configuração congelada."
        )


    print(
        "✅ Artefatos carregados."
    )


    print(
        "\nConfiguração encontrada:"
    )


    print(
        f"  Random Forest = "
        f"{peso_rf:.2f}"
    )


    print(
        f"  BiGRU         = "
        f"{peso_bigru:.2f}"
    )


    print(
        f"  Viterbi       = "
        f"{peso_viterbi:.2f}"
    )


    print(
        f"  Janela        = "
        f"{raio_janela * 2 + 1} épocas"
    )


    print(
        f"  Features      = "
        f"{len(features)}"
    )


    # ========================================================
    # CARREGAR TESTE
    # ========================================================

    print(
        "\nCarregando conjunto de teste independente..."
    )


    teste = pd.read_csv(
        ARQUIVO_TESTE
    )


    desenvolvimento = pd.read_csv(
        ARQUIVO_DESENVOLVIMENTO,
        usecols=[
            "participante"
        ]
    )


    participantes_encontrados = sorted(
        teste[
            "participante"
        ]
        .astype(str)
        .unique()
        .tolist()
    )


    if participantes_encontrados != sorted(
        PARTICIPANTES_TESTE
    ):

        raise RuntimeError(
            "Os participantes do arquivo de teste "
            "não são exatamente os participantes congelados.\n"
            f"Encontrados: {participantes_encontrados}"
        )


    participantes_dev = set(
        desenvolvimento[
            "participante"
        ]
        .astype(str)
        .unique()
    )


    sobreposicao = (
        participantes_dev
        & set(
            participantes_encontrados
        )
    )


    if sobreposicao:

        raise RuntimeError(
            "Foi encontrada sobreposição entre "
            "desenvolvimento e teste:\n"
            + ", ".join(
                sorted(
                    sobreposicao
                )
            )
        )


    if len(teste) != QUANTIDADE_TESTE_ESPERADA:

        raise RuntimeError(
            f"Esperávamos "
            f"{QUANTIDADE_TESTE_ESPERADA} épocas, "
            f"mas encontramos {len(teste)}."
        )


    quantidade_noites = (
        teste[
            [
                "participante",
                "noite"
            ]
        ]
        .drop_duplicates()
        .shape[
            0
        ]
    )


    if quantidade_noites != NOITES_TESTE_ESPERADAS:

        raise RuntimeError(
            f"Esperávamos "
            f"{NOITES_TESTE_ESPERADAS} noites, "
            f"mas encontramos {quantidade_noites}."
        )


    faltando_features = [
        feature
        for feature in features
        if feature not in teste.columns
    ]


    if faltando_features:

        raise RuntimeError(
            "Features ausentes no teste:\n"
            + "\n".join(
                faltando_features
            )
        )


    if (
        teste[
            features
        ]
        .isna()
        .sum()
        .sum()
        != 0
    ):

        raise RuntimeError(
            "Existem valores ausentes "
            "nas features do teste."
        )


    valores_features = (
        teste[
            features
        ]
        .to_numpy(
            dtype=float
        )
    )


    if not np.isfinite(
        valores_features
    ).all():

        raise RuntimeError(
            "Existem valores infinitos "
            "nas features do teste."
        )


    labels = (
        teste[
            "label"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    if not set(
        np.unique(
            labels
        )
    ).issubset(
        set(
            CLASSES
        )
    ):

        raise RuntimeError(
            "Foram encontrados labels "
            "fora das quatro classes."
        )


    # ========================================================
    # CRIAR ID DE NOITE SOMENTE PARA AGRUPAMENTO
    # ========================================================
    #
    # O dataset novo usa participante + noite.
    # A validação antiga usava id_noite.
    # Aqui criamos um ID determinístico apenas para
    # manter a mesma lógica de agrupamento de janelas
    # e do Viterbi.
    # ========================================================

    teste = teste.copy()


    # Pandas 3.x não aceita mais diretamente uma lista
    # de tuplas em pd.factorize(). Para manter exatamente
    # a mesma ideia — um identificador determinístico para
    # cada combinação participante/noite — usamos ngroup().
    teste[
        "id_noite"
    ] = (
        teste
        .groupby(
            [
                "participante",
                "noite"
            ],
            sort=True
        )
        .ngroup()
        + 1
    )


    teste = (
        teste
        .sort_values(
            [
                "id_noite",
                "segmento",
                "epoca"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    labels = (
        teste[
            "label"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    print(
        "✅ Teste validado:"
    )


    print(
        f"  Participantes: "
        f"{teste['participante'].nunique()}"
    )


    print(
        f"  Noites: "
        f"{quantidade_noites}"
    )


    print(
        f"  Épocas: "
        f"{len(teste)}"
    )


    print(
        "  Sobreposição com desenvolvimento: 0"
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\nGerando probabilidades do Random Forest..."
    )


    probs_rf_original = rf.predict_proba(
        teste[
            features
        ]
    )


    probs_rf = np.zeros(
        (
            len(teste),
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


    # ========================================================
    # BiGRU
    # ========================================================

    print(
        "\nPreparando janelas da BiGRU..."
    )


    indices_janelas = (
        construir_indices_janelas(
            teste,
            raio_janela
        )
    )


    X_escalado = scaler.transform(
        teste[
            features
        ]
    ).astype(
        np.float32
    )


    indices_teste = np.arange(
        len(teste),
        dtype=np.int64
    )


    teste_torch = DatasetSequencial(

        indices_teste,

        X_escalado,

        labels,

        indices_janelas
    )


    loader_teste = DataLoader(

        teste_torch,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    modelo = BiGRUAttention(

        input_size=len(features) + 1,

        hidden_size=hidden_size,

        num_classes=len(CLASSES)

    ).to(
        DEVICE
    )


    modelo.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )


    probs_bigru, labels_loader = (
        prever_probabilidades_bigru(
            modelo,
            loader_teste
        )
    )


    if not np.array_equal(
        labels_loader,
        labels
    ):

        raise RuntimeError(
            "A ordem dos labels da BiGRU "
            "não corresponde ao dataset."
        )


    print(
        "✅ BiGRU concluída."
    )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    print(
        "\nCombinando RF + BiGRU..."
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

        / probs_ensemble.sum(
            axis=1,
            keepdims=True
        )
    )


    print(
        "✅ Ensemble concluído."
    )


    # ========================================================
    # VITERBI
    # ========================================================

    print(
        "\nAplicando Viterbi congelado..."
    )


    metadata = teste[
        [
            "id_noite",
            "segmento",
            "epoca"
        ]
    ].copy()


    pred_final = aplicar_viterbi_dataset(

        metadata,

        probs_ensemble,

        matriz_transicao,

        prob_inicial,

        peso_viterbi
    )


    print(
        "✅ Viterbi concluído."
    )


    # ========================================================
    # A PARTIR DAQUI O TESTE FOI ABERTO
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESULTADO FINAL OFICIAL - 4 CLASSES"
    )

    print(
        "=" * 100
    )


    accuracy = accuracy_score(
        labels,
        pred_final
    )


    balanced = balanced_accuracy_score(
        labels,
        pred_final
    )


    macro = f1_score(
        labels,
        pred_final,
        average="macro"
    )


    weighted = f1_score(
        labels,
        pred_final,
        average="weighted"
    )


    print(
        f"\nAccuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )


    print(
        f"Balanced Accuracy: "
        f"{balanced:.4f} "
        f"({balanced * 100:.2f}%)"
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

        labels,

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
        "RESULTADO POR CLASSE"
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


        dados_classes.append(
            {
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
                    int(
                        suporte[i]
                    )
            }
        )


    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    matriz = confusion_matrix(

        labels,

        pred_final,

        labels=CLASSES
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "MATRIZ DE CONFUSÃO"
    )

    print(
        "=" * 100
    )


    print(
        matriz
    )


    # ========================================================
    # PREDIÇÕES
    # ========================================================

    resultado_predicoes = teste[
        [
            "participante",
            "noite",
            "epoca",
            "segmento",
            "label_5_classes",
            "label_4_classes",
            "label",
            "classe_4"
        ]
    ].copy()


    resultado_predicoes[
        "predicao"
    ] = pred_final


    resultado_predicoes[
        "classe_prevista"
    ] = (
        resultado_predicoes[
            "predicao"
        ]
        .map(
            NOMES_CLASSES
        )
    )


    resultado_predicoes[
        "confianca_experimental"
    ] = probs_ensemble[
        np.arange(
            len(pred_final)
        ),
        pred_final
    ]


    # ========================================================
    # POR PARTICIPANTE
    # ========================================================

    participantes_resultados = []


    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESULTADO POR PARTICIPANTE"
    )

    print(
        "=" * 100
    )


    for participante, grupo in (
        resultado_predicoes.groupby(
            "participante"
        )
    ):


        y_real = (
            grupo[
                "label"
            ]
            .to_numpy()
        )


        y_pred = (
            grupo[
                "predicao"
            ]
            .to_numpy()
        )


        acc_p = accuracy_score(
            y_real,
            y_pred
        )


        bal_p = balanced_accuracy_score(
            y_real,
            y_pred
        )


        macro_p = f1_score(
            y_real,
            y_pred,
            average="macro"
        )


        weighted_p = f1_score(
            y_real,
            y_pred,
            average="weighted"
        )


        print(
            f"\n{participante}"
        )


        print(
            f"  Épocas: "
            f"{len(grupo)}"
        )


        print(
            f"  Accuracy: "
            f"{acc_p:.4f} "
            f"({acc_p * 100:.2f}%)"
        )


        print(
            f"  Balanced Accuracy: "
            f"{bal_p:.4f}"
        )


        print(
            f"  Macro F1: "
            f"{macro_p:.4f}"
        )


        participantes_resultados.append(
            {
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
            }
        )


    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    pd.DataFrame(
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
                    peso_rf,

                "peso_bigru":
                    peso_bigru,

                "peso_viterbi":
                    peso_viterbi,

                "quantidade_teste":
                    len(
                        teste
                    ),

                "participantes_teste":
                    3,

                "noites_teste":
                    quantidade_noites
            }
        ]
    ).to_csv(
        ARQUIVO_RESUMO,
        index=False
    )


    pd.DataFrame(
        dados_classes
    ).to_csv(
        ARQUIVO_CLASSES,
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
        ARQUIVO_MATRIZ
    )


    pd.DataFrame(
        participantes_resultados
    ).to_csv(
        ARQUIVO_PARTICIPANTES,
        index=False
    )


    resultado_predicoes.to_csv(
        ARQUIVO_PREDICOES,
        index=False
    )


    # ========================================================
    # MANIFESTO DO TESTE
    # ========================================================

    manifesto = {

        "avaliacao":
            "Teste final independente - 4 classes",

        "executado_em_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "participantes": (
            PARTICIPANTES_TESTE
        ),

        "quantidade_epocas":
            int(
                len(
                    teste
                )
            ),

        "quantidade_noites":
            int(
                quantidade_noites
            ),

        "dataset_teste":
            str(
                ARQUIVO_TESTE
            ),

        "dataset_teste_sha256":
            calcular_sha256(
                ARQUIVO_TESTE
            ),

        "artefatos_modelo": {

            "random_forest_sha256":
                calcular_sha256(
                    ARQUIVO_RF
                ),

            "scaler_sha256":
                calcular_sha256(
                    ARQUIVO_SCALER
                ),

            "bigru_sha256":
                calcular_sha256(
                    ARQUIVO_BIGRU
                ),

            "ensemble_sha256":
                calcular_sha256(
                    ARQUIVO_ENSEMBLE
                )
        },

        "configuracao": {

            "peso_rf":
                peso_rf,

            "peso_bigru":
                peso_bigru,

            "peso_viterbi":
                peso_viterbi,

            "janela_epocas":
                raio_janela
                * 2
                + 1,

            "features":
                features
        },

        "metricas": {

            "accuracy":
                float(
                    accuracy
                ),

            "balanced_accuracy":
                float(
                    balanced
                ),

            "macro_f1":
                float(
                    macro
                ),

            "weighted_f1":
                float(
                    weighted
                )
        },

        "observacao":
            (
                "Este arquivo registra a primeira "
                "avaliação oficial do modelo congelado "
                "nos participantes independentes "
                "Bidslab01, Bidslab35 e Bidslab47. "
                "A configuração não deve ser ajustada "
                "com base neste resultado."
            )
    }


    with open(
        ARQUIVO_MANIFESTO_TESTE,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            manifesto,
            arquivo,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TESTE FINAL CONCLUÍDO E REGISTRADO"
    )

    print(
        "=" * 100
    )


    print(
        "\nArquivos:"
    )


    for arquivo in [
        ARQUIVO_RESUMO,
        ARQUIVO_CLASSES,
        ARQUIVO_MATRIZ,
        ARQUIVO_PARTICIPANTES,
        ARQUIVO_PREDICOES,
        ARQUIVO_MANIFESTO_TESTE
    ]:

        print(
            f"  ✅ {arquivo}"
        )


    print(
        "\n🔒 O teste final agora foi utilizado."
    )


    print(
        "A configuração RF 0.75 + BiGRU 0.25 "
        "+ Viterbi 0.80 deve permanecer congelada."
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()
