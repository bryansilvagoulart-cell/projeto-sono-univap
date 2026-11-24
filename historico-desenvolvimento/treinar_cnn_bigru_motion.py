from pathlib import Path
from copy import deepcopy
import random
import os

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TENTAR USAR BEM A CPU
# ============================================================

CPU_COUNT = os.cpu_count() or 1

torch.set_num_threads(
    max(
        1,
        CPU_COUNT - 1
    )
)


# ============================================================
# CONFIGURAÇÕES DA REDE
# ============================================================

RAIO_JANELA = 5

TAMANHO_JANELA = (
    RAIO_JANELA * 2
    + 1
)


BATCH_SIZE = 64

MAX_EPOCHS = 15

PATIENCE = 4

LEARNING_RATE = 0.001

HIDDEN_GRU = 64

EMBEDDING_MOTION = 64

EMBEDDING_TABULAR = 32


# ============================================================
# REFERÊNCIAS DOS EXPERIMENTOS ANTERIORES
# ============================================================

RF_VALIDACAO_MACRO = 0.4923
RF_VALIDACAO_ACCURACY = 0.5208

BIGRU_VALIDACAO_MACRO = 0.5283
BIGRU_VALIDACAO_ACCURACY = 0.5419


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
)

ARQUIVO_DIVISAO = Path(
    "data/processed/divisao_participantes_ml.csv"
)

ARQUIVO_MOTION = Path(
    "data/processed/motion_cnn_dev_10hz.npy"
)

ARQUIVO_METADATA = Path(
    "data/processed/motion_cnn_dev_metadata.csv"
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
# 30 FEATURES TABULARES
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
# FUNÇÃO DE SEED
# ============================================================

def definir_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# CRIAR JANELAS SEM ATRAVESSAR NOITES OU GAPS
# ============================================================

def construir_indices_janelas(
    dataset,
    raio
):

    tamanho = (
        raio * 2
        + 1
    )


    indices_janelas = np.full(

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


        for posicao_local in range(
            quantidade
        ):


            inicio = max(
                0,
                posicao_local - raio
            )


            fim = min(
                quantidade,
                posicao_local + raio + 1
            )


            esquerda = (
                posicao_local
                - inicio
            )


            destino_inicio = (
                raio
                - esquerda
            )


            destino_fim = (
                destino_inicio
                + (fim - inicio)
            )


            linha_original = posicoes[
                posicao_local
            ]


            indices_janelas[
                linha_original,
                destino_inicio:destino_fim
            ] = posicoes[
                inicio:fim
            ]


    return indices_janelas


# ============================================================
# NORMALIZAÇÃO DO MOTION APENAS COM O TREINO
# ============================================================

def calcular_normalizacao_motion(
    motion,
    indices_treino,
    tamanho_bloco=1000
):

    print(
        "\nCalculando normalização do movimento "
        "usando APENAS o treino..."
    )


    soma = np.zeros(
        4,
        dtype=np.float64
    )


    soma_quadrados = np.zeros(
        4,
        dtype=np.float64
    )


    quantidade_por_canal = 0


    for inicio in range(
        0,
        len(indices_treino),
        tamanho_bloco
    ):


        bloco_indices = indices_treino[
            inicio:
            inicio + tamanho_bloco
        ]


        bloco = np.asarray(
            motion[
                bloco_indices
            ],
            dtype=np.float64
        )


        soma += bloco.sum(
            axis=(0, 2)
        )


        soma_quadrados += (
            bloco ** 2
        ).sum(
            axis=(0, 2)
        )


        quantidade_por_canal += (
            bloco.shape[0]
            * bloco.shape[2]
        )


    media = (
        soma
        / quantidade_por_canal
    )


    variancia = (

        soma_quadrados
        / quantidade_por_canal

        - media ** 2
    )


    variancia = np.maximum(
        variancia,
        1e-12
    )


    desvio = np.sqrt(
        variancia
    )


    print(
        "\nMédia dos canais:"
    )

    print(
        media
    )


    print(
        "\nDesvio dos canais:"
    )

    print(
        desvio
    )


    return (
        media.astype(
            np.float32
        ),

        desvio.astype(
            np.float32
        )
    )


# ============================================================
# DATASET PYTORCH
# ============================================================

class DatasetCNNSequencial(Dataset):

    def __init__(
        self,
        indices_centrais,
        indices_janelas,
        motion,
        motion_media,
        motion_desvio,
        tabular_escalado,
        labels
    ):


        self.indices_centrais = np.asarray(
            indices_centrais,
            dtype=np.int64
        )


        self.indices_janelas = (
            indices_janelas
        )


        self.motion = motion


        self.motion_media = (
            motion_media
            .reshape(
                1,
                4,
                1
            )
        )


        self.motion_desvio = (
            motion_desvio
            .reshape(
                1,
                4,
                1
            )
        )


        self.tabular = (
            tabular_escalado
        )


        self.labels = labels


    def __len__(
        self
    ):

        return len(
            self.indices_centrais
        )


    def __getitem__(
        self,
        indice
    ):


        centro = self.indices_centrais[
            indice
        ]


        janela = self.indices_janelas[
            centro
        ]


        validos = (
            janela >= 0
        )


        seguros = janela.copy()


        seguros[
            ~validos
        ] = 0


        # ====================================================
        # MOVIMENTO
        # ====================================================

        motion_seq = np.asarray(

            self.motion[
                seguros
            ],

            dtype=np.float32
        ).copy()


        motion_seq = (

            motion_seq
            - self.motion_media

        ) / self.motion_desvio


        motion_seq[
            ~validos
        ] = 0.0


        # ====================================================
        # FEATURES TABULARES
        # ====================================================

        tabular_seq = (

            self.tabular[
                seguros
            ]

            .copy()
        )


        tabular_seq[
            ~validos
        ] = 0.0


        # ====================================================
        # MÁSCARA
        # ====================================================

        mascara = validos.astype(
            np.float32
        )


        label = int(
            self.labels[
                centro
            ]
        )


        return (

            torch.tensor(
                motion_seq,
                dtype=torch.float32
            ),

            torch.tensor(
                tabular_seq,
                dtype=torch.float32
            ),

            torch.tensor(
                mascara,
                dtype=torch.float32
            ),

            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# CNN 1D - CODIFICADOR DO MOVIMENTO
# ============================================================

class MotionCNN(nn.Module):

    def __init__(
        self,
        embedding_size
    ):

        super().__init__()


        self.cnn = nn.Sequential(

            # ------------------------------------------------
            # 4 x 300
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=4,
                out_channels=16,
                kernel_size=7,
                stride=2,
                padding=3
            ),

            nn.BatchNorm1d(
                16
            ),

            nn.ReLU(),


            nn.MaxPool1d(
                kernel_size=2
            ),


            # ------------------------------------------------
            # 16 x ~75
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.BatchNorm1d(
                32
            ),

            nn.ReLU(),


            # ------------------------------------------------
            # 32 x ~38
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),


            # ------------------------------------------------
            # Transformar qualquer comprimento em 1
            # ------------------------------------------------

            nn.AdaptiveAvgPool1d(
                1
            )
        )


        self.projecao = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                embedding_size
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            )
        )


    def forward(
        self,
        x
    ):


        x = self.cnn(
            x
        )


        x = self.projecao(
            x
        )


        return x


# ============================================================
# MODELO COMPLETO
# ============================================================

class CNNBiGRUAttention(nn.Module):

    def __init__(
        self,
        quantidade_features,
        motion_embedding,
        tabular_embedding,
        hidden_gru,
        num_classes
    ):

        super().__init__()


        # ====================================================
        # CNN DO MOVIMENTO
        # ====================================================

        self.motion_encoder = MotionCNN(

            embedding_size=
                motion_embedding
        )


        # ====================================================
        # PEQUENA REDE PARA AS FEATURES TABULARES
        # ====================================================

        self.tabular_encoder = nn.Sequential(

            nn.Linear(
                quantidade_features,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                tabular_embedding
            ),

            nn.ReLU()
        )


        tamanho_entrada_gru = (

            motion_embedding
            + tabular_embedding
        )


        # ====================================================
        # BiGRU
        # ====================================================

        self.gru = nn.GRU(

            input_size=
                tamanho_entrada_gru,

            hidden_size=
                hidden_gru,

            num_layers=1,

            batch_first=True,

            bidirectional=True
        )


        tamanho_saida_gru = (
            hidden_gru * 2
        )


        # ====================================================
        # ATTENTION
        # ====================================================

        self.attention = nn.Sequential(

            nn.Linear(
                tamanho_saida_gru,
                64
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                1
            )
        )


        # ====================================================
        # CLASSIFICADOR
        # ====================================================

        self.classificador = nn.Sequential(

            nn.Linear(
                tamanho_saida_gru,
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
        motion_seq,
        tabular_seq,
        mascara
    ):


        batch = motion_seq.shape[
            0
        ]


        sequencia = motion_seq.shape[
            1
        ]


        # ====================================================
        # CNN PROCESSA CADA ÉPOCA SEPARADAMENTE
        # ====================================================

        motion_flat = motion_seq.reshape(

            batch * sequencia,

            4,

            motion_seq.shape[-1]
        )


        embedding_motion = (
            self.motion_encoder(
                motion_flat
            )
        )


        embedding_motion = (
            embedding_motion.reshape(

                batch,

                sequencia,

                -1
            )
        )


        # ====================================================
        # FEATURES TABULARES
        # ====================================================

        tabular_flat = tabular_seq.reshape(

            batch * sequencia,

            tabular_seq.shape[-1]
        )


        embedding_tabular = (
            self.tabular_encoder(
                tabular_flat
            )
        )


        embedding_tabular = (
            embedding_tabular.reshape(

                batch,

                sequencia,

                -1
            )
        )


        # ====================================================
        # JUNTAR CNN + TABULAR
        # ====================================================

        combinado = torch.cat(

            [
                embedding_motion,
                embedding_tabular
            ],

            dim=2
        )


        # ====================================================
        # BiGRU
        # ====================================================

        saida_gru, _ = self.gru(
            combinado
        )


        # ====================================================
        # ATTENTION
        # ====================================================

        scores = (

            self.attention(
                saida_gru
            )

            .squeeze(
                -1
            )
        )


        mascara_bool = (
            mascara > 0.5
        )


        scores = scores.masked_fill(

            ~mascara_bool,

            -1e9
        )


        pesos = torch.softmax(

            scores,

            dim=1
        )


        contexto = torch.sum(

            saida_gru

            * pesos.unsqueeze(
                -1
            ),

            dim=1
        )


        logits = self.classificador(
            contexto
        )


        return logits


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar_modelo(
    modelo,
    loader
):

    modelo.eval()


    reais = []

    previstos = []


    with torch.no_grad():


        for (
            motion_batch,
            tabular_batch,
            mascara_batch,
            y_batch
        ) in loader:


            motion_batch = (
                motion_batch.to(
                    DEVICE
                )
            )


            tabular_batch = (
                tabular_batch.to(
                    DEVICE
                )
            )


            mascara_batch = (
                mascara_batch.to(
                    DEVICE
                )
            )


            logits = modelo(

                motion_batch,

                tabular_batch,

                mascara_batch
            )


            pred = torch.argmax(

                logits,

                dim=1
            )


            reais.extend(

                y_batch
                .numpy()
                .tolist()
            )


            previstos.extend(

                pred
                .cpu()
                .numpy()
                .tolist()
            )


    reais = np.asarray(
        reais
    )


    previstos = np.asarray(
        previstos
    )


    return {

        "accuracy":
            accuracy_score(
                reais,
                previstos
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                reais,
                previstos
            ),

        "macro_f1":
            f1_score(
                reais,
                previstos,
                average="macro"
            ),

        "weighted_f1":
            f1_score(
                reais,
                previstos,
                average="weighted"
            ),

        "y_true":
            reais,

        "y_pred":
            previstos
    }


# ============================================================
# INÍCIO
# ============================================================

def main():


    definir_seed(
        SEED
    )


    print(
        "=" * 90
    )

    print(
        "PROTÓTIPO CNN 1D + BiGRU + ATTENTION"
    )

    print(
        "=" * 90
    )


    print(
        f"\nPyTorch: "
        f"{torch.__version__}"
    )


    print(
        f"Dispositivo: "
        f"{DEVICE}"
    )


    print(
        f"Threads CPU: "
        f"{torch.get_num_threads()}"
    )


    print(
        f"Janela: "
        f"{TAMANHO_JANELA} épocas "
        f"({TAMANHO_JANELA * 0.5:.1f} min)"
    )


    # ========================================================
    # 1. CARREGAR DATASET PRINCIPAL
    # ========================================================

    print(
        "\nCarregando dataset principal..."
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


    # ========================================================
    # REMOVER TESTE
    # ========================================================

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
            "Esperávamos 48104 épocas "
            "de desenvolvimento."
        )


    # ========================================================
    # 2. CARREGAR METADADOS DO MOTION
    # ========================================================

    print(
        "\nCarregando metadados do movimento..."
    )


    metadata = pd.read_csv(
        ARQUIVO_METADATA
    )


    if len(metadata) != 48104:

        raise ValueError(
            "Metadados do motion não possuem "
            "48104 linhas."
        )


    # ========================================================
    # 3. VALIDAR ALINHAMENTO
    # ========================================================

    print(
        "\nValidando alinhamento dos datasets..."
    )


    for coluna in [
        "participante",
        "noite",
        "epoca"
    ]:


        valores_a = (

            dataset_dev[
                coluna
            ]

            .astype(
                str
            )

            .to_numpy()
        )


        valores_b = (

            metadata[
                coluna
            ]

            .astype(
                str
            )

            .to_numpy()
        )


        if not np.array_equal(
            valores_a,
            valores_b
        ):

            raise ValueError(
                f"Desalinhamento encontrado "
                f"na coluna {coluna}."
            )


    print(
        "✅ Dataset tabular e motion estão alinhados."
    )


    # ========================================================
    # 4. ABRIR NPY COMO MEMMAP
    # ========================================================

    motion = np.load(

        ARQUIVO_MOTION,

        mmap_mode="r"
    )


    print(
        f"\nMotion shape: "
        f"{motion.shape}"
    )


    if motion.shape != (
        48104,
        4,
        300
    ):

        raise ValueError(
            "Shape inesperado do motion."
        )


    # ========================================================
    # 5. VALIDAR FEATURES
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
            "Existem valores ausentes "
            "nas features."
        )


    # ========================================================
    # 6. DIVISÃO 10 / 2
    # ========================================================

    participantes_treino = (

        divisao[
            divisao[
                "grupo"
            ] == "treino"
        ]

        ["participante"]

        .tolist()
    )


    participantes_validacao = (

        divisao[
            divisao[
                "grupo"
            ] == "validacao"
        ]

        ["participante"]

        .tolist()
    )


    indices_treino = dataset_dev.index[

        dataset_dev[
            "participante"
        ].isin(
            participantes_treino
        )

    ].to_numpy()


    indices_validacao = dataset_dev.index[

        dataset_dev[
            "participante"
        ].isin(
            participantes_validacao
        )

    ].to_numpy()


    print(
        f"\nTreino: "
        f"{len(indices_treino)} épocas"
    )


    print(
        f"Validação: "
        f"{len(indices_validacao)} épocas"
    )


    print(
        "\nValidação:"
    )


    for participante in participantes_validacao:

        print(
            f"  - {participante}"
        )


    print(
        "\nTESTE FINAL CONTINUA FORA DESSE DATASET:"
    )


    for participante in participantes_teste:

        print(
            f"  - {participante}"
        )


    # ========================================================
    # 7. JANELAS TEMPORAIS
    # ========================================================

    print(
        "\nConstruindo janelas..."
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


    # ========================================================
    # 8. NORMALIZAR FEATURES TABULARES
    # ========================================================

    scaler = StandardScaler()


    scaler.fit(

        dataset_dev.loc[
            indices_treino,
            FEATURES
        ]
    )


    tabular_escalado = scaler.transform(

        dataset_dev[
            FEATURES
        ]

    ).astype(
        np.float32
    )


    # ========================================================
    # 9. NORMALIZAR MOVIMENTO
    # ========================================================

    (
        motion_media,
        motion_desvio

    ) = calcular_normalizacao_motion(

        motion,

        indices_treino
    )


    # ========================================================
    # 10. LABELS
    # ========================================================

    labels = (

        dataset_dev[
            "label"
        ]

        .astype(
            int
        )

        .to_numpy()
    )


    # ========================================================
    # 11. DATASETS PYTORCH
    # ========================================================

    treino_torch = DatasetCNNSequencial(

        indices_treino,

        indices_janelas,

        motion,

        motion_media,

        motion_desvio,

        tabular_escalado,

        labels
    )


    validacao_torch = DatasetCNNSequencial(

        indices_validacao,

        indices_janelas,

        motion,

        motion_media,

        motion_desvio,

        tabular_escalado,

        labels
    )


    loader_treino = DataLoader(

        treino_torch,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0
    )


    loader_validacao = DataLoader(

        validacao_torch,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    # ========================================================
    # 12. PESOS DAS CLASSES
    # ========================================================

    labels_treino = labels[
        indices_treino
    ]


    contagens = np.bincount(

        labels_treino,

        minlength=len(
            CLASSES
        )
    )


    pesos_classes = (

        len(
            labels_treino
        )

        /

        (
            len(
                CLASSES
            )

            * contagens
        )
    )


    print(
        "\nDistribuição do treino:"
    )


    for classe in CLASSES:

        print(

            f"{NOMES_CLASSES[classe]}: "
            f"{contagens[classe]} "
            f"| peso "
            f"{pesos_classes[classe]:.4f}"
        )


    pesos_tensor = torch.tensor(

        pesos_classes,

        dtype=torch.float32,

        device=DEVICE
    )


    # ========================================================
    # 13. MODELO
    # ========================================================

    modelo = CNNBiGRUAttention(

        quantidade_features=
            len(FEATURES),

        motion_embedding=
            EMBEDDING_MOTION,

        tabular_embedding=
            EMBEDDING_TABULAR,

        hidden_gru=
            HIDDEN_GRU,

        num_classes=
            len(CLASSES)

    ).to(
        DEVICE
    )


    quantidade_parametros = sum(

        parametro.numel()

        for parametro in modelo.parameters()

        if parametro.requires_grad
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "ARQUITETURA"
    )

    print(
        "=" * 90
    )


    print(
        modelo
    )


    print(
        f"\nParâmetros treináveis: "
        f"{quantidade_parametros:,}"
    )


    # ========================================================
    # 14. LOSS E OTIMIZADOR
    # ========================================================

    criterio = nn.CrossEntropyLoss(

        weight=
            pesos_tensor,

        label_smoothing=
            0.05
    )


    otimizador = torch.optim.AdamW(

        modelo.parameters(),

        lr=
            LEARNING_RATE,

        weight_decay=
            1e-4
    )


    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            otimizador,

            mode="max",

            factor=0.5,

            patience=2
        )
    )


    # ========================================================
    # 15. TREINAMENTO
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TREINAMENTO"
    )

    print(
        "=" * 90
    )


    melhor_macro = -1.0

    melhor_estado = None

    melhor_epoca = 0

    sem_melhora = 0


    historico = []


    for epoca in range(
        1,
        MAX_EPOCHS + 1
    ):


        modelo.train()


        soma_loss = 0.0

        quantidade = 0


        for (
            motion_batch,
            tabular_batch,
            mascara_batch,
            y_batch
        ) in loader_treino:


            motion_batch = (
                motion_batch.to(
                    DEVICE
                )
            )


            tabular_batch = (
                tabular_batch.to(
                    DEVICE
                )
            )


            mascara_batch = (
                mascara_batch.to(
                    DEVICE
                )
            )


            y_batch = y_batch.to(
                DEVICE
            )


            otimizador.zero_grad()


            logits = modelo(

                motion_batch,

                tabular_batch,

                mascara_batch
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

                * len(
                    y_batch
                )
            )


            quantidade += len(
                y_batch
            )


        loss_media = (

            soma_loss
            / quantidade
        )


        metricas = avaliar_modelo(

            modelo,

            loader_validacao
        )


        macro = metricas[
            "macro_f1"
        ]


        scheduler.step(
            macro
        )


        lr_atual = (
            otimizador
            .param_groups[0]
            ["lr"]
        )


        print(
            f"\nÉpoca "
            f"{epoca:02d}/"
            f"{MAX_EPOCHS}"
        )


        print(
            f"  Loss: "
            f"{loss_media:.4f}"
        )


        print(
            f"  Val Accuracy: "
            f"{metricas['accuracy']:.4f}"
        )


        print(
            f"  Val Balanced Acc: "
            f"{metricas['balanced_accuracy']:.4f}"
        )


        print(
            f"  Val Macro F1: "
            f"{macro:.4f}"
        )


        print(
            f"  Learning rate: "
            f"{lr_atual:.6f}"
        )


        historico.append({

            "epoca":
                epoca,

            "loss":
                loss_media,

            "accuracy_validacao":
                metricas[
                    "accuracy"
                ],

            "balanced_accuracy_validacao":
                metricas[
                    "balanced_accuracy"
                ],

            "macro_f1_validacao":
                macro,

            "weighted_f1_validacao":
                metricas[
                    "weighted_f1"
                ],

            "learning_rate":
                lr_atual
        })


        if macro > melhor_macro:


            melhor_macro = macro

            melhor_epoca = epoca

            melhor_estado = deepcopy(
                modelo.state_dict()
            )

            sem_melhora = 0


            print(
                "  ⭐ Novo melhor modelo."
            )


        else:

            sem_melhora += 1


        if sem_melhora >= PATIENCE:


            print(
                "\nEarly stopping ativado."
            )

            break


    # ========================================================
    # 16. RESTAURAR MELHOR MODELO
    # ========================================================

    if melhor_estado is None:

        raise RuntimeError(
            "Nenhum modelo foi salvo."
        )


    modelo.load_state_dict(
        melhor_estado
    )


    # ========================================================
    # 17. RESULTADO FINAL
    # ========================================================

    resultado = avaliar_modelo(

        modelo,

        loader_validacao
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "RESULTADO CNN + BiGRU"
    )

    print(
        "=" * 90
    )


    print(
        f"\nMelhor época: "
        f"{melhor_epoca}"
    )


    print(
        f"Accuracy: "
        f"{resultado['accuracy']:.4f}"
    )


    print(
        f"Balanced Accuracy: "
        f"{resultado['balanced_accuracy']:.4f}"
    )


    print(
        f"Macro F1: "
        f"{resultado['macro_f1']:.4f}"
    )


    print(
        f"Weighted F1: "
        f"{resultado['weighted_f1']:.4f}"
    )


    # ========================================================
    # 18. MÉTRICAS POR CLASSE
    # ========================================================

    (
        precisao,
        recall,
        f1_classes,
        suporte

    ) = precision_recall_fscore_support(

        resultado[
            "y_true"
        ],

        resultado[
            "y_pred"
        ],

        labels=CLASSES,

        zero_division=0
    )


    resultados_classes = []


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


        resultados_classes.append({

            "classe":
                classe,

            "nome":
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
    # 19. MATRIZ DE CONFUSÃO
    # ========================================================

    matriz = confusion_matrix(

        resultado[
            "y_true"
        ],

        resultado[
            "y_pred"
        ],

        labels=CLASSES
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "MATRIZ DE CONFUSÃO"
    )

    print(
        "=" * 90
    )


    print(
        matriz
    )


    # ========================================================
    # 20. COMPARAR COM MODELOS ANTERIORES
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "COMPARAÇÃO NA MESMA VALIDAÇÃO"
    )

    print(
        "=" * 90
    )


    print(
        "\nRandom Forest:"
    )

    print(
        f"  Accuracy: "
        f"{RF_VALIDACAO_ACCURACY:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{RF_VALIDACAO_MACRO:.4f}"
    )


    print(
        "\nBiGRU tabular:"
    )

    print(
        f"  Accuracy: "
        f"{BIGRU_VALIDACAO_ACCURACY:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{BIGRU_VALIDACAO_MACRO:.4f}"
    )


    print(
        "\nCNN + BiGRU:"
    )

    print(
        f"  Accuracy: "
        f"{resultado['accuracy']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{resultado['macro_f1']:.4f}"
    )


    print(
        "\nGanho contra BiGRU tabular:"
    )

    print(
        f"  Accuracy: "
        f"{resultado['accuracy'] - BIGRU_VALIDACAO_ACCURACY:+.4f}"
    )

    print(
        f"  Macro F1: "
        f"{resultado['macro_f1'] - BIGRU_VALIDACAO_MACRO:+.4f}"
    )


    # ========================================================
    # 21. SALVAR MODELO
    # ========================================================

    ARQUIVO_MODELO = (
        PASTA_MODELOS
        / "cnn_bigru_motion_prototipo.pt"
    )


    torch.save(

        {
            "state_dict":
                modelo.state_dict(),

            "features":
                FEATURES,

            "raio_janela":
                RAIO_JANELA,

            "motion_media":
                motion_media,

            "motion_desvio":
                motion_desvio,

            "tabular_media":
                scaler.mean_,

            "tabular_escala":
                scaler.scale_,

            "melhor_epoca":
                melhor_epoca,

            "accuracy_validacao":
                resultado[
                    "accuracy"
                ],

            "macro_f1_validacao":
                resultado[
                    "macro_f1"
                ]
        },

        ARQUIVO_MODELO
    )


    # ========================================================
    # 22. SALVAR RESULTADOS
    # ========================================================

    pd.DataFrame(
        historico
    ).to_csv(

        PASTA_RESULTADOS
        / "cnn_bigru_motion_historico.csv",

        index=False
    )


    pd.DataFrame(
        resultados_classes
    ).to_csv(

        PASTA_RESULTADOS
        / "cnn_bigru_motion_classes.csv",

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
        / "cnn_bigru_motion_matriz_confusao.csv"
    )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PROTÓTIPO CNN + BiGRU CONCLUÍDO"
    )

    print(
        "=" * 90
    )


    print(
        "\nModelo:"
    )

    print(
        ARQUIVO_MODELO
    )


    print(
        "\n⚠️ Os participantes finais de teste "
        "continuam fora do dataset de movimento."
    )


    print(
        "\n✅ TESTE FINAL CONTINUA FECHADO"
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()