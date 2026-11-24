from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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

TAMANHO_JANELA = (
    RAIO_JANELA * 2
    + 1
)


# ============================================================
# REFERÊNCIA ATUAL
# ============================================================

RF_VITERBI_MACRO_F1 = 0.5117

RF_VITERBI_ACCURACY = 0.5632


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
# SEEDS
# ============================================================

def definir_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# CONSTRUIR JANELAS
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
                pos_local
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

        self.X_escalado = X_escalado

        self.labels = labels

        self.indices_janelas = indices_janelas


    def __len__(
        self
    ):

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
            self.X_escalado[
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
# MODELO
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
            ]
            > 0.5
        )


        saida_gru, _ = self.gru(
            x
        )


        scores = (

            self.attention(
                saida_gru
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

            saida_gru

            * pesos.unsqueeze(
                -1
            ),

            dim=1
        )


        return self.classificador(
            contexto
        )


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar(
    modelo,
    loader
):

    modelo.eval()


    reais = []

    previstos = []


    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )


            logits = modelo(
                X_batch
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


    return (
        reais,
        previstos
    )


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
        "VALIDAÇÃO CRUZADA POR PARTICIPANTE"
    )

    print(
        "BiGRU + ATTENTION"
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
        f"Features: "
        f"{len(FEATURES)}"
    )


    print(
        f"Janela: "
        f"{TAMANHO_JANELA} épocas "
        f"({TAMANHO_JANELA * 0.5:.1f} min)"
    )


    print(
        f"Épocas de treinamento por fold: "
        f"{EPOCAS_TREINO}"
    )


    # ========================================================
    # CARREGAR DATASET
    # ========================================================

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


    # ========================================================
    # TESTE FINAL
    # ========================================================

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


    # ========================================================
    # REMOVER COMPLETAMENTE O TESTE
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


    print(
        f"\nÉpocas desenvolvimento: "
        f"{len(dataset_dev)}"
    )


    print(
        f"Participantes desenvolvimento: "
        f"{dataset_dev['participante'].nunique()}"
    )


    if len(dataset_dev) != 48104:

        raise ValueError(
            "Esperávamos 48104 épocas "
            "de desenvolvimento."
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


    print(
        "✅ Dados validados."
    )


    # ========================================================
    # LABELS
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


    grupos = dataset_dev[
        "participante"
    ]


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


    # ========================================================
    # 6 FOLDS
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


    resultados_folds = []

    resultados_classes = []


    # ========================================================
    # EXECUTAR FOLDS
    # ========================================================

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


        definir_seed(
            SEED + fold
        )


        participantes_validacao = sorted(

            dataset_dev.iloc[
                indices_validacao
            ][
                "participante"
            ]
            .unique()
        )


        print(
            "\nValidação:"
        )


        for participante in participantes_validacao:

            print(
                f"  - {participante}"
            )


        print(
            f"\nTreino: "
            f"{len(indices_treino)} épocas"
        )


        print(
            f"Validação: "
            f"{len(indices_validacao)} épocas"
        )


        # ====================================================
        # SCALER APENAS NO TREINO
        # ====================================================

        scaler = StandardScaler()


        scaler.fit(

            dataset_dev.iloc[
                indices_treino
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
        # PYTORCH DATASETS
        # ====================================================

        dataset_treino = DatasetSequencial(

            indices_treino,

            X_escalado,

            labels,

            indices_janelas
        )


        dataset_validacao = DatasetSequencial(

            indices_validacao,

            X_escalado,

            labels,

            indices_janelas
        )


        loader_treino = DataLoader(

            dataset_treino,

            batch_size=BATCH_SIZE,

            shuffle=True,

            num_workers=0
        )


        loader_validacao = DataLoader(

            dataset_validacao,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=0
        )


        # ====================================================
        # PESOS DAS CLASSES
        # ====================================================

        labels_treino = labels[
            indices_treino
        ]


        contagens = np.bincount(

            labels_treino,

            minlength=len(
                CLASSES
            )
        )


        pesos = (

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


        pesos_tensor = torch.tensor(

            pesos,

            dtype=torch.float32,

            device=DEVICE
        )


        # ====================================================
        # MODELO NOVO PARA ESTE FOLD
        # ====================================================

        modelo = BiGRUAttention(

            input_size=(
                len(FEATURES)
                + 1
            ),

            hidden_size=HIDDEN_SIZE,

            num_classes=len(
                CLASSES
            )

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


        # ====================================================
        # TREINAMENTO FIXO
        # ====================================================
        #
        # IMPORTANTE:
        #
        # NÃO usamos o resultado da validação
        # para escolher a época.
        #
        # O fold de validação permanece intocado
        # até o final das 8 épocas.
        #
        # ====================================================

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


            print(
                f"Época "
                f"{epoca:02d}/"
                f"{EPOCAS_TREINO}"
                f" | "
                f"Loss: "
                f"{loss_media:.4f}"
            )


        # ====================================================
        # SOMENTE AGORA ABRIMOS A VALIDAÇÃO DO FOLD
        # ====================================================

        reais, previstos = avaliar(

            modelo,

            loader_validacao
        )


        accuracy = accuracy_score(
            reais,
            previstos
        )


        balanced = balanced_accuracy_score(
            reais,
            previstos
        )


        macro = f1_score(
            reais,
            previstos,
            average="macro"
        )


        weighted = f1_score(
            reais,
            previstos,
            average="weighted"
        )


        print(
            "\nRESULTADO DO FOLD"
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
            f"{macro:.4f}"
        )


        print(
            f"Weighted F1: "
            f"{weighted:.4f}"
        )


        resultados_folds.append({

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
                macro,

            "weighted_f1":
                weighted
        })


        # ====================================================
        # MÉTRICAS POR CLASSE
        # ====================================================

        precisao, recall, f1_classes, suporte = (
            precision_recall_fscore_support(

                reais,

                previstos,

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


        # Liberar objetos grandes deste fold.

        del modelo
        del otimizador
        del loader_treino
        del loader_validacao
        del X_escalado


    # ========================================================
    # RESULTADOS
    # ========================================================

    df_folds = pd.DataFrame(
        resultados_folds
    )


    df_classes = pd.DataFrame(
        resultados_classes
    )


    # ========================================================
    # MÉDIA GERAL
    # ========================================================

    accuracy_media = (
        df_folds[
            "accuracy"
        ].mean()
    )


    accuracy_desvio = (
        df_folds[
            "accuracy"
        ].std()
    )


    balanced_media = (
        df_folds[
            "balanced_accuracy"
        ].mean()
    )


    balanced_desvio = (
        df_folds[
            "balanced_accuracy"
        ].std()
    )


    macro_media = (
        df_folds[
            "macro_f1"
        ].mean()
    )


    macro_desvio = (
        df_folds[
            "macro_f1"
        ].std()
    )


    weighted_media = (
        df_folds[
            "weighted_f1"
        ].mean()
    )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )


    print(
        "RESULTADO FINAL DA VALIDAÇÃO CRUZADA"
    )


    print(
        "=" * 90
    )


    print(
        "\nFolds:"
    )


    print(

        df_folds.round(
            4
        ).to_string(
            index=False
        )
    )


    print(
        "\nMÉDIAS:"
    )


    print(
        f"Accuracy: "
        f"{accuracy_media:.4f} "
        f"± "
        f"{accuracy_desvio:.4f}"
    )


    print(
        f"Balanced Accuracy: "
        f"{balanced_media:.4f} "
        f"± "
        f"{balanced_desvio:.4f}"
    )


    print(
        f"Macro F1: "
        f"{macro_media:.4f} "
        f"± "
        f"{macro_desvio:.4f}"
    )


    print(
        f"Weighted F1: "
        f"{weighted_media:.4f}"
    )


    # ========================================================
    # CLASSES
    # ========================================================

    resumo_classes = (

        df_classes

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
        "RESULTADO MÉDIO POR CLASSE"
    )


    print(
        "=" * 90
    )


    print(

        "\n"

        + resumo_classes.round(
            4
        ).to_string(
            index=False
        )
    )


    # ========================================================
    # COMPARAÇÃO COM CAMPEÃO ATUAL
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )


    print(
        "COMPARAÇÃO COM RF + VITERBI"
    )


    print(
        "=" * 90
    )


    print(
        "\nRF + Viterbi:"
    )


    print(
        f"  Accuracy média: "
        f"{RF_VITERBI_ACCURACY:.4f}"
    )


    print(
        f"  Macro F1 médio: "
        f"{RF_VITERBI_MACRO_F1:.4f}"
    )


    print(
        "\nBiGRU + Attention:"
    )


    print(
        f"  Accuracy média: "
        f"{accuracy_media:.4f}"
    )


    print(
        f"  Macro F1 médio: "
        f"{macro_media:.4f}"
    )


    print(
        "\nDiferença:"
    )


    print(
        f"  Accuracy: "
        f"{accuracy_media - RF_VITERBI_ACCURACY:+.4f}"
    )


    print(
        f"  Macro F1: "
        f"{macro_media - RF_VITERBI_MACRO_F1:+.4f}"
    )


    # ========================================================
    # SALVAR
    # ========================================================

    ARQUIVO_FOLDS = (
        PASTA_RESULTADOS
        / "validacao_cruzada_bigru_folds.csv"
    )


    ARQUIVO_CLASSES = (
        PASTA_RESULTADOS
        / "validacao_cruzada_bigru_classes.csv"
    )


    ARQUIVO_CLASSES_MEDIA = (
        PASTA_RESULTADOS
        / "validacao_cruzada_bigru_classes_media.csv"
    )


    df_folds.to_csv(
        ARQUIVO_FOLDS,
        index=False
    )


    df_classes.to_csv(
        ARQUIVO_CLASSES,
        index=False
    )


    resumo_classes.to_csv(
        ARQUIVO_CLASSES_MEDIA,
        index=False
    )


    print(
        "\n"
        + "=" * 90
    )


    print(
        "VALIDAÇÃO CRUZADA CONCLUÍDA"
    )


    print(
        "=" * 90
    )


    print(
        "\nArquivos:"
    )


    print(
        ARQUIVO_FOLDS
    )


    print(
        ARQUIVO_CLASSES
    )


    print(
        ARQUIVO_CLASSES_MEDIA
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