from pathlib import Path
import hashlib
import json
import random
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURAÇÃO OFICIAL CONGELADA
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
# PESOS FINAIS CONGELADOS PELA VALIDAÇÃO CRUZADA
# ============================================================

PESO_RF = 0.75

PESO_BIGRU = 0.25

PESO_VITERBI = 0.80


# ============================================================
# ARQUIVOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_4_classes.csv"
)

ARQUIVO_RANKING_CV = Path(
    "data/processed/resultados_ml_4_classes/"
    "cv_4_classes_ranking.csv"
)

PASTA_MODELOS = Path(
    "models/4_classes"
)

PASTA_RESULTADOS = Path(
    "data/processed/resultados_ml_4_classes_final"
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

ARQUIVO_MANIFESTO = (
    PASTA_MODELOS
    / "manifesto_modelo_final_4_classes.json"
)

ARQUIVO_LOSS = (
    PASTA_RESULTADOS
    / "treino_bigru_final_loss.csv"
)


# ============================================================
# 30 FEATURES OFICIAIS
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
# REPRODUTIBILIDADE
# ============================================================

def definir_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


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
# VALIDAR CONFIGURAÇÃO CONGELADA
# ============================================================

def validar_ranking_cv():

    if not ARQUIVO_RANKING_CV.exists():

        raise FileNotFoundError(
            "Ranking da validação cruzada não encontrado:\n"
            f"{ARQUIVO_RANKING_CV}"
        )


    ranking = pd.read_csv(
        ARQUIVO_RANKING_CV
    )


    if len(ranking) == 0:

        raise RuntimeError(
            "O ranking da validação cruzada está vazio."
        )


    melhor = ranking.iloc[
        0
    ]


    valores = {
        "peso_rf":
            PESO_RF,

        "peso_bigru":
            PESO_BIGRU,

        "peso_viterbi":
            PESO_VITERBI
    }


    for coluna, esperado in valores.items():

        encontrado = float(
            melhor[
                coluna
            ]
        )


        if not np.isclose(
            encontrado,
            esperado
        ):

            raise RuntimeError(
                "A configuração congelada do script "
                "não coincide com o primeiro lugar "
                "da validação cruzada.\n"
                f"{coluna}: "
                f"esperado={esperado} | "
                f"ranking={encontrado}"
            )


    print(
        "✅ Pesos finais conferidos com "
        "o ranking da validação cruzada."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    definir_seed(
        SEED
    )


    print(
        "=" * 100
    )

    print(
        "TREINAMENTO FINAL - MODELO OFICIAL DE 4 CLASSES"
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


    print(
        "\nCONFIGURAÇÃO CONGELADA:"
    )


    print(
        f"  Seed: "
        f"{SEED}"
    )


    print(
        f"  Random Forest: "
        f"{PESO_RF:.2f}"
    )


    print(
        f"  BiGRU: "
        f"{PESO_BIGRU:.2f}"
    )


    print(
        f"  Viterbi: "
        f"{PESO_VITERBI:.2f}"
    )


    print(
        f"  Janela BiGRU: "
        f"{TAMANHO_JANELA} épocas "
        f"({TAMANHO_JANELA * 0.5:.1f} min)"
    )


    print(
        f"  Épocas de treino BiGRU: "
        f"{EPOCAS_BIGRU}"
    )


    # ========================================================
    # PASTAS
    # ========================================================

    PASTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True
    )


    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CONFERIR RANKING
    # ========================================================

    print(
        "\nConferindo configuração "
        "com a validação cruzada..."
    )


    validar_ranking_cv()


    # ========================================================
    # CARREGAR DATASET DE DESENVOLVIMENTO
    # ========================================================

    print(
        "\nCarregando dataset de desenvolvimento..."
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
            "Esperávamos exatamente "
            "60165 épocas de desenvolvimento."
        )


    if (
        dataset[
            "participante"
        ].nunique()
        != 15
    ):

        raise ValueError(
            "Esperávamos exatamente "
            "15 participantes."
        )


    if (
        dataset[
            "id_noite"
        ].nunique()
        != 75
    ):

        raise ValueError(
            "Esperávamos exatamente "
            "75 noites."
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
            "Existem valores ausentes "
            "nas 30 features."
        )


    valores_features = (
        dataset[
            FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )


    if not np.isfinite(
        valores_features
    ).all():

        raise ValueError(
            "Existem valores infinitos "
            "ou inválidos nas features."
        )


    # ========================================================
    # LABELS
    # ========================================================

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


    print(
        "\nDistribuição no desenvolvimento:"
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
    # ÍNDICES DAS JANELAS
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

        random_state=SEED,

        n_jobs=-1
    )


    rf.fit(
        dataset[
            FEATURES
        ],
        labels
    )


    print(
        "✅ Random Forest final treinado."
    )


    # ========================================================
    # NORMALIZAÇÃO DA BiGRU
    # ========================================================

    print(
        "\nAjustando StandardScaler "
        "somente no desenvolvimento..."
    )


    scaler = StandardScaler()


    scaler.fit(
        dataset[
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


    print(
        "✅ Scaler ajustado."
    )


    # ========================================================
    # DATASET PYTORCH
    # ========================================================

    indices_treino = np.arange(
        len(dataset),
        dtype=np.int64
    )


    treino_torch = DatasetSequencial(

        indices_treino,

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


    # ========================================================
    # PESOS DE CLASSE
    # ========================================================

    contagens = np.bincount(
        labels,
        minlength=len(
            CLASSES
        )
    )


    pesos_classes = (

        len(labels)

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


    historico_loss = []


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


        loss_media = (
            soma_loss
            / quantidade
        )


        historico_loss.append(
            {
                "epoca":
                    epoca,

                "loss":
                    loss_media
            }
        )


        print(
            f"Época "
            f"{epoca:02d}/"
            f"{EPOCAS_BIGRU}"
            f" | Loss "
            f"{loss_media:.4f}"
        )


    print(
        "✅ BiGRU final treinada."
    )


    # ========================================================
    # TRANSIÇÕES E PRIOR
    # ========================================================

    print(
        "\nCalculando matriz de transição "
        "e probabilidade inicial..."
    )


    matriz_transicao = (
        calcular_transicoes(
            dataset
        )
    )


    prob_inicial = (
        calcular_probabilidade_inicial(
            dataset
        )
    )


    print(
        "✅ Transições calculadas "
        "somente com o desenvolvimento."
    )


    # ========================================================
    # SALVAR ARTEFATOS
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SALVANDO ARTEFATOS FINAIS"
    )

    print(
        "=" * 100
    )


    joblib.dump(
        rf,
        ARQUIVO_RF
    )


    joblib.dump(
        scaler,
        ARQUIVO_SCALER
    )


    torch.save(

        {
            "state_dict":
                modelo.state_dict(),

            "features":
                FEATURES,

            "classes":
                CLASSES,

            "nomes_classes":
                NOMES_CLASSES,

            "hidden_size":
                HIDDEN_SIZE,

            "raio_janela":
                RAIO_JANELA,

            "tamanho_janela":
                TAMANHO_JANELA,

            "epocas_treinamento":
                EPOCAS_BIGRU,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "seed":
                SEED,

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
                FEATURES,

            "classes":
                CLASSES,

            "nomes_classes":
                NOMES_CLASSES,

            "seed":
                SEED
        },

        ARQUIVO_ENSEMBLE
    )


    pd.DataFrame(
        historico_loss
    ).to_csv(
        ARQUIVO_LOSS,
        index=False
    )


    # ========================================================
    # MANIFESTO DE REPRODUTIBILIDADE
    # ========================================================

    manifesto = {

        "modelo":
            "Ensemble Random Forest + BiGRU + Viterbi",

        "versao":
            "4_classes_final_1.0",

        "criado_em_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "dataset_desenvolvimento":
            str(
                ARQUIVO_DATASET
            ),

        "dataset_sha256":
            calcular_sha256(
                ARQUIVO_DATASET
            ),

        "ranking_cv":
            str(
                ARQUIVO_RANKING_CV
            ),

        "ranking_cv_sha256":
            calcular_sha256(
                ARQUIVO_RANKING_CV
            ),

        "quantidade_epocas_desenvolvimento":
            int(
                len(dataset)
            ),

        "quantidade_participantes":
            int(
                dataset[
                    "participante"
                ].nunique()
            ),

        "quantidade_noites":
            int(
                dataset[
                    "id_noite"
                ].nunique()
            ),

        "classes":
            NOMES_CLASSES,

        "features":
            FEATURES,

        "configuracao": {

            "seed":
                SEED,

            "random_forest": {
                "n_estimators":
                    300,

                "max_depth":
                    16,

                "min_samples_split":
                    10,

                "min_samples_leaf":
                    5,

                "max_features":
                    "sqrt",

                "class_weight":
                    "balanced_subsample",

                "random_state":
                    SEED
            },

            "bigru": {
                "janela_epocas":
                    TAMANHO_JANELA,

                "raio_janela":
                    RAIO_JANELA,

                "hidden_size":
                    HIDDEN_SIZE,

                "epocas_treinamento":
                    EPOCAS_BIGRU,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    0.0001,

                "label_smoothing":
                    0.05,

                "dropout":
                    0.30
            },

            "ensemble": {
                "peso_rf":
                    PESO_RF,

                "peso_bigru":
                    PESO_BIGRU,

                "peso_viterbi":
                    PESO_VITERBI
            }
        },

        "arquivos": {
            "random_forest":
                str(
                    ARQUIVO_RF
                ),

            "scaler":
                str(
                    ARQUIVO_SCALER
                ),

            "bigru":
                str(
                    ARQUIVO_BIGRU
                ),

            "ensemble":
                str(
                    ARQUIVO_ENSEMBLE
                ),

            "historico_loss":
                str(
                    ARQUIVO_LOSS
                )
        },

        "observacao":
            (
                "Treinamento final realizado somente "
                "nos 15 participantes de desenvolvimento. "
                "O conjunto independente Bidslab01, "
                "Bidslab35 e Bidslab47 não é carregado "
                "nem avaliado por este script."
            )
    }


    with open(
        ARQUIVO_MANIFESTO,
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
    # VERIFICAR ARQUIVOS
    # ========================================================

    arquivos_esperados = [
        ARQUIVO_RF,
        ARQUIVO_SCALER,
        ARQUIVO_BIGRU,
        ARQUIVO_ENSEMBLE,
        ARQUIVO_MANIFESTO,
        ARQUIVO_LOSS
    ]


    faltando_saida = [
        str(
            arquivo
        )
        for arquivo in arquivos_esperados
        if not arquivo.exists()
    ]


    if faltando_saida:

        raise RuntimeError(
            "Alguns artefatos finais "
            "não foram salvos:\n"
            + "\n".join(
                faltando_saida
            )
        )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TREINAMENTO FINAL CONCLUÍDO"
    )

    print(
        "=" * 100
    )


    print(
        "\nArtefatos salvos:"
    )


    for arquivo in arquivos_esperados:

        print(
            f"  ✅ {arquivo}"
        )


    print(
        "\nIMPORTANTE:"
    )


    print(
        "✅ O modelo foi treinado com "
        "os 15 participantes de desenvolvimento."
    )


    print(
        "✅ Nenhuma métrica do teste "
        "independente foi calculada."
    )


    print(
        "✅ Bidslab01, Bidslab35 e Bidslab47 "
        "não foram carregados por este script."
    )


    print(
        "\nPróxima etapa:"
    )


    print(
        "Avaliar UMA ÚNICA VEZ o modelo congelado "
        "nas 12.252 épocas do teste independente."
    )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()
