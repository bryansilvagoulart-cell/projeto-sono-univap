from pathlib import Path
from copy import deepcopy
import random

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
# 30 FEATURES ESCOLHIDAS
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
# CONFIGURAÇÃO DA SEQUÊNCIA
# ============================================================
#
# 10 épocas anteriores
# +
# época atual
# +
# 10 épocas posteriores
#
# 21 épocas × 30 segundos
# = 10,5 minutos
#
# ============================================================

RAIO_JANELA = 10

TAMANHO_JANELA = (
    RAIO_JANELA * 2
    + 1
)


# ============================================================
# CONFIGURAÇÃO DA REDE
# ============================================================

HIDDEN_SIZE = 64

BATCH_SIZE = 256

MAX_EPOCHS = 25

PATIENCE = 5

LEARNING_RATE = 0.001


# ============================================================
# CONSTRUIR JANELAS SEM ATRAVESSAR GAPS OU NOITES
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


            quantidade_esquerda = (
                posicao_local
                - inicio
            )


            destino_inicio = (
                raio
                - quantidade_esquerda
            )


            destino_fim = (
                destino_inicio
                + (fim - inicio)
            )


            indice_original = posicoes[
                posicao_local
            ]


            indices_janelas[
                indice_original,
                destino_inicio:destino_fim
            ] = posicoes[
                inicio:fim
            ]


    return indices_janelas


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

        self.indices_janelas = (
            indices_janelas
        )


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


        janela_indices = (
            self.indices_janelas[
                linha
            ]
        )


        validos = (
            janela_indices
            >= 0
        )


        indices_seguros = (
            janela_indices.copy()
        )


        indices_seguros[
            ~validos
        ] = 0


        sequencia = (
            self.X_escalado[
                indices_seguros
            ]
            .copy()
        )


        # --------------------------------------------
        # Padding vira zero.
        #
        # Como os dados foram normalizados,
        # zero representa aproximadamente a média.
        # --------------------------------------------

        sequencia[
            ~validos
        ] = 0.0


        # --------------------------------------------
        # Máscara:
        #
        # 1 = época real
        # 0 = padding
        #
        # A máscara também entra como feature.
        # --------------------------------------------

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
# REDE BiGRU + ATTENTION
# ============================================================

class BiGRUAttention(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes
    ):

        super().__init__()


        # ----------------------------------------------------
        # GRU bidirecional
        # ----------------------------------------------------

        self.gru = nn.GRU(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=1,

            batch_first=True,

            bidirectional=True
        )


        tamanho_saida = (
            hidden_size
            * 2
        )


        # ----------------------------------------------------
        # ATTENTION
        #
        # Aprende quais épocas da janela são
        # mais importantes para a classificação.
        # ----------------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                tamanho_saida,
                64
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                1
            )
        )


        # ----------------------------------------------------
        # CLASSIFICADOR FINAL
        # ----------------------------------------------------

        self.classificador = nn.Sequential(

            nn.Linear(
                tamanho_saida,
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


        # A última feature é nossa máscara.
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


        # Padding não pode receber atenção.
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


        logits = self.classificador(
            contexto
        )


        return logits


# ============================================================
# FUNÇÃO DE AVALIAÇÃO
# ============================================================

def avaliar_modelo(
    modelo,
    loader
):

    modelo.eval()


    labels_reais = []

    predicoes = []


    perda_total = 0.0

    quantidade = 0


    with torch.no_grad():


        for X_batch, y_batch in loader:


            X_batch = X_batch.to(
                DEVICE
            )


            y_batch = y_batch.to(
                DEVICE
            )


            logits = modelo(
                X_batch
            )


            pred = torch.argmax(
                logits,
                dim=1
            )


            labels_reais.extend(

                y_batch
                .cpu()
                .numpy()
                .tolist()
            )


            predicoes.extend(

                pred
                .cpu()
                .numpy()
                .tolist()
            )


            quantidade += len(
                y_batch
            )


    labels_reais = np.asarray(
        labels_reais
    )


    predicoes = np.asarray(
        predicoes
    )


    accuracy = accuracy_score(
        labels_reais,
        predicoes
    )


    balanced = balanced_accuracy_score(
        labels_reais,
        predicoes
    )


    macro_f1 = f1_score(
        labels_reais,
        predicoes,
        average="macro"
    )


    weighted_f1 = f1_score(
        labels_reais,
        predicoes,
        average="weighted"
    )


    return {

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "y_true":
            labels_reais,

        "y_pred":
            predicoes
    }


# ============================================================
# INÍCIO
# ============================================================

print("=" * 90)
print("PROTÓTIPO NEURAL - BiGRU + ATTENTION")
print("=" * 90)


print(
    f"\nPyTorch: "
    f"{torch.__version__}"
)


print(
    f"Dispositivo: "
    f"{DEVICE}"
)


print(
    f"Features originais: "
    f"{len(FEATURES)}"
)


print(
    f"Tamanho da janela: "
    f"{TAMANHO_JANELA} épocas"
)


print(
    f"Contexto aproximado: "
    f"{TAMANHO_JANELA * 0.5:.1f} minutos"
)


# ============================================================
# 1. CARREGAR DADOS
# ============================================================

print(
    "\nCarregando dataset..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
).reset_index(
    drop=True
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


print(
    f"Épocas totais: "
    f"{len(dataset)}"
)


# ============================================================
# 2. CARREGAR DIVISÃO
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


print(
    "\nGrupos encontrados na divisão:"
)


print(
    divisao[
        "grupo"
    ]
    .value_counts()
    .to_string()
)


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


participantes_teste = (

    divisao[
        divisao[
            "grupo"
        ] == "teste"
    ]

    ["participante"]

    .tolist()
)


if len(participantes_treino) == 0:

    raise ValueError(
        "Nenhum participante de treino encontrado."
    )


if len(participantes_validacao) == 0:

    raise ValueError(
        "Nenhum participante de validação encontrado."
    )


if len(participantes_teste) == 0:

    raise ValueError(
        "Nenhum participante de teste encontrado."
    )


print(
    "\nTreino:"
)


for participante in participantes_treino:

    print(
        f"  - {participante}"
    )


print(
    "\nValidação:"
)


for participante in participantes_validacao:

    print(
        f"  - {participante}"
    )


print(
    "\nTESTE FINAL - CONTINUA FECHADO:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# 3. VALIDAR FEATURES
# ============================================================

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


if dataset[
    FEATURES
].isna().sum().sum() != 0:

    raise ValueError(
        "Existem valores ausentes "
        "nas features."
    )


print(
    "\n✅ Features validadas."
)


# ============================================================
# 4. ÍNDICES DE TREINO / VALIDAÇÃO / TESTE
# ============================================================

indices_treino = dataset.index[
    dataset[
        "participante"
    ].isin(
        participantes_treino
    )
].to_numpy()


indices_validacao = dataset.index[
    dataset[
        "participante"
    ].isin(
        participantes_validacao
    )
].to_numpy()


indices_teste = dataset.index[
    dataset[
        "participante"
    ].isin(
        participantes_teste
    )
].to_numpy()


print(
    f"\nÉpocas treino: "
    f"{len(indices_treino)}"
)


print(
    f"Épocas validação: "
    f"{len(indices_validacao)}"
)


print(
    f"Épocas teste reservadas: "
    f"{len(indices_teste)}"
)


if len(indices_teste) != 12061:

    raise ValueError(
        "O teste deveria possuir "
        "12061 épocas."
    )


# ============================================================
# 5. RANDOM FOREST DE REFERÊNCIA
# ============================================================
#
# Antes da rede, treinamos o RF usando exatamente
# estes mesmos participantes.
#
# Assim teremos uma comparação justa.
#
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "RANDOM FOREST DE REFERÊNCIA"
)

print(
    "=" * 90
)


X_rf_treino = dataset.loc[
    indices_treino,
    FEATURES
]


y_rf_treino = dataset.loc[
    indices_treino,
    "label"
]


X_rf_validacao = dataset.loc[
    indices_validacao,
    FEATURES
]


y_rf_validacao = dataset.loc[
    indices_validacao,
    "label"
]


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


print(
    "\nTreinando Random Forest..."
)


rf.fit(
    X_rf_treino,
    y_rf_treino
)


pred_rf = rf.predict(
    X_rf_validacao
)


rf_accuracy = accuracy_score(
    y_rf_validacao,
    pred_rf
)


rf_balanced = balanced_accuracy_score(
    y_rf_validacao,
    pred_rf
)


rf_macro = f1_score(
    y_rf_validacao,
    pred_rf,
    average="macro"
)


rf_weighted = f1_score(
    y_rf_validacao,
    pred_rf,
    average="weighted"
)


print(
    f"\nRF Accuracy: "
    f"{rf_accuracy:.4f}"
)


print(
    f"RF Balanced Accuracy: "
    f"{rf_balanced:.4f}"
)


print(
    f"RF Macro F1: "
    f"{rf_macro:.4f}"
)


print(
    f"RF Weighted F1: "
    f"{rf_weighted:.4f}"
)


# ============================================================
# 6. NORMALIZAÇÃO
# ============================================================
#
# MUITO IMPORTANTE:
#
# O scaler é treinado APENAS nos participantes de treino.
#
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PREPARANDO DADOS PARA A REDE"
)

print(
    "=" * 90
)


scaler = StandardScaler()


scaler.fit(
    dataset.loc[
        indices_treino,
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


labels = dataset[
    "label"
].astype(
    int
).to_numpy()


# ============================================================
# 7. CONSTRUIR JANELAS
# ============================================================

print(
    "\nConstruindo janelas temporais..."
)


indices_janelas = construir_indices_janelas(

    dataset,

    RAIO_JANELA
)


print(
    "✅ Janelas construídas."
)


# ============================================================
# 8. DATASETS PYTORCH
# ============================================================

dataset_treino_torch = DatasetSequencial(

    indices_treino,

    X_escalado,

    labels,

    indices_janelas
)


dataset_validacao_torch = DatasetSequencial(

    indices_validacao,

    X_escalado,

    labels,

    indices_janelas
)


loader_treino = DataLoader(

    dataset_treino_torch,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


loader_validacao = DataLoader(

    dataset_validacao_torch,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# 9. PESOS DAS CLASSES
# ============================================================

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
        f"| peso = "
        f"{pesos_classes[classe]:.4f}"
    )


pesos_tensor = torch.tensor(

    pesos_classes,

    dtype=torch.float32,

    device=DEVICE
)


# ============================================================
# 10. CRIAR REDE
# ============================================================

input_size = (
    len(FEATURES)
    + 1
)


modelo = BiGRUAttention(

    input_size=input_size,

    hidden_size=HIDDEN_SIZE,

    num_classes=len(
        CLASSES
    )
).to(
    DEVICE
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


quantidade_parametros = sum(

    parametro.numel()

    for parametro in modelo.parameters()

    if parametro.requires_grad
)


print(
    f"\nParâmetros treináveis: "
    f"{quantidade_parametros:,}"
)


# ============================================================
# 11. LOSS / OTIMIZADOR
# ============================================================

criterio = nn.CrossEntropyLoss(

    weight=pesos_tensor,

    label_smoothing=0.05
)


otimizador = torch.optim.AdamW(

    modelo.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        otimizador,

        mode="max",

        factor=0.5,

        patience=2
    )
)


# ============================================================
# 12. TREINAMENTO
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "TREINAMENTO DA BiGRU"
)

print(
    "=" * 90
)


melhor_macro = -1.0

melhor_estado = None

melhor_epoca = 0

epocas_sem_melhora = 0


historico = []


for epoca_treino in range(
    1,
    MAX_EPOCHS + 1
):


    modelo.train()


    perda_acumulada = 0.0

    quantidade_amostras = 0


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


        perda_acumulada += (

            loss.item()

            * len(
                y_batch
            )
        )


        quantidade_amostras += len(
            y_batch
        )


    perda_media = (

        perda_acumulada

        / quantidade_amostras
    )


    metricas_val = avaliar_modelo(

        modelo,

        loader_validacao
    )


    macro_val = metricas_val[
        "macro_f1"
    ]


    scheduler.step(
        macro_val
    )


    lr_atual = (
        otimizador
        .param_groups[
            0
        ][
            "lr"
        ]
    )


    print(
        f"\nÉpoca "
        f"{epoca_treino:02d}"
        f"/{MAX_EPOCHS}"
    )


    print(
        f"  Loss treino: "
        f"{perda_media:.4f}"
    )


    print(
        f"  Val Accuracy: "
        f"{metricas_val['accuracy']:.4f}"
    )


    print(
        f"  Val Balanced Acc: "
        f"{metricas_val['balanced_accuracy']:.4f}"
    )


    print(
        f"  Val Macro F1: "
        f"{macro_val:.4f}"
    )


    print(
        f"  Learning rate: "
        f"{lr_atual:.6f}"
    )


    historico.append({

        "epoca":
            epoca_treino,

        "loss_treino":
            perda_media,

        "accuracy_validacao":
            metricas_val[
                "accuracy"
            ],

        "balanced_accuracy_validacao":
            metricas_val[
                "balanced_accuracy"
            ],

        "macro_f1_validacao":
            macro_val,

        "weighted_f1_validacao":
            metricas_val[
                "weighted_f1"
            ],

        "learning_rate":
            lr_atual
    })


    # ========================================================
    # MELHOR MODELO
    # ========================================================

    if macro_val > melhor_macro:

        melhor_macro = macro_val

        melhor_epoca = epoca_treino

        melhor_estado = deepcopy(
            modelo.state_dict()
        )

        epocas_sem_melhora = 0


        print(
            "  ⭐ Novo melhor modelo."
        )


    else:

        epocas_sem_melhora += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epocas_sem_melhora >= PATIENCE:

        print(
            "\nEarly stopping ativado."
        )

        break


# ============================================================
# 13. RESTAURAR MELHOR MODELO
# ============================================================

if melhor_estado is None:

    raise RuntimeError(
        "Nenhum estado de modelo foi salvo."
    )


modelo.load_state_dict(
    melhor_estado
)


# ============================================================
# 14. AVALIAÇÃO FINAL NA VALIDAÇÃO
# ============================================================

resultado = avaliar_modelo(

    modelo,

    loader_validacao
)


print(
    "\n"
    + "=" * 90
)

print(
    "RESULTADO DO MELHOR MODELO NEURAL"
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


# ============================================================
# 15. RESULTADO POR CLASSE
# ============================================================

precisao, recall, f1_classes, suporte = (
    precision_recall_fscore_support(

        resultado[
            "y_true"
        ],

        resultado[
            "y_pred"
        ],

        labels=CLASSES,

        zero_division=0
    )
)


resultado_classes = []


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


    print(
        f"  Support: "
        f"{suporte[i]}"
    )


    resultado_classes.append({

        "classe":
            classe,

        "nome":
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
# 16. MATRIZ DE CONFUSÃO
# ============================================================

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


# ============================================================
# 17. COMPARAÇÃO COM RANDOM FOREST
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "COMPARAÇÃO DIRETA"
)

print(
    "=" * 90
)


print(
    "\nRandom Forest:"
)


print(
    f"  Accuracy: "
    f"{rf_accuracy:.4f}"
)


print(
    f"  Macro F1: "
    f"{rf_macro:.4f}"
)


print(
    "\nBiGRU + Attention:"
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
    "\nDiferença:"
)


print(
    f"  Accuracy: "
    f"{resultado['accuracy'] - rf_accuracy:+.4f}"
)


print(
    f"  Macro F1: "
    f"{resultado['macro_f1'] - rf_macro:+.4f}"
)


# ============================================================
# 18. SALVAR MODELO
# ============================================================

ARQUIVO_MODELO = (
    PASTA_MODELOS
    / "bigru_atencao_prototipo.pt"
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

        "melhor_epoca":
            melhor_epoca,

        "macro_f1_validacao":
            resultado[
                "macro_f1"
            ],

        "accuracy_validacao":
            resultado[
                "accuracy"
            ],

        "media_scaler":
            scaler.mean_,

        "escala_scaler":
            scaler.scale_
    },

    ARQUIVO_MODELO
)


# ============================================================
# 19. SALVAR RESULTADOS
# ============================================================

pd.DataFrame(
    historico
).to_csv(

    PASTA_RESULTADOS
    / "bigru_atencao_historico.csv",

    index=False
)


pd.DataFrame(
    resultado_classes
).to_csv(

    PASTA_RESULTADOS
    / "bigru_atencao_classes.csv",

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
    / "bigru_atencao_matriz_confusao.csv"
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PROTÓTIPO NEURAL CONCLUÍDO"
)

print(
    "=" * 90
)


print(
    f"\nModelo salvo em:"
)


print(
    ARQUIVO_MODELO
)


print(
    "\n⚠️ Bidslab16, Bidslab22 e Bidslab32 "
    "NÃO foram utilizados."
)


print(
    "\n✅ TESTE FINAL CONTINUA FECHADO"
)