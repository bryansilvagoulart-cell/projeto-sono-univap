import pandas as pd
import numpy as np
from scipy.io import loadmat
from datetime import datetime, timedelta, timezone


# ==========================================================
# 1. CARREGAR HEART RATE
# ==========================================================

print("Carregando Heart Rate...")

hr = pd.read_csv("data/raw/amostra/hr.csv")

hr.columns = ["timestamp", "heart_rate"]

hr["datetime"] = pd.to_datetime(
    hr["timestamp"],
    unit="s",
    utc=True
)


# ==========================================================
# 2. CARREGAR MOVIMENTO
# ==========================================================

print("Carregando movimento...")

motion = pd.read_csv("data/raw/amostra/motion.csv")

motion.columns = ["timestamp", "x", "y", "z"]

motion["datetime"] = pd.to_datetime(
    motion["timestamp"],
    unit="s",
    utc=True
)


# ==========================================================
# 3. CARREGAR LABELS
# ==========================================================

print("Carregando labels...")

labels = loadmat("data/raw/amostra/labels.mat")

expert_labels = labels["expert_label"].flatten()

rec_start = datetime.strptime(
    labels["recStart"][0],
    "%Y-%m-%d %H:%M:%S"
)

rec_start = rec_start.replace(
    tzinfo=timezone(timedelta(hours=-5))
)

rec_start_utc = rec_start.astimezone(timezone.utc)


# ==========================================================
# 4. PREPARAR LISTA DO DATASET FINAL
# ==========================================================

resultados = []


# ==========================================================
# 5. PROCESSAR TODAS AS 935 ÉPOCAS
# ==========================================================

total_epocas = len(expert_labels)

print()
print("Total de épocas:", total_epocas)
print("Iniciando processamento...")
print()


for epoca in range(total_epocas):

    # ------------------------------------------------------
    # Definir intervalo da época
    # ------------------------------------------------------

    inicio = rec_start_utc + timedelta(
        seconds=epoca * 30
    )

    fim = inicio + timedelta(seconds=30)


    # ------------------------------------------------------
    # Selecionar Heart Rate da época
    # ------------------------------------------------------

    hr_epoca = hr[
        (hr["datetime"] >= inicio) &
        (hr["datetime"] < fim)
    ]


    # ------------------------------------------------------
    # Selecionar Movimento da época
    # ------------------------------------------------------

    motion_epoca = motion[
        (motion["datetime"] >= inicio) &
        (motion["datetime"] < fim)
    ].copy()


    # ------------------------------------------------------
    # Características da frequência cardíaca
    # ------------------------------------------------------

    if len(hr_epoca) > 0:

        hr_mean = hr_epoca["heart_rate"].mean()
        hr_min = hr_epoca["heart_rate"].min()
        hr_max = hr_epoca["heart_rate"].max()
        hr_std = hr_epoca["heart_rate"].std()

    else:

        hr_mean = np.nan
        hr_min = np.nan
        hr_max = np.nan
        hr_std = np.nan


    # ------------------------------------------------------
    # Características do movimento
    # ------------------------------------------------------

    if len(motion_epoca) > 0:

        # Magnitude da aceleração
        motion_epoca["magnitude"] = np.sqrt(
            motion_epoca["x"] ** 2 +
            motion_epoca["y"] ** 2 +
            motion_epoca["z"] ** 2
        )

        motion_mean = motion_epoca["magnitude"].mean()

        motion_std = motion_epoca["magnitude"].std()

        motion_max = motion_epoca["magnitude"].max()

        # Mudança entre uma medição e a seguinte
        motion_change = motion_epoca["magnitude"].diff().abs()

        motion_change_mean = motion_change.mean()

    else:

        motion_mean = np.nan
        motion_std = np.nan
        motion_max = np.nan
        motion_change_mean = np.nan


    # ------------------------------------------------------
    # Label de referência
    # ------------------------------------------------------

    label = int(expert_labels[epoca])


    # ------------------------------------------------------
    # Adicionar resultado
    # ------------------------------------------------------

    resultados.append({

        "epoca": epoca,

        "inicio": inicio,

        "fim": fim,

        "hr_mean": hr_mean,

        "hr_min": hr_min,

        "hr_max": hr_max,

        "hr_std": hr_std,

        "motion_mean": motion_mean,

        "motion_std": motion_std,

        "motion_max": motion_max,

        "motion_change_mean": motion_change_mean,

        "label": label

    })


    # ------------------------------------------------------
    # Mostrar progresso
    # ------------------------------------------------------

    if (epoca + 1) % 100 == 0:

        print(
            f"Processadas: {epoca + 1}/{total_epocas}"
        )


# ==========================================================
# 6. CRIAR DATAFRAME
# ==========================================================

dataset = pd.DataFrame(resultados)


# ==========================================================
# 7. SALVAR DATASET PROCESSADO
# ==========================================================

arquivo_saida = "data/processed/dataset_processado.csv"

dataset.to_csv(
    arquivo_saida,
    index=False
)


# ==========================================================
# 8. MOSTRAR RESULTADO
# ==========================================================

print()
print("=" * 60)
print("PROCESSAMENTO CONCLUÍDO!")
print("=" * 60)

print()

print("Quantidade de linhas:", len(dataset))

print("Quantidade de colunas:", len(dataset.columns))

print()

print("Colunas:")

for coluna in dataset.columns:
    print("-", coluna)

print()

print("Primeiras linhas:")

print(dataset.head())

print()

print("Quantidade de valores ausentes:")

print(dataset.isna().sum())

print()

print("Arquivo salvo em:")

print(arquivo_saida)