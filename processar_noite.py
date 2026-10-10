from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PARTICIPANTE = "Bidslab00"
NOITE = 1


# ============================================================
# ENCONTRAR A PASTA DA NOITE
# ============================================================

def encontrar_pasta(participante, noite):

    # Estrutura que usaremos futuramente
    pasta_noite = Path(
        f"data/raw/participantes/{participante}/noite_{noite}"
    )

    if pasta_noite.exists():
        return pasta_noite

    # Estrutura atual do Bidslab01
    pasta_participante = Path(
        f"data/raw/participantes/{participante}"
    )

    if noite == 1 and pasta_participante.exists():
        return pasta_participante

    # Nosso primeiro teste com o Bidslab00
    if participante == "Bidslab00" and noite == 1:

        pasta_amostra = Path("data/raw/amostra")

        if pasta_amostra.exists():
            return pasta_amostra

    raise FileNotFoundError(
        f"Não encontrei os arquivos de "
        f"{participante}, noite {noite}."
    )


# ============================================================
# PROCESSAR UMA NOITE
# ============================================================

def processar_noite(participante, noite):

    print("=" * 65)
    print("PROCESSAMENTO AUTOMÁTICO DE UMA NOITE")
    print("=" * 65)

    print(f"\nParticipante: {participante}")
    print(f"Noite: {noite}")

    # --------------------------------------------------------
    # 1. LOCALIZAR ARQUIVOS
    # --------------------------------------------------------

    pasta = encontrar_pasta(
        participante,
        noite
    )

    arquivo_hr = pasta / "hr.csv"
    arquivo_motion = pasta / "motion.csv"
    arquivo_labels = pasta / "labels.mat"

    print(f"\nPasta encontrada:")
    print(pasta)

    for arquivo in [
        arquivo_hr,
        arquivo_motion,
        arquivo_labels
    ]:

        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {arquivo}"
            )

    print("\nTodos os 3 arquivos foram encontrados.")

    # --------------------------------------------------------
    # 2. CARREGAR LABELS
    # --------------------------------------------------------

    print("\nCarregando labels...")

    labels = loadmat(arquivo_labels)

    expert_labels = (
        labels["expert_label"]
        .flatten()
        .astype(int)
    )

    rec_start_texto = labels["recStart"].item()

    print(
        f"Quantidade de épocas nos labels: "
        f"{len(expert_labels)}"
    )

    print(
        f"recStart original: "
        f"{rec_start_texto}"
    )

    # --------------------------------------------------------
    # 3. CONVERTER O HORÁRIO
    # --------------------------------------------------------

    rec_start_local = datetime.strptime(
        rec_start_texto,
        "%Y-%m-%d %H:%M:%S"
    )

    # O recStart está em Eastern Time.
    # ZoneInfo trata automaticamente UTC-4 / UTC-5.
    rec_start_local = rec_start_local.replace(
        tzinfo=ZoneInfo("America/New_York")
    )

    rec_start_utc = rec_start_local.astimezone(
        ZoneInfo("UTC")
    )

    rec_start_timestamp = rec_start_utc.timestamp()

    print(
        f"recStart UTC: "
        f"{rec_start_utc}"
    )

    # --------------------------------------------------------
    # 4. CARREGAR HEART RATE
    # --------------------------------------------------------

    print("\nCarregando frequência cardíaca...")

    hr = pd.read_csv(
        arquivo_hr,
        header=None,
        names=[
            "timestamp",
            "heart_rate"
        ]
    )

    print(
        f"Medições de HR: "
        f"{len(hr)}"
    )

    # Descobrir a época de 30 segundos
    # de cada medição.
    hr["epoca"] = np.floor(
        (
            hr["timestamp"]
            - rec_start_timestamp
        ) / 30
    ).astype(int)

    # Manter apenas as medições que
    # pertencem aos labels.
    hr = hr[
        (hr["epoca"] >= 0)
        &
        (hr["epoca"] < len(expert_labels))
    ].copy()

    print(
        f"Medições de HR dentro "
        f"do período rotulado: {len(hr)}"
    )

    # --------------------------------------------------------
    # 5. RESUMIR HR POR ÉPOCA
    # --------------------------------------------------------

    hr_epocas = (
        hr.groupby("epoca")
        .agg(
            hr_mean=(
                "heart_rate",
                "mean"
            ),
            hr_min=(
                "heart_rate",
                "min"
            ),
            hr_max=(
                "heart_rate",
                "max"
            ),
            hr_std=(
                "heart_rate",
                "std"
            ),
            hr_samples=(
                "heart_rate",
                "count"
            )
        )
        .reset_index()
    )

    # Se existe somente uma medição,
    # o Pandas não consegue calcular
    # o desvio padrão.
    hr_epocas["hr_std"] = (
        hr_epocas["hr_std"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # 6. CARREGAR MOVIMENTO
    # --------------------------------------------------------

    print("\nCarregando acelerômetro...")

    motion = pd.read_csv(
        arquivo_motion
    )

    print(
        f"Medições de movimento: "
        f"{len(motion)}"
    )

    # --------------------------------------------------------
    # 7. CALCULAR MOVIMENTO
    # --------------------------------------------------------

    motion["magnitude"] = np.sqrt(
        motion["x"] ** 2
        + motion["y"] ** 2
        + motion["z"] ** 2
    )

    # Quanto a magnitude mudou
    # entre uma medição e outra.
    motion["movimento"] = (
        motion["magnitude"]
        .diff()
        .abs()
    )

    # Identificar época de 30 segundos.
    motion["epoca"] = np.floor(
        (
            motion["Timestamp"]
            - rec_start_timestamp
        ) / 30
    ).astype(int)

    # Manter somente o período
    # correspondente aos labels.
    motion = motion[
        (motion["epoca"] >= 0)
        &
        (
            motion["epoca"]
            < len(expert_labels)
        )
    ].copy()

    print(
        f"Medições de movimento dentro "
        f"do período rotulado: {len(motion)}"
    )

    # --------------------------------------------------------
    # 8. RESUMIR MOVIMENTO POR ÉPOCA
    # --------------------------------------------------------

    motion_epocas = (
        motion.groupby("epoca")
        .agg(
            motion_mean=(
                "movimento",
                "mean"
            ),
            motion_std=(
                "movimento",
                "std"
            ),
            motion_max=(
                "movimento",
                "max"
            ),
            motion_samples=(
                "movimento",
                "count"
            )
        )
        .reset_index()
    )

    motion_epocas["motion_std"] = (
        motion_epocas["motion_std"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # 9. CRIAR AS ÉPOCAS DA NOITE
    # --------------------------------------------------------

    dataset = pd.DataFrame({
        "epoca": range(
            len(expert_labels)
        )
    })

    dataset["participante"] = participante
    dataset["noite"] = noite

    dataset["inicio"] = [
        rec_start_utc
        + timedelta(seconds=i * 30)

        for i in dataset["epoca"]
    ]

    dataset["fim"] = (
        dataset["inicio"]
        + pd.Timedelta(seconds=30)
    )

    dataset["label"] = expert_labels

    # --------------------------------------------------------
    # 10. JUNTAR HR + MOVIMENTO
    # --------------------------------------------------------

    dataset = dataset.merge(
        hr_epocas,
        on="epoca",
        how="left"
    )

    dataset = dataset.merge(
        motion_epocas,
        on="epoca",
        how="left"
    )

    print("\nDados sincronizados.")

    # --------------------------------------------------------
    # 11. REMOVER LABEL UNKNOWN
    # --------------------------------------------------------

    unknown = (
        dataset["label"] == 5
    ).sum()

    if unknown > 0:

        print(
            f"\nÉpocas Unknown encontradas: "
            f"{unknown}"
        )

        dataset = dataset[
            dataset["label"] != 5
        ].copy()

    # --------------------------------------------------------
    # 12. REMOVER ÉPOCAS SEM SINAIS
    # --------------------------------------------------------

    antes = len(dataset)

    dataset = dataset.dropna(
        subset=[
            "hr_mean",
            "motion_mean"
        ]
    ).copy()

    depois = len(dataset)

    removidas = antes - depois

    print(
        f"\nÉpocas removidas por falta "
        f"de sinal: {removidas}"
    )

    # --------------------------------------------------------
    # 13. VERIFICAR VALORES AUSENTES
    # --------------------------------------------------------

    print("\nValores ausentes:")

    print(
        dataset.isna().sum()
    )

    # --------------------------------------------------------
    # 14. SALVAR
    # --------------------------------------------------------

    pasta_saida = Path(
        f"data/processed/participantes/"
        f"{participante}"
    )

    pasta_saida.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_saida = (
        pasta_saida
        / f"noite_{noite}.csv"
    )

    dataset.to_csv(
        arquivo_saida,
        index=False
    )

    # --------------------------------------------------------
    # 15. RESUMO
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 65)

    print(
        f"\nParticipante: "
        f"{participante}"
    )

    print(
        f"Noite: "
        f"{noite}"
    )

    print(
        f"Épocas finais: "
        f"{len(dataset)}"
    )

    print(
        f"Quantidade de colunas: "
        f"{len(dataset.columns)}"
    )

    print("\nDistribuição dos labels:")

    print(
        dataset["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nArquivo criado:\n"
        f"{arquivo_saida}"
    )

    return dataset


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    processar_noite(
        PARTICIPANTE,
        NOITE
    )