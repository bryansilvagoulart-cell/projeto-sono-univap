import scipy.io
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ==========================
# 1. CARREGAR LABELS
# ==========================

labels = scipy.io.loadmat(
    "data/raw/amostra/labels.mat"
)

expert = labels["expert_label"].flatten()

rec_start_text = labels["recStart"].item()

print("recStart original:")
print(rec_start_text)

# ==========================
# 2. CONVERTER recStart
# ==========================

# O recStart do dataset está em Eastern Time.
rec_start = datetime.strptime(
    rec_start_text,
    "%Y-%m-%d %H:%M:%S"
).replace(
    tzinfo=ZoneInfo("America/New_York")
)

print("\nrecStart com fuso:")
print(rec_start)

print("\nrecStart em UTC:")
print(rec_start.astimezone(ZoneInfo("UTC")))

# ==========================
# 3. CRIAR TEMPOS DAS ÉPOCAS
# ==========================

print("\nPrimeiras épocas:")

for i in range(10):

    inicio = rec_start + timedelta(
        seconds=i * 30
    )

    fim = inicio + timedelta(seconds=30)

    print(
        f"Época {i}: "
        f"{inicio.astimezone(ZoneInfo('UTC'))} "
        f"→ "
        f"{fim.astimezone(ZoneInfo('UTC'))} "
        f"| Label: {expert[i]}"
    )

# ==========================
# 4. ÚLTIMA ÉPOCA
# ==========================

ultima = len(expert) - 1

inicio = rec_start + timedelta(
    seconds=ultima * 30
)

fim = inicio + timedelta(seconds=30)

print("\nÚltima época:")

print(
    f"Época {ultima}: "
    f"{inicio.astimezone(ZoneInfo('UTC'))} "
    f"→ "
    f"{fim.astimezone(ZoneInfo('UTC'))} "
    f"| Label: {expert[ultima]}"
)