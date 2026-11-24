import pandas as pd
from datetime import datetime, timezone

# =========================
# FREQUÊNCIA CARDÍACA
# =========================

hr = pd.read_csv(
    "data/raw/amostra/hr.csv",
    header=None,
    names=["timestamp", "heart_rate"]
)

# =========================
# MOVIMENTO
# =========================

motion = pd.read_csv(
    "data/raw/amostra/motion.csv"
)

# =========================
# FUNÇÃO PARA CONVERTER
# TIMESTAMP UNIX
# =========================

def converter_timestamp(valor):
    return datetime.fromtimestamp(
        valor,
        tz=timezone.utc
    )


print("========== HEART RATE ==========")

print("Primeiro timestamp:")
print(hr["timestamp"].iloc[0])

print("Último timestamp:")
print(hr["timestamp"].iloc[-1])

print("Primeiro horário UTC:")
print(converter_timestamp(hr["timestamp"].iloc[0]))

print("Último horário UTC:")
print(converter_timestamp(hr["timestamp"].iloc[-1]))


print("\n========== MOTION ==========")

print("Primeiro timestamp:")
print(motion["Timestamp"].iloc[0])

print("Último timestamp:")
print(motion["Timestamp"].iloc[-1])

print("Primeiro horário UTC:")
print(converter_timestamp(motion["Timestamp"].iloc[0]))

print("Último horário UTC:")
print(converter_timestamp(motion["Timestamp"].iloc[-1]))