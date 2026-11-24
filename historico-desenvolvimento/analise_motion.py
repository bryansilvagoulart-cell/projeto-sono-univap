import pandas as pd

arquivo = "data/raw/amostra/motion.csv"

# O motion.csv já possui cabeçalho
dados = pd.read_csv(
    arquivo,
    nrows=10000
)

print("Colunas:")
print(dados.columns.tolist())

print("\nPrimeiras linhas:")
print(dados.head())

print("\nTipos dos dados:")
print(dados.dtypes)

print("\nQuantidade de linhas:")
print(len(dados))

# Converte as colunas para números
dados["Timestamp"] = pd.to_numeric(dados["Timestamp"])
dados["x"] = pd.to_numeric(dados["x"])
dados["y"] = pd.to_numeric(dados["y"])
dados["z"] = pd.to_numeric(dados["z"])

# Calcula o intervalo entre medições
dados["intervalo"] = dados["Timestamp"].diff()

print("\nIntervalo entre as medições:")
print(dados["intervalo"].describe())

print("\nPrimeiros intervalos:")
print(dados["intervalo"].head(10))