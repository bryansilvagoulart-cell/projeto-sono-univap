import pandas as pd

# Caminho do arquivo de frequência cardíaca
arquivo = "data/raw/amostra/hr.csv"

# Carrega o CSV
dados = pd.read_csv(
    arquivo,
    header=None,
    names=["timestamp", "heart_rate"]
)

# Mostra as primeiras linhas
print(dados.head())

# Mostra informações sobre o arquivo
print("\nInformações do dataset:")
print(dados.info())

# Mostra o tamanho
print("\nQuantidade de linhas e colunas:")
print(dados.shape)

# Calcula a diferença de tempo entre cada medição
dados["intervalo"] = dados["timestamp"].diff()

print("\nIntervalo entre as medições:")
print(dados["intervalo"].describe())

print("\nPrimeiros intervalos:")
print(dados["intervalo"].head(10))