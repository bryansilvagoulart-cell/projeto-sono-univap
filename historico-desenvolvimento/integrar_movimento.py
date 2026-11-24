import pandas as pd
import numpy as np


# ==========================================================
# 1. CARREGAR OS DADOS
# ==========================================================

print("=" * 60)
print("INTEGRAÇÃO DO MOVIMENTO COM AS ÉPOCAS")
print("=" * 60)

print("\nCarregando movimento...")

motion = pd.read_csv(
    "data/processed/movimento_calculado.csv"
)

print("Medições de movimento:", len(motion))


print("\nCarregando dataset limpo...")

df = pd.read_csv(
    "data/processed/dataset_limpo.csv"
)

# Converter datas para datetime
df["inicio"] = pd.to_datetime(df["inicio"], utc=True)
df["fim"] = pd.to_datetime(df["fim"], utc=True)

# Converter timestamp do movimento
motion["Timestamp"] = pd.to_datetime(
    motion["Timestamp"],
    unit="s",
    utc=True
)

print("Épocas:", len(df))


# ==========================================================
# 2. PREPARAR AS ÉPOCAS
# ==========================================================

print("\nPreparando épocas de 30 segundos...")

resultados = []


# ==========================================================
# 3. PROCESSAR CADA ÉPOCA
# ==========================================================

for _, epoca in df.iterrows():

    inicio = epoca["inicio"]
    fim = epoca["fim"]

    # Selecionar as medições de movimento
    # que aconteceram dentro da época
    dados = motion[
        (motion["Timestamp"] >= inicio) &
        (motion["Timestamp"] < fim)
    ]

    # Estatísticas do movimento
    if len(dados) > 0:

        movimento_medio = dados["movimento"].mean()
        movimento_maximo = dados["movimento"].max()
        movimento_desvio = dados["movimento"].std()

        # Quantidade de movimentos
        quantidade_medicoes = len(dados)

    else:

        movimento_medio = np.nan
        movimento_maximo = np.nan
        movimento_desvio = np.nan
        quantidade_medicoes = 0


    # ======================================================
    # 4. GUARDAR RESULTADO
    # ======================================================

    resultados.append({

        "epoca": epoca["epoca"],

        "inicio": inicio,

        "fim": fim,

        "hr_mean": epoca["hr_mean"],

        "hr_min": epoca["hr_min"],

        "hr_max": epoca["hr_max"],

        "hr_std": epoca["hr_std"],

        "motion_mean": movimento_medio,

        "motion_std": movimento_desvio,

        "motion_max": movimento_maximo,

        "motion_samples": quantidade_medicoes,

        "label": epoca["label"]

    })


# ==========================================================
# 5. CRIAR DATAFRAME FINAL
# ==========================================================

resultado = pd.DataFrame(resultados)


# ==========================================================
# 6. MOSTRAR RESULTADO
# ==========================================================

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)

print("\nQuantidade de épocas:")
print(len(resultado))

print("\nQuantidade de colunas:")
print(len(resultado.columns))

print("\nColunas:")
print(resultado.columns.tolist())


print("\nPrimeiras épocas:")

print(
    resultado.head(10).to_string(index=False)
)


# ==========================================================
# 7. VERIFICAR QUANTIDADE DE MEDIÇÕES POR ÉPOCA
# ==========================================================

print("\n" + "=" * 60)
print("MEDIÇÕES DE MOVIMENTO POR ÉPOCA")
print("=" * 60)

print(
    resultado["motion_samples"].describe()
)


# ==========================================================
# 8. VERIFICAR VALORES AUSENTES
# ==========================================================

print("\n" + "=" * 60)
print("VALORES AUSENTES")
print("=" * 60)

print(
    resultado.isna().sum()
)


# ==========================================================
# 9. SALVAR
# ==========================================================

arquivo_saida = (
    "data/processed/dataset_final.csv"
)

resultado.to_csv(
    arquivo_saida,
    index=False
)


print("\n" + "=" * 60)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 60)

print("\nArquivo salvo em:")
print(arquivo_saida)