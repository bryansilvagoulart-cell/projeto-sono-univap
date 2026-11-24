from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_temporais.csv"
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
# FEATURES ORIGINAIS
# ============================================================

FEATURES_ORIGINAIS = [
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",

    "motion_mean",
    "motion_std",
    "motion_max"
]


# ============================================================
# FEATURES TEMPORAIS
# ============================================================

FEATURES_TEMPORAIS = [
    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean"
]


FEATURES = (
    FEATURES_ORIGINAIS
    + FEATURES_TEMPORAIS
)


NOMES_ESTAGIOS = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM"
]


# ============================================================
# INÍCIO
# ============================================================

print("=" * 75)
print("RANDOM FOREST COM FEATURES TEMPORAIS")
print("=" * 75)


print(
    f"\nQuantidade total de features: "
    f"{len(FEATURES)}"
)


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

print(
    "\nCarregando dataset temporal..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


print(
    f"Épocas carregadas: "
    f"{len(dataset)}"
)


if len(dataset) != 60165:

    raise ValueError(
        "Quantidade de épocas diferente "
        "das 60.165 esperadas."
    )


# ============================================================
# 2. CARREGAR DIVISÃO
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


participantes_treino = (
    divisao[
        divisao["grupo"] == "treino"
    ]["participante"]
    .tolist()
)


participantes_validacao = (
    divisao[
        divisao["grupo"] == "validacao"
    ]["participante"]
    .tolist()
)


participantes_teste = (
    divisao[
        divisao["grupo"] == "teste"
    ]["participante"]
    .tolist()
)


# ============================================================
# 3. SEPARAR OS GRUPOS
# ============================================================

treino = dataset[
    dataset["participante"].isin(
        participantes_treino
    )
].copy()


validacao = dataset[
    dataset["participante"].isin(
        participantes_validacao
    )
].copy()


teste = dataset[
    dataset["participante"].isin(
        participantes_teste
    )
].copy()


print(
    "\n"
    + "=" * 75
)

print(
    "DIVISÃO DOS DADOS"
)

print(
    "=" * 75
)


print(
    f"\nTreino: "
    f"{len(treino)} épocas"
)

print(
    f"Validação: "
    f"{len(validacao)} épocas"
)

print(
    f"Teste reservado: "
    f"{len(teste)} épocas"
)


# ============================================================
# 4. X E Y
# ============================================================

X_treino = treino[
    FEATURES
]


y_treino = treino[
    "label"
]


X_validacao = validacao[
    FEATURES
]


y_validacao = validacao[
    "label"
]


# ============================================================
# 5. VALIDAR FEATURES
# ============================================================

if X_treino.isna().sum().sum() != 0:

    raise ValueError(
        "Existem valores ausentes "
        "no conjunto de treino."
    )


if X_validacao.isna().sum().sum() != 0:

    raise ValueError(
        "Existem valores ausentes "
        "na validação."
    )


print(
    "\n✅ Features validadas."
)


# ============================================================
# 6. MODELO
# ============================================================

print(
    "\nCriando Random Forest..."
)


modelo = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_split=20,

    min_samples_leaf=10,

    max_features="sqrt",

    class_weight="balanced_subsample",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 7. TREINO
# ============================================================

print(
    "Treinando modelo..."
)


modelo.fit(
    X_treino,
    y_treino
)


print(
    "✅ Treinamento concluído."
)


# ============================================================
# 8. PREDIÇÕES
# ============================================================

print(
    "\nCalculando desempenho no treino..."
)


pred_treino = modelo.predict(
    X_treino
)


print(
    "Calculando desempenho na validação..."
)


pred_validacao = modelo.predict(
    X_validacao
)


# ============================================================
# 9. MÉTRICAS DE TREINO
# ============================================================

acc_treino = accuracy_score(
    y_treino,
    pred_treino
)


bal_treino = balanced_accuracy_score(
    y_treino,
    pred_treino
)


macro_treino = f1_score(
    y_treino,
    pred_treino,
    average="macro"
)


# ============================================================
# 10. MÉTRICAS DE VALIDAÇÃO
# ============================================================

acc_validacao = accuracy_score(
    y_validacao,
    pred_validacao
)


bal_validacao = balanced_accuracy_score(
    y_validacao,
    pred_validacao
)


macro_validacao = f1_score(
    y_validacao,
    pred_validacao,
    average="macro"
)


weighted_validacao = f1_score(
    y_validacao,
    pred_validacao,
    average="weighted"
)


# ============================================================
# 11. COMPARAÇÃO
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "TREINO x VALIDAÇÃO"
)

print(
    "=" * 75
)


print(
    f"\n{'Métrica':<25}"
    f"{'Treino':>15}"
    f"{'Validação':>15}"
)


print(
    "-" * 55
)


print(
    f"{'Accuracy':<25}"
    f"{acc_treino:>15.4f}"
    f"{acc_validacao:>15.4f}"
)


print(
    f"{'Balanced Accuracy':<25}"
    f"{bal_treino:>15.4f}"
    f"{bal_validacao:>15.4f}"
)


print(
    f"{'Macro F1':<25}"
    f"{macro_treino:>15.4f}"
    f"{macro_validacao:>15.4f}"
)


print(
    f"\nWeighted F1 validação: "
    f"{weighted_validacao:.4f}"
)


# ============================================================
# 12. RESULTADO POR ESTÁGIO
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "RESULTADO POR ESTÁGIO"
)

print(
    "=" * 75
)


relatorio = classification_report(

    y_validacao,
    pred_validacao,

    labels=[
        0,
        1,
        2,
        3,
        4
    ],

    target_names=
        NOMES_ESTAGIOS,

    digits=4,

    zero_division=0
)


print(
    "\n"
    + relatorio
)


# ============================================================
# 13. MATRIZ DE CONFUSÃO
# ============================================================

matriz = confusion_matrix(

    y_validacao,
    pred_validacao,

    labels=[
        0,
        1,
        2,
        3,
        4
    ]
)


matriz_df = pd.DataFrame(

    matriz,

    index=[
        "Real_Wake",
        "Real_N1",
        "Real_N2",
        "Real_N3",
        "Real_REM"
    ],

    columns=[
        "Pred_Wake",
        "Pred_N1",
        "Pred_N2",
        "Pred_N3",
        "Pred_REM"
    ]
)


print(
    "\n"
    + "=" * 75
)

print(
    "MATRIZ DE CONFUSÃO"
)

print(
    "=" * 75
)


print(
    "\n"
    + matriz_df.to_string()
)


# ============================================================
# 14. IMPORTÂNCIA DAS FEATURES
# ============================================================

importancias = pd.DataFrame({

    "feature":
        FEATURES,

    "importancia":
        modelo.feature_importances_
})


importancias = (

    importancias

    .sort_values(
        "importancia",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


print(
    "\n"
    + "=" * 75
)

print(
    "IMPORTÂNCIA DAS FEATURES"
)

print(
    "=" * 75
)


print(
    "\n"
    + importancias.to_string(
        index=False
    )
)


# ============================================================
# 15. SALVAR RESULTADOS
# ============================================================

matriz_df.to_csv(

    PASTA_RESULTADOS
    / "matriz_confusao_rf_temporal.csv"
)


importancias.to_csv(

    PASTA_RESULTADOS
    / "importancia_features_rf_temporal.csv",

    index=False
)


# ============================================================
# 16. SALVAR MODELO
# ============================================================

arquivo_modelo = (

    PASTA_MODELOS
    / "random_forest_temporal.joblib"
)


joblib.dump(
    modelo,
    arquivo_modelo
)


print(
    "\n"
    + "=" * 75
)

print(
    "MODELO TEMPORAL FINALIZADO"
)

print(
    "=" * 75
)


print(
    f"\nModelo salvo em:"
)

print(
    arquivo_modelo
)


print(
    "\n⚠️ O conjunto de teste "
    "continua sem ser utilizado."
)


print(
    "\n✅ EXPERIMENTO CONCLUÍDO"
)