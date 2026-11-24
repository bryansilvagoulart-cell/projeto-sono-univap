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
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_multiescala.csv"
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
# FEATURES DA ÉPOCA ATUAL
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
# FEATURES DE 2,5 MINUTOS
# ============================================================

FEATURES_2_5_MIN = [
    "hr_mean_delta1",
    "motion_mean_delta1",

    "hr_mean_roll5_mean",
    "hr_mean_roll5_std",
    "hr_std_roll5_mean",

    "motion_mean_roll5_mean",
    "motion_mean_roll5_std",
    "motion_max_roll5_mean"
]


# ============================================================
# FEATURES DE 5 MINUTOS
# ============================================================

FEATURES_5_MIN = [
    "hr_mean_roll10_mean",
    "hr_mean_roll10_std",
    "hr_std_roll10_mean",

    "motion_mean_roll10_mean",
    "motion_mean_roll10_std",
    "motion_max_roll10_mean"
]


# ============================================================
# FEATURES DE 10 MINUTOS
# ============================================================

FEATURES_10_MIN = [
    "hr_mean_roll20_mean",
    "hr_mean_roll20_std",
    "hr_std_roll20_mean",

    "motion_mean_roll20_mean",
    "motion_mean_roll20_std",
    "motion_max_roll20_mean"
]


# ============================================================
# TODAS AS FEATURES
# ============================================================

FEATURES = (
    FEATURES_ORIGINAIS
    + FEATURES_2_5_MIN
    + FEATURES_5_MIN
    + FEATURES_10_MIN
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
print("RANDOM FOREST MULTIESCALA SEM BALANCEAMENTO DE CLASSES")
print("=" * 75)


print(
    f"\nQuantidade de features: "
    f"{len(FEATURES)}"
)


print(
    "\nConfiguração:"
)

print(
    "  Random Forest"
)

print(
    "  27 features"
)

print(
    "  Sem class_weight"
)

print(
    "  Teste continua reservado"
)


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

print(
    "\nCarregando dataset multiescala..."
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
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


# ============================================================
# 2. VERIFICAR FEATURES
# ============================================================

features_faltando = [
    feature
    for feature in FEATURES
    if feature not in dataset.columns
]


if features_faltando:

    raise ValueError(
        "Features ausentes:\n"
        + "\n".join(
            features_faltando
        )
    )


print(
    "✅ Todas as 27 features foram encontradas."
)


# ============================================================
# 3. CARREGAR DIVISÃO DOS PARTICIPANTES
# ============================================================

print(
    "\nCarregando divisão dos participantes..."
)


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
# 4. SEPARAR TREINO
# ============================================================

treino = dataset[
    dataset[
        "participante"
    ].isin(
        participantes_treino
    )
].copy()


# ============================================================
# 5. SEPARAR VALIDAÇÃO
# ============================================================

validacao = dataset[
    dataset[
        "participante"
    ].isin(
        participantes_validacao
    )
].copy()


# ============================================================
# 6. SEPARAR TESTE
# ============================================================
#
# O teste será separado apenas para conferir
# sua quantidade.
#
# NÃO será utilizado pelo modelo.
#
# ============================================================

teste = dataset[
    dataset[
        "participante"
    ].isin(
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
# 7. VERIFICAR VAZAMENTO ENTRE PARTICIPANTES
# ============================================================

treino_ids = set(
    treino["participante"].unique()
)

validacao_ids = set(
    validacao["participante"].unique()
)

teste_ids = set(
    teste["participante"].unique()
)


if treino_ids & validacao_ids:

    raise ValueError(
        "Participante presente em treino e validação."
    )


if treino_ids & teste_ids:

    raise ValueError(
        "Participante presente em treino e teste."
    )


if validacao_ids & teste_ids:

    raise ValueError(
        "Participante presente em validação e teste."
    )


print(
    "\n✅ Nenhum participante aparece "
    "em mais de um grupo."
)


# ============================================================
# 8. CRIAR X E Y
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
# 9. VERIFICAR VALORES AUSENTES
# ============================================================

ausentes_treino = (
    X_treino
    .isna()
    .sum()
    .sum()
)


ausentes_validacao = (
    X_validacao
    .isna()
    .sum()
    .sum()
)


if ausentes_treino != 0:

    raise ValueError(
        f"Existem {ausentes_treino} "
        f"valores ausentes no treino."
    )


if ausentes_validacao != 0:

    raise ValueError(
        f"Existem {ausentes_validacao} "
        f"valores ausentes na validação."
    )


print(
    "✅ Dados validados."
)


# ============================================================
# 10. CRIAR RANDOM FOREST
# ============================================================
#
# Este modelo é praticamente igual ao nosso
# melhor modelo multiescala.
#
# A diferença principal é:
#
# class_weight=None
#
# Ou seja:
#
# nenhuma classe recebe peso artificial.
#
# ============================================================

print(
    "\nCriando Random Forest "
    "sem balanceamento..."
)


modelo = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_split=20,

    min_samples_leaf=10,

    max_features="sqrt",

    class_weight=None,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 11. TREINAR
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
# 12. PREVISÕES NO TREINO
# ============================================================

print(
    "\nCalculando resultado do treino..."
)


pred_treino = modelo.predict(
    X_treino
)


# ============================================================
# 13. PREVISÕES NA VALIDAÇÃO
# ============================================================

print(
    "Calculando resultado da validação..."
)


pred_validacao = modelo.predict(
    X_validacao
)


# ============================================================
# 14. MÉTRICAS DO TREINO
# ============================================================

accuracy_treino = accuracy_score(
    y_treino,
    pred_treino
)


balanced_treino = (
    balanced_accuracy_score(
        y_treino,
        pred_treino
    )
)


macro_f1_treino = f1_score(
    y_treino,
    pred_treino,
    average="macro"
)


# ============================================================
# 15. MÉTRICAS DA VALIDAÇÃO
# ============================================================

accuracy_validacao = accuracy_score(
    y_validacao,
    pred_validacao
)


balanced_validacao = (
    balanced_accuracy_score(
        y_validacao,
        pred_validacao
    )
)


macro_f1_validacao = f1_score(
    y_validacao,
    pred_validacao,
    average="macro"
)


weighted_f1_validacao = f1_score(
    y_validacao,
    pred_validacao,
    average="weighted"
)


# ============================================================
# 16. MOSTRAR MÉTRICAS
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
    f"{accuracy_treino:>15.4f}"
    f"{accuracy_validacao:>15.4f}"
)


print(
    f"{'Balanced Accuracy':<25}"
    f"{balanced_treino:>15.4f}"
    f"{balanced_validacao:>15.4f}"
)


print(
    f"{'Macro F1':<25}"
    f"{macro_f1_treino:>15.4f}"
    f"{macro_f1_validacao:>15.4f}"
)


print(
    f"\nWeighted F1 validação: "
    f"{weighted_f1_validacao:.4f}"
)


# ============================================================
# 17. RESULTADO POR ESTÁGIO
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


relatorio_texto = classification_report(

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
    + relatorio_texto
)


# ============================================================
# 18. MATRIZ DE CONFUSÃO
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
# 19. IMPORTÂNCIA DAS FEATURES
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
# 20. CRIAR RELATÓRIO PARA CSV
# ============================================================

relatorio_dict = classification_report(

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

    output_dict=True,

    zero_division=0
)


relatorio_df = pd.DataFrame(
    relatorio_dict
).transpose()


# ============================================================
# 21. CAMINHOS DOS RESULTADOS
# ============================================================

arquivo_modelo = (
    PASTA_MODELOS
    / "random_forest_sem_balanceamento.joblib"
)


arquivo_matriz = (
    PASTA_RESULTADOS
    / "matriz_confusao_rf_sem_balanceamento.csv"
)


arquivo_importancias = (
    PASTA_RESULTADOS
    / "importancia_features_rf_sem_balanceamento.csv"
)


arquivo_relatorio = (
    PASTA_RESULTADOS
    / "relatorio_rf_sem_balanceamento.csv"
)


# ============================================================
# 22. SALVAR
# ============================================================

joblib.dump(
    modelo,
    arquivo_modelo
)


matriz_df.to_csv(
    arquivo_matriz
)


importancias.to_csv(
    arquivo_importancias,
    index=False
)


relatorio_df.to_csv(
    arquivo_relatorio
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "EXPERIMENTO FINALIZADO"
)

print(
    "=" * 75
)


print(
    "\nModelo:"
)

print(
    arquivo_modelo
)


print(
    "\nMatriz de confusão:"
)

print(
    arquivo_matriz
)


print(
    "\nImportância das features:"
)

print(
    arquivo_importancias
)


print(
    "\nRelatório:"
)

print(
    arquivo_relatorio
)


print(
    "\n⚠️ As 12.061 épocas de teste "
    "continuam sem ser utilizadas."
)


print(
    "\n✅ RANDOM FOREST SEM BALANCEAMENTO CONCLUÍDA"
)