from pathlib import Path
from getpass import getpass

import joblib
import pandas as pd
import mysql.connector

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

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
# FEATURES UTILIZADAS
# ============================================================

FEATURES = [
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",

    "motion_mean",
    "motion_std",
    "motion_max"
]


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
print("TREINAMENTO DO PRIMEIRO MODELO BASELINE")
print("=" * 75)

print(
    "\nModelo: Random Forest"
)

print(
    "Objetivo: classificar "
    "Wake / N1 / N2 / N3 / REM"
)


# ============================================================
# 1. CARREGAR DIVISÃO DOS PARTICIPANTES
# ============================================================

print(
    "\nCarregando divisão "
    "treino / validação / teste..."
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


print(
    f"Treino: "
    f"{len(participantes_treino)} participantes"
)

print(
    f"Validação: "
    f"{len(participantes_validacao)} participantes"
)

print(
    f"Teste: "
    f"{len(participantes_teste)} participantes"
)


# ============================================================
# 2. CONECTAR AO MYSQL
# ============================================================

senha = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


conexao = None


try:

    conexao = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,

        user="projeto_sono_app",
        password=senha,

        database="projeto_sono"
    )


    print(
        "\n✅ Conexão com MySQL realizada."
    )


    # ========================================================
    # 3. BUSCAR OS DADOS DO BANCO
    # ========================================================

    print(
        "\nCarregando épocas do banco..."
    )


    consulta = """
        SELECT

            p.codigo AS participante,

            ep.hr_mean,
            ep.hr_min,
            ep.hr_max,
            ep.hr_std,

            ep.motion_mean,
            ep.motion_std,
            ep.motion_max,

            ep.id_estagio_referencia AS label

        FROM epocas ep

        INNER JOIN noites n
            ON n.id_noite = ep.id_noite

        INNER JOIN participantes p
            ON p.id_participante =
               n.id_participante;
    """


    cursor = conexao.cursor()

    cursor.execute(
        consulta
    )


    dados = cursor.fetchall()


    colunas = [
        "participante",

        "hr_mean",
        "hr_min",
        "hr_max",
        "hr_std",

        "motion_mean",
        "motion_std",
        "motion_max",

        "label"
    ]


    dataset = pd.DataFrame(
        dados,
        columns=colunas
    )


    cursor.close()


    print(
        f"Épocas carregadas: "
        f"{len(dataset)}"
    )


    # ========================================================
    # 4. VALIDAR DATASET
    # ========================================================

    if len(dataset) != 60165:

        raise ValueError(
            f"Foram encontradas "
            f"{len(dataset)} épocas. "
            f"Eram esperadas 60165."
        )


    if dataset.isna().sum().sum() > 0:

        raise ValueError(
            "Foram encontrados valores ausentes."
        )


    # ========================================================
    # 5. SEPARAR TREINO
    # ========================================================

    treino = dataset[
        dataset[
            "participante"
        ].isin(
            participantes_treino
        )
    ].copy()


    # ========================================================
    # 6. SEPARAR VALIDAÇÃO
    # ========================================================

    validacao = dataset[
        dataset[
            "participante"
        ].isin(
            participantes_validacao
        )
    ].copy()


    # ========================================================
    # 7. SEPARAR TESTE
    # ========================================================

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


    # ========================================================
    # 8. GARANTIR QUE NÃO EXISTE VAZAMENTO
    # ========================================================

    conjunto_treino = set(
        treino["participante"].unique()
    )

    conjunto_validacao = set(
        validacao["participante"].unique()
    )

    conjunto_teste = set(
        teste["participante"].unique()
    )


    if (
        conjunto_treino
        & conjunto_validacao
    ):

        raise ValueError(
            "Existe participante presente "
            "em treino e validação."
        )


    if (
        conjunto_treino
        & conjunto_teste
    ):

        raise ValueError(
            "Existe participante presente "
            "em treino e teste."
        )


    if (
        conjunto_validacao
        & conjunto_teste
    ):

        raise ValueError(
            "Existe participante presente "
            "em validação e teste."
        )


    print(
        "\n✅ Nenhum participante "
        "está presente em dois grupos."
    )


    # ========================================================
    # 9. X E Y DE TREINO
    # ========================================================

    X_treino = treino[
        FEATURES
    ]


    y_treino = treino[
        "label"
    ]


    # ========================================================
    # 10. X E Y DE VALIDAÇÃO
    # ========================================================

    X_validacao = validacao[
        FEATURES
    ]


    y_validacao = validacao[
        "label"
    ]


    # ========================================================
    # IMPORTANTE
    # ========================================================
    #
    # O conjunto de teste NÃO será utilizado
    # nesta etapa.
    #
    # Ele permanece reservado para a avaliação
    # final do modelo escolhido.
    # ========================================================


    # ========================================================
    # 11. CRIAR RANDOM FOREST
    # ========================================================

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


    # ========================================================
    # 12. TREINAR
    # ========================================================

    print(
        "\nTreinando modelo..."
    )


    modelo.fit(
        X_treino,
        y_treino
    )


    print(
        "✅ Treinamento concluído."
    )


    # ========================================================
    # 13. PREDIZER VALIDAÇÃO
    # ========================================================

    print(
        "\nClassificando conjunto "
        "de validação..."
    )


    y_pred = modelo.predict(
        X_validacao
    )


    # ========================================================
    # 14. MÉTRICAS GERAIS
    # ========================================================

    accuracy = accuracy_score(
        y_validacao,
        y_pred
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_validacao,
            y_pred
        )
    )


    macro_f1 = f1_score(
        y_validacao,
        y_pred,
        average="macro"
    )


    weighted_f1 = f1_score(
        y_validacao,
        y_pred,
        average="weighted"
    )


    # ========================================================
    # 15. RESULTADOS
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RESULTADO NA VALIDAÇÃO"
    )

    print(
        "=" * 75
    )


    print(
        f"\nAccuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )


    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f} "
        f"({balanced_accuracy * 100:.2f}%)"
    )


    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )


    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )


    # ========================================================
    # 16. CLASSIFICATION REPORT
    # ========================================================

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


    relatorio_texto = (
        classification_report(

            y_validacao,
            y_pred,

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
    )


    print(
        "\n"
        + relatorio_texto
    )


    # ========================================================
    # 17. MATRIZ DE CONFUSÃO
    # ========================================================

    matriz = confusion_matrix(

        y_validacao,
        y_pred,

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


    # ========================================================
    # 18. IMPORTÂNCIA DAS FEATURES
    # ========================================================

    importancia = pd.DataFrame({

        "feature":
            FEATURES,

        "importancia":
            modelo.feature_importances_

    })


    importancia = (
        importancia

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
        "IMPORTÂNCIA DAS CARACTERÍSTICAS"
    )

    print(
        "=" * 75
    )


    print(
        "\n"
        + importancia.to_string(
            index=False
        )
    )


    # ========================================================
    # 19. SALVAR MATRIZ
    # ========================================================

    matriz_df.to_csv(

        PASTA_RESULTADOS
        / "matriz_confusao_baseline_validacao.csv"
    )


    # ========================================================
    # 20. SALVAR IMPORTÂNCIA
    # ========================================================

    importancia.to_csv(

        PASTA_RESULTADOS
        / "importancia_features_baseline.csv",

        index=False
    )


    # ========================================================
    # 21. SALVAR RELATÓRIO COMPLETO
    # ========================================================

    relatorio_dict = classification_report(

        y_validacao,
        y_pred,

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


    relatorio_df.to_csv(

        PASTA_RESULTADOS
        / "relatorio_baseline_validacao.csv"
    )


    # ========================================================
    # 22. SALVAR MODELO
    # ========================================================

    arquivo_modelo = (

        PASTA_MODELOS
        / "random_forest_regularizada.joblib"
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
        "ARQUIVOS CRIADOS"
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
        "\nResultados:"
    )

    print(
        PASTA_RESULTADOS
    )


    print(
        "\n✅ BASELINE FINALIZADA"
    )


except Exception as erro:

    print(
        "\n❌ ERRO:"
    )

    print(
        erro
    )


finally:

    if (
        conexao is not None
        and conexao.is_connected()
    ):

        conexao.close()

        print(
            "\nConexão com MySQL encerrada."
        )