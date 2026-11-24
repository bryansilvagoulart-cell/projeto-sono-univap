from pathlib import Path
from getpass import getpass

import joblib
import pandas as pd
import mysql.connector

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_DIVISAO = Path(
    "data/processed/divisao_participantes_ml.csv"
)

ARQUIVO_MODELO = Path(
    "models/random_forest_baseline.joblib"
)


FEATURES = [
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",

    "motion_mean",
    "motion_std",
    "motion_max"
]


print("=" * 75)
print("DIAGNÓSTICO DO MODELO BASELINE")
print("=" * 75)


# ============================================================
# 1. CARREGAR DIVISÃO
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


# ============================================================
# 2. CARREGAR MODELO
# ============================================================

print(
    "\nCarregando modelo..."
)


modelo = joblib.load(
    ARQUIVO_MODELO
)


print(
    "✅ Modelo carregado."
)


# ============================================================
# 3. MYSQL
# ============================================================

senha = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


conexao = None
cursor = None


try:

    conexao = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,

        user="projeto_sono_app",
        password=senha,

        database="projeto_sono"
    )


    cursor = conexao.cursor()


    print(
        "\n✅ Conexão com MySQL realizada."
    )


    # ========================================================
    # 4. CARREGAR DADOS
    # ========================================================

    cursor.execute(
        """
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
    )


    dados = cursor.fetchall()


    dataset = pd.DataFrame(
        dados,
        columns=[
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
    )


    print(
        f"Épocas carregadas: "
        f"{len(dataset)}"
    )


    # ========================================================
    # 5. TREINO
    # ========================================================

    treino = dataset[
        dataset["participante"].isin(
            participantes_treino
        )
    ].copy()


    # ========================================================
    # 6. VALIDAÇÃO
    # ========================================================

    validacao = dataset[
        dataset["participante"].isin(
            participantes_validacao
        )
    ].copy()


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


    # ========================================================
    # 7. PREVISÕES
    # ========================================================

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


    # ========================================================
    # 8. MÉTRICAS DE TREINO
    # ========================================================

    acc_treino = accuracy_score(
        y_treino,
        pred_treino
    )


    bal_treino = balanced_accuracy_score(
        y_treino,
        pred_treino
    )


    f1_treino = f1_score(
        y_treino,
        pred_treino,
        average="macro"
    )


    # ========================================================
    # 9. MÉTRICAS DE VALIDAÇÃO
    # ========================================================

    acc_validacao = accuracy_score(
        y_validacao,
        pred_validacao
    )


    bal_validacao = balanced_accuracy_score(
        y_validacao,
        pred_validacao
    )


    f1_validacao = f1_score(
        y_validacao,
        pred_validacao,
        average="macro"
    )


    # ========================================================
    # 10. BASELINE MAIS SIMPLES POSSÍVEL
    # ========================================================

    classe_majoritaria = int(
        y_treino
        .value_counts()
        .idxmax()
    )


    pred_majoritaria = [
        classe_majoritaria
    ] * len(y_validacao)


    acc_majoritaria = accuracy_score(
        y_validacao,
        pred_majoritaria
    )


    f1_majoritaria = f1_score(
        y_validacao,
        pred_majoritaria,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # RESULTADO GERAL
    # ========================================================

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
        f"{f1_treino:>15.4f}"
        f"{f1_validacao:>15.4f}"
    )


    # ========================================================
    # BASELINE MAJORITÁRIO
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CLASSIFICADOR MAJORITÁRIO"
    )

    print(
        "=" * 75
    )


    print(
        f"\nClasse mais comum no treino: "
        f"{classe_majoritaria}"
    )


    print(
        f"Accuracy na validação: "
        f"{acc_majoritaria:.4f} "
        f"({acc_majoritaria * 100:.2f}%)"
    )


    print(
        f"Macro F1 na validação: "
        f"{f1_majoritaria:.4f}"
    )


    # ========================================================
    # 11. RESULTADO POR PARTICIPANTE DE VALIDAÇÃO
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DESEMPENHO POR PARTICIPANTE DE VALIDAÇÃO"
    )

    print(
        "=" * 75
    )


    for participante in participantes_validacao:

        dados_participante = validacao[
            validacao["participante"]
            == participante
        ].copy()


        X_participante = dados_participante[
            FEATURES
        ]


        y_participante = dados_participante[
            "label"
        ]


        pred_participante = modelo.predict(
            X_participante
        )


        acc = accuracy_score(
            y_participante,
            pred_participante
        )


        bal_acc = balanced_accuracy_score(
            y_participante,
            pred_participante
        )


        macro_f1 = f1_score(
            y_participante,
            pred_participante,
            average="macro",
            zero_division=0
        )


        print(
            f"\n{participante}"
        )


        print(
            f"Épocas: "
            f"{len(dados_participante)}"
        )


        print(
            f"Accuracy: "
            f"{acc:.4f} "
            f"({acc * 100:.2f}%)"
        )


        print(
            f"Balanced Accuracy: "
            f"{bal_acc:.4f}"
        )


        print(
            f"Macro F1: "
            f"{macro_f1:.4f}"
        )


    print(
        "\n"
        + "=" * 75
    )

    print(
        "DIAGNÓSTICO FINALIZADO"
    )

    print(
        "=" * 75
    )


except Exception as erro:

    print(
        "\n❌ ERRO:"
    )

    print(
        erro
    )


finally:

    if cursor is not None:
        cursor.close()


    if (
        conexao is not None
        and conexao.is_connected()
    ):

        conexao.close()

        print(
            "\nConexão com MySQL encerrada."
        )