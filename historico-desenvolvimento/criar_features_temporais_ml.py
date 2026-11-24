from pathlib import Path
from getpass import getpass

import pandas as pd
import mysql.connector


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ARQUIVO_SAIDA = Path(
    "data/processed/dataset_ml_features_temporais.csv"
)


# ============================================================
# INÍCIO
# ============================================================

print("=" * 75)
print("CRIAÇÃO DAS FEATURES TEMPORAIS PARA MACHINE LEARNING")
print("=" * 75)


senha = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


conexao = None
cursor = None


try:

    # ========================================================
    # 1. CONECTAR AO MYSQL
    # ========================================================

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
    # 2. CARREGAR ÉPOCAS
    # ========================================================

    print(
        "\nCarregando épocas..."
    )


    cursor.execute(
        """
        SELECT

            p.codigo AS participante,

            n.id_noite,
            n.numero_noite,

            ep.numero_epoca,

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
            ON p.id_participante = n.id_participante

        ORDER BY
            p.codigo,
            n.numero_noite,
            ep.numero_epoca;
        """
    )


    dados = cursor.fetchall()


    dataset = pd.DataFrame(
        dados,
        columns=[
            "participante",
            "id_noite",
            "noite",
            "epoca",

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
        f"Épocas carregadas: {len(dataset)}"
    )


    if len(dataset) != 60165:

        raise ValueError(
            f"Esperávamos 60165 épocas, "
            f"mas encontramos {len(dataset)}."
        )


    # ========================================================
    # 3. IDENTIFICAR BURACOS ENTRE ÉPOCAS
    # ========================================================

    print(
        "\nIdentificando trechos contínuos..."
    )


    diferenca = (
        dataset
        .groupby("id_noite")["epoca"]
        .diff()
    )


    # Quando a diferença não é 1, significa
    # que existe um buraco causado pela remoção
    # de épocas no controle de qualidade.

    dataset["novo_segmento"] = (
        diferenca.ne(1)
    )


    dataset["segmento"] = (
        dataset
        .groupby("id_noite")["novo_segmento"]
        .cumsum()
        .astype(int)
    )


    # ========================================================
    # 4. AGRUPAR TRECHOS CONTÍNUOS
    # ========================================================

    grupos = dataset.groupby(
        [
            "id_noite",
            "segmento"
        ],
        sort=False
    )


    # ========================================================
    # 5. DIFERENÇA EM RELAÇÃO À ÉPOCA ANTERIOR
    # ========================================================

    dataset["hr_mean_delta1"] = (
        grupos["hr_mean"]
        .diff()
        .fillna(0)
    )


    dataset["motion_mean_delta1"] = (
        grupos["motion_mean"]
        .diff()
        .fillna(0)
    )


    # ========================================================
    # 6. FC MÉDIA NOS ÚLTIMOS 5 BLOCOS
    # ========================================================

    dataset["hr_mean_roll5_mean"] = (
        grupos["hr_mean"]
        .transform(
            lambda serie:
                serie
                .rolling(
                    window=5,
                    min_periods=1
                )
                .mean()
        )
    )


    # ========================================================
    # 7. VARIAÇÃO DA FC NOS ÚLTIMOS 5 BLOCOS
    # ========================================================

    dataset["hr_mean_roll5_std"] = (
        grupos["hr_mean"]
        .transform(
            lambda serie:
                serie
                .rolling(
                    window=5,
                    min_periods=1
                )
                .std(ddof=0)
        )
    )


    # ========================================================
    # 8. MÉDIA DO HR_STD NOS ÚLTIMOS 5 BLOCOS
    # ========================================================

    dataset["hr_std_roll5_mean"] = (
        grupos["hr_std"]
        .transform(
            lambda serie:
                serie
                .rolling(
                    window=5,
                    min_periods=1
                )
                .mean()
        )
    )


    # ========================================================
    # 9. MOVIMENTO MÉDIO NOS ÚLTIMOS 5 BLOCOS
    # ========================================================

    dataset["motion_mean_roll5_mean"] = (
        grupos["motion_mean"]
        .transform(
            lambda serie:
                serie
                .rolling(
                    window=5,
                    min_periods=1
                )
                .mean()
        )
    )


    # ========================================================
    # 10. VARIAÇÃO DO MOVIMENTO NOS ÚLTIMOS 5 BLOCOS
    # ========================================================

    dataset["motion_mean_roll5_std"] = (
        grupos["motion_mean"]
        .transform(
            lambda serie:
                serie
                .rolling(
                    window=5,
                    min_periods=1
                )
                .std(ddof=0)
        )
    )


    # ========================================================
    # 11. MOVIMENTO MÁXIMO MÉDIO NOS ÚLTIMOS 5 BLOCOS
    # ========================================================

    dataset["motion_max_roll5_mean"] = (
        grupos["motion_max"]
        .transform(
            lambda serie:
                serie
                .rolling(
                    window=5,
                    min_periods=1
                )
                .mean()
        )
    )


    # ========================================================
    # 12. FEATURES TEMPORAIS
    # ========================================================

    features_temporais = [
        "hr_mean_delta1",
        "motion_mean_delta1",

        "hr_mean_roll5_mean",
        "hr_mean_roll5_std",
        "hr_std_roll5_mean",

        "motion_mean_roll5_mean",
        "motion_mean_roll5_std",
        "motion_max_roll5_mean"
    ]


    # ========================================================
    # 13. VALIDAR
    # ========================================================

    print(
        "\nValidando features..."
    )


    ausentes = (
        dataset[
            features_temporais
        ]
        .isna()
        .sum()
        .sum()
    )


    numero_segmentos = (
        dataset[
            [
                "id_noite",
                "segmento"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


    print(
        f"Linhas finais: {len(dataset)}"
    )


    print(
        f"Trechos contínuos encontrados: "
        f"{numero_segmentos}"
    )


    print(
        f"Valores ausentes nas novas features: "
        f"{ausentes}"
    )


    if ausentes != 0:

        raise ValueError(
            "Existem valores ausentes "
            "nas features temporais."
        )


    # ========================================================
    # 14. REMOVER COLUNA AUXILIAR
    # ========================================================

    dataset = dataset.drop(
        columns=[
            "novo_segmento"
        ]
    )


    # ========================================================
    # 15. SALVAR
    # ========================================================

    ARQUIVO_SAIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    dataset.to_csv(
        ARQUIVO_SAIDA,
        index=False
    )


    # ========================================================
    # RESULTADO
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FEATURES TEMPORAIS CRIADAS"
    )

    print(
        "=" * 75
    )


    print(
        "\nArquivo:"
    )

    print(
        ARQUIVO_SAIDA
    )


    print(
        f"\nFeatures temporais adicionadas: "
        f"{len(features_temporais)}"
    )


    for feature in features_temporais:

        print(
            f"  - {feature}"
        )


    print(
        "\n✅ DATASET TEMPORAL CRIADO COM SUCESSO"
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