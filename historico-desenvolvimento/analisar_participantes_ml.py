from pathlib import Path
from getpass import getpass

import pandas as pd
import mysql.connector


# ============================================================
# INÍCIO
# ============================================================

print("=" * 75)
print("ANÁLISE DOS PARTICIPANTES PARA MACHINE LEARNING")
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
    # 2. BUSCAR DISTRIBUIÇÃO DOS ESTÁGIOS
    # ========================================================

    cursor.execute(
        """
        SELECT

            p.codigo AS participante,

            ep.id_estagio_referencia AS label,

            COUNT(*) AS quantidade

        FROM epocas ep

        INNER JOIN noites n
            ON n.id_noite = ep.id_noite

        INNER JOIN participantes p
            ON p.id_participante =
               n.id_participante

        GROUP BY
            p.codigo,
            ep.id_estagio_referencia

        ORDER BY
            p.codigo,
            ep.id_estagio_referencia;
        """
    )


    dados = cursor.fetchall()


    # ========================================================
    # 3. TRANSFORMAR EM DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        dados,
        columns=[
            "participante",
            "label",
            "quantidade"
        ]
    )


    # ========================================================
    # 4. CRIAR TABELA PARTICIPANTE x ESTÁGIO
    # ========================================================

    tabela = df.pivot(
        index="participante",
        columns="label",
        values="quantidade"
    )


    tabela = tabela.fillna(
        0
    ).astype(int)


    # ========================================================
    # 5. GARANTIR AS CINCO CLASSES
    # ========================================================

    for label in [
        0,
        1,
        2,
        3,
        4
    ]:

        if label not in tabela.columns:

            tabela[label] = 0


    tabela = tabela[
        [
            0,
            1,
            2,
            3,
            4
        ]
    ]


    # ========================================================
    # 6. RENOMEAR
    # ========================================================

    tabela.columns = [
        "Wake",
        "N1",
        "N2",
        "N3",
        "REM"
    ]


    # ========================================================
    # 7. TOTAL DE ÉPOCAS
    # ========================================================

    tabela[
        "Total"
    ] = tabela.sum(
        axis=1
    )


    # ========================================================
    # 8. PERCENTUAIS
    # ========================================================

    for estagio in [
        "Wake",
        "N1",
        "N2",
        "N3",
        "REM"
    ]:

        tabela[
            f"{estagio}_pct"
        ] = (

            tabela[
                estagio
            ]

            / tabela[
                "Total"
            ]

            * 100
        )


    # ========================================================
    # 9. MOSTRAR CONTAGENS
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "QUANTIDADE DE ÉPOCAS POR PARTICIPANTE"
    )

    print(
        "=" * 75
    )


    print(

        tabela[
            [
                "Wake",
                "N1",
                "N2",
                "N3",
                "REM",
                "Total"
            ]
        ]

        .to_string()
    )


    # ========================================================
    # 10. MOSTRAR PERCENTUAIS
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PERCENTUAL DOS ESTÁGIOS POR PARTICIPANTE"
    )

    print(
        "=" * 75
    )


    percentuais = tabela[
        [
            "Wake_pct",
            "N1_pct",
            "N2_pct",
            "N3_pct",
            "REM_pct"
        ]
    ].copy()


    print(

        percentuais

        .round(2)

        .to_string()
    )


    # ========================================================
    # 11. DISTRIBUIÇÃO GERAL
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DISTRIBUIÇÃO GERAL"
    )

    print(
        "=" * 75
    )


    total_geral = tabela[
        [
            "Wake",
            "N1",
            "N2",
            "N3",
            "REM"
        ]
    ].sum()


    total_epocas = int(
        total_geral.sum()
    )


    for estagio in [
        "Wake",
        "N1",
        "N2",
        "N3",
        "REM"
    ]:

        quantidade = int(
            total_geral[
                estagio
            ]
        )


        percentual = (
            quantidade
            / total_epocas
            * 100
        )


        print(

            f"{estagio:5s} "

            f"-> {quantidade:5d} "

            f"({percentual:.2f}%)"
        )


    print(
        f"\nTotal: {total_epocas}"
    )


    # ========================================================
    # 12. VERIFICAR PARTICIPANTES SEM ALGUMA CLASSE
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "VERIFICAÇÃO DAS CINCO CLASSES"
    )

    print(
        "=" * 75
    )


    participantes_problematicos = []


    for participante, linha in tabela.iterrows():

        faltantes = []


        for estagio in [
            "Wake",
            "N1",
            "N2",
            "N3",
            "REM"
        ]:

            if linha[
                estagio
            ] == 0:

                faltantes.append(
                    estagio
                )


        if faltantes:

            participantes_problematicos.append(
                (
                    participante,
                    faltantes
                )
            )


    if len(
        participantes_problematicos
    ) == 0:

        print(
            "\n✅ Todos os participantes "
            "possuem exemplos das cinco classes."
        )

    else:

        print(
            "\nParticipantes sem alguma classe:"
        )


        for (
            participante,
            faltantes
        ) in participantes_problematicos:

            print(
                f"{participante}: "
                f"{', '.join(faltantes)}"
            )


    # ========================================================
    # 13. SALVAR RELATÓRIO
    # ========================================================

    arquivo_saida = Path(
        "data/processed/"
        "distribuicao_participantes_ml.csv"
    )


    tabela.to_csv(
        arquivo_saida
    )


    print(
        "\nArquivo criado:"
    )

    print(
        arquivo_saida
    )


    print(
        "\n✅ ANÁLISE CONCLUÍDA"
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