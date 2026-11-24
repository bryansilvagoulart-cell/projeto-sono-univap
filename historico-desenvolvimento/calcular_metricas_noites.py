from getpass import getpass

import pandas as pd
import mysql.connector


# ============================================================
# INÍCIO
# ============================================================

print("=" * 70)
print("CÁLCULO DAS MÉTRICAS DAS NOITES")
print("=" * 70)


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
    # 2. BUSCAR AS 75 NOITES
    # ========================================================

    cursor.execute(
        """
        SELECT
            n.id_noite,
            p.codigo,
            n.numero_noite,
            n.cobertura_percentual

        FROM noites n

        INNER JOIN participantes p
            ON p.id_participante =
               n.id_participante

        ORDER BY
            p.codigo,
            n.numero_noite;
        """
    )


    noites = cursor.fetchall()


    print(
        f"\nNoites encontradas: "
        f"{len(noites)}"
    )


    if len(noites) != 75:

        raise ValueError(
            "O banco deveria possuir "
            "75 noites."
        )


    # ========================================================
    # 3. SQL PARA SALVAR AS MÉTRICAS
    # ========================================================

    sql_metricas = """
        INSERT INTO metricas_noite (

            id_noite,

            duracao_analisada_min,
            tempo_sono_min,

            wake_pct,
            n1_pct,
            n2_pct,
            n3_pct,
            rem_pct,

            hr_media,
            movimento_medio,

            transicoes_estagio,
            possiveis_despertares,

            indice_experimental
        )

        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )

        ON DUPLICATE KEY UPDATE

            duracao_analisada_min =
                VALUES(duracao_analisada_min),

            tempo_sono_min =
                VALUES(tempo_sono_min),

            wake_pct =
                VALUES(wake_pct),

            n1_pct =
                VALUES(n1_pct),

            n2_pct =
                VALUES(n2_pct),

            n3_pct =
                VALUES(n3_pct),

            rem_pct =
                VALUES(rem_pct),

            hr_media =
                VALUES(hr_media),

            movimento_medio =
                VALUES(movimento_medio),

            transicoes_estagio =
                VALUES(transicoes_estagio),

            possiveis_despertares =
                VALUES(possiveis_despertares),

            indice_experimental =
                VALUES(indice_experimental);
    """


    # ========================================================
    # 4. PROCESSAR CADA NOITE
    # ========================================================

    contador = 0


    for (
        id_noite,
        participante,
        numero_noite,
        cobertura_noite
    ) in noites:


        # ====================================================
        # 5. BUSCAR ÉPOCAS DA NOITE
        # ====================================================

        cursor.execute(
            """
            SELECT

                numero_epoca,
                id_estagio_referencia,

                hr_mean,
                motion_mean

            FROM epocas

            WHERE id_noite = %s

            ORDER BY numero_epoca;
            """,

            (
                id_noite,
            )
        )


        dados = cursor.fetchall()


        if len(dados) == 0:

            print(
                f"\n⚠️ {participante} / "
                f"noite {numero_noite} "
                f"não possui épocas."
            )

            continue


        # ====================================================
        # 6. TRANSFORMAR EM DATAFRAME
        # ====================================================

        df = pd.DataFrame(

            dados,

            columns=[
                "epoca",
                "label",
                "hr_mean",
                "motion_mean"
            ]
        )


        total_epocas = len(
            df
        )


        # ====================================================
        # 7. DURAÇÃO ANALISADA
        # ====================================================

        # Cada época possui 30 segundos.
        # Portanto:
        #
        # 2 épocas = 1 minuto.

        duracao_analisada_min = (
            total_epocas
            * 0.5
        )


        # ====================================================
        # 8. TEMPO DE SONO
        # ====================================================

        # Label 0 = Wake.
        #
        # Consideramos como sono:
        # N1, N2, N3 e REM.

        epocas_sono = int(
            (
                df["label"] != 0
            ).sum()
        )


        tempo_sono_min = (
            epocas_sono
            * 0.5
        )


        # ====================================================
        # 9. DISTRIBUIÇÃO DOS ESTÁGIOS
        # ====================================================

        contagem = (
            df["label"]
            .value_counts()
        )


        wake = int(
            contagem.get(
                0,
                0
            )
        )

        n1 = int(
            contagem.get(
                1,
                0
            )
        )

        n2 = int(
            contagem.get(
                2,
                0
            )
        )

        n3 = int(
            contagem.get(
                3,
                0
            )
        )

        rem = int(
            contagem.get(
                4,
                0
            )
        )


        wake_pct = (
            wake
            / total_epocas
            * 100
        )

        n1_pct = (
            n1
            / total_epocas
            * 100
        )

        n2_pct = (
            n2
            / total_epocas
            * 100
        )

        n3_pct = (
            n3
            / total_epocas
            * 100
        )

        rem_pct = (
            rem
            / total_epocas
            * 100
        )


        # ====================================================
        # 10. HR E MOVIMENTO
        # ====================================================

        hr_media = float(
            df["hr_mean"].mean()
        )


        movimento_medio = float(
            df["motion_mean"].mean()
        )


        # ====================================================
        # 11. TRANSIÇÕES ENTRE ESTÁGIOS
        # ====================================================

        transicoes = 0

        possiveis_despertares = 0


        for i in range(
            1,
            len(df)
        ):

            epoca_anterior = int(
                df.iloc[
                    i - 1
                ]["epoca"]
            )

            epoca_atual = int(
                df.iloc[
                    i
                ]["epoca"]
            )


            # -----------------------------------------------
            # IMPORTANTE
            # -----------------------------------------------
            #
            # Algumas épocas foram removidas pelo
            # controle de qualidade.
            #
            # Não consideramos uma transição quando
            # existe um buraco entre elas.
            #
            # Exemplo:
            #
            # época 100 -> N2
            # épocas 101 a 105 removidas
            # época 106 -> REM
            #
            # Não sabemos exatamente quando aconteceu
            # a mudança.
            # -----------------------------------------------

            if (
                epoca_atual
                != epoca_anterior + 1
            ):

                continue


            label_anterior = int(
                df.iloc[
                    i - 1
                ]["label"]
            )

            label_atual = int(
                df.iloc[
                    i
                ]["label"]
            )


            # -----------------------------------------------
            # TRANSIÇÃO
            # -----------------------------------------------

            if (
                label_atual
                != label_anterior
            ):

                transicoes += 1


            # -----------------------------------------------
            # POSSÍVEL DESPERTAR
            # -----------------------------------------------
            #
            # Estamos chamando de possível despertar
            # uma entrada em Wake após estar em algum
            # estágio de sono.
            #
            # Não é um diagnóstico clínico de arousal.
            # -----------------------------------------------

            if (
                label_anterior != 0
                and
                label_atual == 0
            ):

                possiveis_despertares += 1


        # ====================================================
        # 12. ÍNDICE EXPERIMENTAL
        # ====================================================

        # Ainda NÃO calcularemos o índice.
        #
        # A fórmula será redefinida depois,
        # utilizando a distribuição das
        # 75 noites disponíveis.

        indice_experimental = None


        # ====================================================
        # 13. SALVAR NO MYSQL
        # ====================================================

        cursor.execute(

            sql_metricas,

            (
                id_noite,

                round(
                    duracao_analisada_min,
                    2
                ),

                round(
                    tempo_sono_min,
                    2
                ),

                round(
                    wake_pct,
                    4
                ),

                round(
                    n1_pct,
                    4
                ),

                round(
                    n2_pct,
                    4
                ),

                round(
                    n3_pct,
                    4
                ),

                round(
                    rem_pct,
                    4
                ),

                round(
                    hr_media,
                    4
                ),

                movimento_medio,

                transicoes,

                possiveis_despertares,

                indice_experimental
            )
        )


        contador += 1


        print(

            f"[{contador:02d}/75] "

            f"{participante} / "
            f"noite {numero_noite} "

            f"-> "

            f"{duracao_analisada_min:.1f} min | "

            f"sono: "
            f"{tempo_sono_min:.1f} min | "

            f"despertares: "
            f"{possiveis_despertares}"
        )


    # ========================================================
    # 14. COMMIT
    # ========================================================

    conexao.commit()


    # ========================================================
    # 15. VALIDAR
    # ========================================================

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM metricas_noite;
        """
    )


    total_metricas = (
        cursor.fetchone()[0]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "MÉTRICAS FINALIZADAS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nNoites esperadas: "
        f"75"
    )


    print(
        f"Métricas no banco: "
        f"{total_metricas}"
    )


    if total_metricas == 75:

        print(
            "\n✅ MÉTRICAS DAS 75 NOITES "
            "CARREGADAS CORRETAMENTE"
        )

    else:

        print(
            "\n⚠️ A QUANTIDADE DE MÉTRICAS "
            "NÃO É A ESPERADA"
        )


# ============================================================
# ERRO
# ============================================================

except Exception as erro:

    print(
        "\n❌ ERRO:"
    )

    print(
        erro
    )


    if conexao is not None:

        conexao.rollback()


# ============================================================
# ENCERRAR
# ============================================================

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