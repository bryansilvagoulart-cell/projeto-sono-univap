from pathlib import Path
from getpass import getpass

import pandas as pd
import mysql.connector


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_geral_epocas.csv"
)

TAMANHO_LOTE = 1000


# ============================================================
# INÍCIO
# ============================================================

print("=" * 70)
print("CARGA DAS ÉPOCAS NO MYSQL")
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
    # 2. CARREGAR DATASET
    # ========================================================

    print(
        "\nCarregando dataset_geral_epocas.csv..."
    )


    dataset = pd.read_csv(
        ARQUIVO_DATASET
    )


    print(
        f"Linhas encontradas: "
        f"{len(dataset)}"
    )


    # ========================================================
    # 3. VALIDAR QUANTIDADE
    # ========================================================

    if len(dataset) != 60165:

        raise ValueError(

            f"O dataset possui "
            f"{len(dataset)} linhas, "
            f"mas eram esperadas 60165."
        )


    # ========================================================
    # 4. PEGAR OS IDs DAS NOITES
    # ========================================================

    print(
        "\nCarregando relacionamento "
        "participante -> noite..."
    )


    cursor.execute(
        """
        SELECT
            n.id_noite,
            p.codigo,
            n.numero_noite

        FROM noites n

        INNER JOIN participantes p
            ON p.id_participante =
               n.id_participante;
        """
    )


    resultados_noites = (
        cursor.fetchall()
    )


    # Exemplo:
    #
    # ("Bidslab02", 1) -> 1
    # ("Bidslab02", 2) -> 2

    mapa_noites = {

        (
            codigo,
            int(numero_noite)
        ):
        id_noite

        for (
            id_noite,
            codigo,
            numero_noite
        )

        in resultados_noites
    }


    print(
        f"Noites encontradas no MySQL: "
        f"{len(mapa_noites)}"
    )


    if len(mapa_noites) != 75:

        raise ValueError(
            "O banco deveria possuir "
            "75 noites."
        )


    # ========================================================
    # 5. CONVERTER DATAS
    # ========================================================

    print(
        "\nConvertendo datas..."
    )


    dataset["inicio"] = pd.to_datetime(
        dataset["inicio"],
        utc=True
    )


    dataset["fim"] = pd.to_datetime(
        dataset["fim"],
        utc=True
    )


    # ========================================================
    # 6. SQL DE INSERÇÃO
    # ========================================================

    sql_epoca = """
        INSERT INTO epocas (

            id_noite,
            numero_epoca,

            inicio_utc,
            fim_utc,

            hr_mean,
            hr_min,
            hr_max,
            hr_std,
            hr_samples,

            motion_mean,
            motion_std,
            motion_max,
            motion_samples,

            duracao_coberta,
            cobertura_temporal_pct,
            maior_gap,

            id_estagio_referencia

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
            %s,

            %s,
            %s,
            %s,

            %s
        )

        ON DUPLICATE KEY UPDATE

            inicio_utc =
                VALUES(inicio_utc),

            fim_utc =
                VALUES(fim_utc),

            hr_mean =
                VALUES(hr_mean),

            hr_min =
                VALUES(hr_min),

            hr_max =
                VALUES(hr_max),

            hr_std =
                VALUES(hr_std),

            hr_samples =
                VALUES(hr_samples),

            motion_mean =
                VALUES(motion_mean),

            motion_std =
                VALUES(motion_std),

            motion_max =
                VALUES(motion_max),

            motion_samples =
                VALUES(motion_samples),

            duracao_coberta =
                VALUES(duracao_coberta),

            cobertura_temporal_pct =
                VALUES(cobertura_temporal_pct),

            maior_gap =
                VALUES(maior_gap),

            id_estagio_referencia =
                VALUES(id_estagio_referencia);
    """


    # ========================================================
    # 7. PREPARAR DADOS
    # ========================================================

    print(
        "\nPreparando registros..."
    )


    registros = []


    for _, linha in dataset.iterrows():

        participante = (
            linha["participante"]
        )


        numero_noite = int(
            linha["noite"]
        )


        chave_noite = (
            participante,
            numero_noite
        )


        if chave_noite not in mapa_noites:

            raise ValueError(

                f"Noite não encontrada no banco: "
                f"{participante} / "
                f"noite {numero_noite}"
            )


        id_noite = (
            mapa_noites[
                chave_noite
            ]
        )


        # ----------------------------------------------------
        # DATETIME
        # ----------------------------------------------------

        inicio_mysql = (

            linha["inicio"]

            .to_pydatetime()

            .replace(
                tzinfo=None
            )
        )


        fim_mysql = (

            linha["fim"]

            .to_pydatetime()

            .replace(
                tzinfo=None
            )
        )


        # ----------------------------------------------------
        # REGISTRO
        # ----------------------------------------------------

        registro = (

            int(id_noite),

            int(
                linha["epoca"]
            ),

            inicio_mysql,
            fim_mysql,

            float(
                linha["hr_mean"]
            ),

            float(
                linha["hr_min"]
            ),

            float(
                linha["hr_max"]
            ),

            float(
                linha["hr_std"]
            ),

            int(
                linha["hr_samples"]
            ),

            float(
                linha["motion_mean"]
            ),

            float(
                linha["motion_std"]
            ),

            float(
                linha["motion_max"]
            ),

            int(
                linha["motion_samples"]
            ),

            float(
                linha["duracao_coberta"]
            ),

            float(
                linha[
                    "cobertura_temporal_pct"
                ]
            ),

            float(
                linha["maior_gap"]
            ),

            int(
                linha["label"]
            )
        )


        registros.append(
            registro
        )


    print(
        f"Registros preparados: "
        f"{len(registros)}"
    )


    # ========================================================
    # 8. INSERIR EM LOTES
    # ========================================================

    print(
        "\nInserindo épocas no MySQL..."
    )


    total = len(
        registros
    )


    inseridos = 0


    for inicio in range(
        0,
        total,
        TAMANHO_LOTE
    ):

        fim = min(

            inicio
            + TAMANHO_LOTE,

            total
        )


        lote = registros[
            inicio:fim
        ]


        cursor.executemany(
            sql_epoca,
            lote
        )


        conexao.commit()


        inseridos = fim


        percentual = (

            inseridos

            / total

            * 100
        )


        print(

            f"{inseridos:5d}"
            f"/{total} "

            f"({percentual:6.2f}%)"
        )


    # ========================================================
    # 9. VALIDAR TOTAL NO BANCO
    # ========================================================

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM epocas;
        """
    )


    total_banco = (
        cursor.fetchone()[0]
    )


    # ========================================================
    # 10. VALIDAR DISTRIBUIÇÃO DOS ESTÁGIOS
    # ========================================================

    cursor.execute(
        """
        SELECT

            e.codigo,

            COUNT(*) AS quantidade

        FROM epocas ep

        INNER JOIN estagios_sono e
            ON e.id_estagio =
               ep.id_estagio_referencia

        GROUP BY
            e.id_estagio,
            e.codigo

        ORDER BY
            e.id_estagio;
        """
    )


    distribuicao = (
        cursor.fetchall()
    )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CARGA DAS ÉPOCAS FINALIZADA"
    )

    print(
        "=" * 70
    )


    print(
        f"\nÉpocas esperadas: "
        f"60165"
    )


    print(
        f"Épocas no banco: "
        f"{total_banco}"
    )


    print(
        "\nDistribuição dos estágios:"
    )


    for (
        codigo,
        quantidade
    ) in distribuicao:

        print(
            f"{codigo:5s} "
            f"-> "
            f"{quantidade}"
        )


    # ========================================================
    # VALIDAÇÃO FINAL
    # ========================================================

    if total_banco == 60165:

        print(
            "\n✅ TODAS AS 60.165 ÉPOCAS "
            "FORAM CARREGADAS CORRETAMENTE"
        )

    else:

        print(
            "\n⚠️ O TOTAL DE ÉPOCAS "
            "NÃO É O ESPERADO"
        )


# ============================================================
# ERRO
# ============================================================

except Exception as erro:

    print(
        "\n❌ ERRO DURANTE A CARGA:"
    )

    print(
        erro
    )


    if conexao is not None:

        conexao.rollback()

        print(
            "\nO lote atual foi revertido."
        )


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