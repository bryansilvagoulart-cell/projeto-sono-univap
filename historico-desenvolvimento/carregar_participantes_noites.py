from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from getpass import getpass

import pandas as pd
import mysql.connector
from scipy.io import loadmat


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_CONTROLE = Path(
    "data/processed/controle_qualidade_noites.csv"
)

ARQUIVO_DATASET = Path(
    "data/processed/dataset_geral_epocas.csv"
)

PASTA_RAW = Path(
    "data/raw/participantes"
)


# ============================================================
# CONEXÃO COM MYSQL
# ============================================================

print("=" * 70)
print("CARGA DE PARTICIPANTES E NOITES")
print("=" * 70)


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
        "\n✅ Conexão com o MySQL realizada."
    )


    # ========================================================
    # 1. CARREGAR ARQUIVOS
    # ========================================================

    print(
        "\nCarregando arquivos processados..."
    )


    controle = pd.read_csv(
        ARQUIVO_CONTROLE
    )


    dataset_final = pd.read_csv(
        ARQUIVO_DATASET,
        usecols=[
            "participante",
            "noite",
            "epoca"
        ]
    )


    print(
        f"Noites no controle de qualidade: "
        f"{len(controle)}"
    )


    print(
        f"Épocas no dataset final: "
        f"{len(dataset_final)}"
    )


    # ========================================================
    # 2. CONTAR ÉPOCAS FINAIS POR NOITE
    # ========================================================

    epocas_finais = (

        dataset_final

        .groupby(
            [
                "participante",
                "noite"
            ]
        )

        .size()

        .reset_index(
            name="epocas_validas_final"
        )

    )


    # ========================================================
    # 3. JUNTAR COM O CONTROLE DE QUALIDADE
    # ========================================================

    noites = controle.merge(

        epocas_finais,

        on=[
            "participante",
            "noite"
        ],

        how="left",

        validate="one_to_one"
    )


    noites[
        "epocas_validas_final"
    ] = (

        noites[
            "epocas_validas_final"
        ]

        .fillna(0)

        .astype(int)
    )


    # ========================================================
    # 4. PARTICIPANTES
    # ========================================================

    participantes = sorted(

        noites[
            "participante"
        ]

        .unique()
    )


    print(
        f"\nParticipantes encontrados: "
        f"{len(participantes)}"
    )


    # ========================================================
    # 5. INSERIR PARTICIPANTES
    # ========================================================

    print(
        "\nInserindo participantes..."
    )


    sql_participante = """
        INSERT INTO participantes (
            codigo
        )
        VALUES (%s)

        ON DUPLICATE KEY UPDATE
            codigo = VALUES(codigo);
    """


    for participante in participantes:

        cursor.execute(
            sql_participante,
            (
                participante,
            )
        )


    conexao.commit()


    print(
        f"✅ {len(participantes)} "
        f"participantes processados."
    )


    # ========================================================
    # 6. OBTER IDs DOS PARTICIPANTES
    # ========================================================

    cursor.execute(
        """
        SELECT
            id_participante,
            codigo
        FROM participantes;
        """
    )


    mapa_participantes = {

        codigo:
            id_participante

        for (
            id_participante,
            codigo
        )

        in cursor.fetchall()
    }


    # ========================================================
    # 7. INSERIR AS 75 NOITES
    # ========================================================

    print(
        "\nInserindo noites..."
    )


    sql_noite = """
        INSERT INTO noites (

            id_participante,
            numero_noite,

            inicio_utc,
            fim_utc,

            epocas_originais,
            epocas_unknown,

            epocas_validas,
            epocas_removidas,

            cobertura_percentual
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
            %s
        )

        ON DUPLICATE KEY UPDATE

            inicio_utc =
                VALUES(inicio_utc),

            fim_utc =
                VALUES(fim_utc),

            epocas_originais =
                VALUES(epocas_originais),

            epocas_unknown =
                VALUES(epocas_unknown),

            epocas_validas =
                VALUES(epocas_validas),

            epocas_removidas =
                VALUES(epocas_removidas),

            cobertura_percentual =
                VALUES(cobertura_percentual);
    """


    contador = 0


    for _, linha in noites.iterrows():

        participante = (
            linha[
                "participante"
            ]
        )


        numero_noite = int(
            linha[
                "noite"
            ]
        )


        id_participante = (
            mapa_participantes[
                participante
            ]
        )


        # ====================================================
        # 8. CARREGAR recStart ORIGINAL DA NOITE
        # ====================================================

        arquivo_labels = (

            PASTA_RAW

            / participante

            / f"noite_{numero_noite}"

            / "labels.mat"
        )


        dados_labels = loadmat(
            arquivo_labels
        )


        rec_start_texto = (

            dados_labels[
                "recStart"
            ]

            .item()
        )


        # ====================================================
        # 9. CONVERTER HORÁRIO PARA UTC
        # ====================================================

        inicio_local = datetime.strptime(

            rec_start_texto,

            "%Y-%m-%d %H:%M:%S"
        )


        inicio_local = (

            inicio_local

            .replace(

                tzinfo=ZoneInfo(
                    "America/New_York"
                )
            )
        )


        inicio_utc = (

            inicio_local

            .astimezone(
                ZoneInfo("UTC")
            )
        )


        # MySQL DATETIME não armazena timezone.
        # Portanto salvamos o valor já convertido
        # para UTC, mas sem tzinfo.

        inicio_mysql = (

            inicio_utc

            .replace(
                tzinfo=None
            )
        )


        # ====================================================
        # 10. CALCULAR FINAL DO PERÍODO ROTULADO
        # ====================================================

        epocas_originais = int(

            linha[
                "epocas_originais"
            ]
        )


        fim_mysql = (

            inicio_mysql

            + timedelta(

                seconds=(
                    epocas_originais
                    * 30
                )
            )
        )


        # ====================================================
        # 11. QUANTIDADES
        # ====================================================

        unknown = int(

            linha[
                "unknown"
            ]
        )


        validas = int(

            linha[
                "epocas_validas_final"
            ]
        )


        utilizaveis = (

            epocas_originais
            - unknown
        )


        removidas = (

            utilizaveis
            - validas
        )


        # ====================================================
        # 12. COBERTURA FINAL
        # ====================================================

        if utilizaveis > 0:

            cobertura = (

                validas

                / utilizaveis

                * 100
            )

        else:

            cobertura = 0


        # ====================================================
        # 13. INSERIR NO MYSQL
        # ====================================================

        cursor.execute(

            sql_noite,

            (
                id_participante,

                numero_noite,

                inicio_mysql,
                fim_mysql,

                epocas_originais,
                unknown,

                validas,
                removidas,

                round(
                    cobertura,
                    3
                )
            )
        )


        contador += 1


        print(

            f"[{contador:02d}/75] "

            f"{participante} / "

            f"noite {numero_noite} "

            f"-> {validas} épocas válidas"
        )


    # ========================================================
    # 14. COMMIT
    # ========================================================

    conexao.commit()


    # ========================================================
    # 15. VALIDAÇÃO
    # ========================================================

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM participantes;
        """
    )


    total_participantes = (
        cursor.fetchone()[0]
    )


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM noites;
        """
    )


    total_noites = (
        cursor.fetchone()[0]
    )


    cursor.execute(
        """
        SELECT SUM(epocas_validas)
        FROM noites;
        """
    )


    total_epocas = (
        cursor.fetchone()[0]
    )


    # ========================================================
    # RESULTADO
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CARGA FINALIZADA"
    )

    print(
        "=" * 70
    )


    print(
        f"\nParticipantes no banco: "
        f"{total_participantes}"
    )


    print(
        f"Noites no banco: "
        f"{total_noites}"
    )


    print(
        f"Soma das épocas válidas "
        f"das noites: "
        f"{total_epocas}"
    )


    if (
        total_participantes == 15
        and total_noites == 75
        and total_epocas == 60165
    ):

        print(
            "\n✅ PARTICIPANTES E NOITES "
            "CARREGADOS CORRETAMENTE"
        )

    else:

        print(
            "\n⚠️ OS TOTAIS NÃO BATEM "
            "COM O DATASET ESPERADO"
        )


# ============================================================
# ERROS
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
            "\nAlterações da execução "
            "foram revertidas."
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