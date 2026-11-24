from getpass import getpass

import mysql.connector
from mysql.connector import Error


HOST = "127.0.0.1"
PORTA = 3306
BANCO = "projeto_sono"
USUARIO = "projeto_sono_app"

TABELAS = [
    "participantes",
    "noites",
    "metricas_noite"
]


print("=" * 95)
print("DIAGNÓSTICO DAS TABELAS PARA A API")
print("=" * 95)

print(f"\nBanco: {BANCO}")
print(f"Usuário: {USUARIO}")
print(f"Host: {HOST}:{PORTA}")


senha = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


conexao = None
cursor = None


try:

    print(
        "\nConectando ao MySQL..."
    )

    conexao = mysql.connector.connect(
        host=HOST,
        port=PORTA,
        user=USUARIO,
        password=senha,
        database=BANCO
    )

    if not conexao.is_connected():

        raise RuntimeError(
            "Não foi possível estabelecer conexão."
        )

    print(
        "✅ Conexão realizada."
    )


    cursor = conexao.cursor(
        dictionary=True
    )


    # ========================================================
    # ESTRUTURA DAS TABELAS
    # ========================================================

    for tabela in TABELAS:

        print(
            "\n"
            + "=" * 95
        )

        print(
            f"TABELA: {tabela}"
        )

        print(
            "=" * 95
        )


        cursor.execute(
            f"SHOW COLUMNS FROM `{tabela}`"
        )

        colunas = cursor.fetchall()


        print(
            "\nCOLUNAS:"
        )

        for coluna in colunas:

            print(
                f"  {coluna['Field']:<32} "
                f"{coluna['Type']:<25} "
                f"Null={coluna['Null']:<3} "
                f"Key={coluna['Key']:<3} "
                f"Default={coluna['Default']} "
                f"Extra={coluna['Extra']}"
            )


        cursor.execute(
            f"""
            SELECT COUNT(*) AS total
            FROM `{tabela}`
            """
        )

        total = int(
            cursor.fetchone()[
                "total"
            ]
        )


        print(
            f"\nQuantidade de registros: "
            f"{total}"
        )


        cursor.execute(
            f"""
            SELECT *
            FROM `{tabela}`
            LIMIT 5
            """
        )

        registros = cursor.fetchall()


        print(
            "\nPRIMEIROS REGISTROS:"
        )


        for numero, registro in enumerate(
            registros,
            start=1
        ):

            print(
                f"\n  Registro {numero}:"
            )

            for chave, valor in registro.items():

                print(
                    f"    {chave}: {valor}"
                )


    # ========================================================
    # PARTICIPANTES -> NOITES
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "PARTICIPANTES E QUANTIDADE DE NOITES"
    )

    print(
        "=" * 95
    )


    cursor.execute(
        """
        SELECT
            p.id_participante,
            p.codigo AS participante,
            COUNT(n.id_noite) AS quantidade_noites
        FROM participantes p

        LEFT JOIN noites n
            ON n.id_participante = p.id_participante

        GROUP BY
            p.id_participante,
            p.codigo

        ORDER BY
            p.codigo
        """
    )


    participantes = cursor.fetchall()


    print()


    for linha in participantes:

        print(
            f"  ID {linha['id_participante']:<3} "
            f"{linha['participante']:<15} "
            f"-> "
            f"{linha['quantidade_noites']} noites"
        )


    # ========================================================
    # NOITES -> MÉTRICAS
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "RELACIONAMENTO NOITES -> MÉTRICAS"
    )

    print(
        "=" * 95
    )


    cursor.execute(
        """
        SELECT
            COUNT(*) AS total
        FROM noites n

        INNER JOIN metricas_noite m
            ON m.id_noite = n.id_noite
        """
    )


    noites_com_metricas = int(
        cursor.fetchone()[
            "total"
        ]
    )


    print(
        f"\nNoites com métricas: "
        f"{noites_com_metricas}"
    )


    # ========================================================
    # PARTICIPANTES DO TESTE FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "PARTICIPANTES DO TESTE FINAL"
    )

    print(
        "=" * 95
    )


    cursor.execute(
        """
        SELECT
            p.id_participante,
            p.codigo,
            COUNT(DISTINCT n.id_noite) AS noites,
            COUNT(ep.id_epoca) AS epocas
        FROM participantes p

        LEFT JOIN noites n
            ON n.id_participante = p.id_participante

        LEFT JOIN epocas ep
            ON ep.id_noite = n.id_noite

        WHERE p.codigo IN (
            'Bidslab16',
            'Bidslab22',
            'Bidslab32'
        )

        GROUP BY
            p.id_participante,
            p.codigo

        ORDER BY
            p.codigo
        """
    )


    teste_final = cursor.fetchall()


    print()


    for linha in teste_final:

        print(
            f"  {linha['codigo']:<12} "
            f"| ID {linha['id_participante']:<3} "
            f"| noites {linha['noites']:<2} "
            f"| épocas {linha['epocas']}"
        )


    # ========================================================
    # PREDIÇÕES DO MODELO FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "PREDIÇÕES DO MODELO FINAL"
    )

    print(
        "=" * 95
    )


    cursor.execute(
        """
        SELECT
            COUNT(*) AS total
        FROM predicoes
        WHERE id_modelo = 1
        """
    )


    total_predicoes = int(
        cursor.fetchone()[
            "total"
        ]
    )


    print(
        f"\nPredições do modelo 1: "
        f"{total_predicoes}"
    )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "DIAGNÓSTICO CONCLUÍDO"
    )

    print(
        "=" * 95
    )


    print(
        "\n✅ Nenhuma informação foi alterada."
    )

    print(
        "✅ Banco analisado somente com consultas."
    )


except Error as erro:

    print(
        "\n❌ ERRO DO MYSQL:"
    )

    print(
        erro
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