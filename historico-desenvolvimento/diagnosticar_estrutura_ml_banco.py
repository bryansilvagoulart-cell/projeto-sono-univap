from getpass import getpass

import mysql.connector
from mysql.connector import Error


# ============================================================
# CONFIGURAÇÃO
# ============================================================

HOST = "127.0.0.1"
PORTA = 3306
BANCO = "projeto_sono"
USUARIO = "projeto_sono_app"


# ============================================================
# TABELAS QUE QUEREMOS CONFERIR
# ============================================================

TABELAS = [
    "modelos_ml",
    "predicoes",
    "epocas",
    "estagios_sono"
]


# ============================================================
# INÍCIO
# ============================================================

print("=" * 90)
print("DIAGNÓSTICO DA ESTRUTURA DO BANCO - ML")
print("=" * 90)

print(f"\nBanco: {BANCO}")
print(f"Usuário: {USUARIO}")
print(f"Host: {HOST}:{PORTA}")


# ============================================================
# SENHA
# ============================================================

senha = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


conexao = None
cursor = None


try:

    # ========================================================
    # CONECTAR
    # ========================================================

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
            "Não foi possível estabelecer a conexão."
        )


    print(
        "✅ Conexão realizada."
    )


    cursor = conexao.cursor(
        dictionary=True
    )


    # ========================================================
    # BANCO ATUAL
    # ========================================================

    cursor.execute(
        "SELECT DATABASE() AS banco_atual"
    )


    banco_atual = cursor.fetchone()


    print(
        f"\nBanco atualmente selecionado: "
        f"{banco_atual['banco_atual']}"
    )


    # ========================================================
    # LISTAR TABELAS
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TABELAS EXISTENTES"
    )

    print(
        "=" * 90
    )


    cursor.execute(
        "SHOW TABLES"
    )


    tabelas_existentes_raw = (
        cursor.fetchall()
    )


    tabelas_existentes = []


    for linha in tabelas_existentes_raw:

        nome = list(
            linha.values()
        )[0]

        tabelas_existentes.append(
            nome
        )

        print(
            f"  - {nome}"
        )


    # ========================================================
    # ESTRUTURA DAS TABELAS IMPORTANTES
    # ========================================================

    for tabela in TABELAS:


        print(
            "\n"
            + "=" * 90
        )

        print(
            f"ESTRUTURA: {tabela}"
        )

        print(
            "=" * 90
        )


        if tabela not in tabelas_existentes:

            print(
                "❌ Tabela não encontrada."
            )

            continue


        # ----------------------------------------------------
        # COLUNAS
        # ----------------------------------------------------

        cursor.execute(
            f"SHOW COLUMNS FROM `{tabela}`"
        )


        colunas = cursor.fetchall()


        print(
            "\nCOLUNAS:"
        )


        for coluna in colunas:

            print(
                f"  {coluna['Field']:<30} "
                f"{coluna['Type']:<25} "
                f"Null={coluna['Null']:<3} "
                f"Key={coluna['Key']:<3} "
                f"Default={coluna['Default']} "
                f"Extra={coluna['Extra']}"
            )


        # ----------------------------------------------------
        # CHAVES E ÍNDICES
        # ----------------------------------------------------

        cursor.execute(
            f"SHOW INDEX FROM `{tabela}`"
        )


        indices = cursor.fetchall()


        print(
            "\nÍNDICES / CHAVES:"
        )


        for indice in indices:

            print(
                f"  Key_name={indice['Key_name']:<25} "
                f"Column={indice['Column_name']:<25} "
                f"Unique={0 if indice['Non_unique'] else 1}"
            )


        # ----------------------------------------------------
        # CREATE TABLE COMPLETO
        # ----------------------------------------------------

        cursor.execute(
            f"SHOW CREATE TABLE `{tabela}`"
        )


        create_table = cursor.fetchone()


        print(
            "\nCREATE TABLE:"
        )


        print(
            create_table[
                "Create Table"
            ]
        )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DIAGNÓSTICO CONCLUÍDO"
    )

    print(
        "=" * 90
    )


    print(
        "\n✅ Nenhuma informação foi alterada."
    )

    print(
        "✅ Apenas consultas de estrutura foram executadas."
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