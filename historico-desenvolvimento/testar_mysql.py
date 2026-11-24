import mysql.connector
from getpass import getpass


print("=" * 60)
print("TESTE DE CONEXÃO DA APLICAÇÃO -> MYSQL")
print("=" * 60)

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

    if conexao.is_connected():

        print(
            "\n✅ Conexão realizada com sucesso."
        )

        cursor = conexao.cursor()

        cursor.execute(
            "SELECT DATABASE();"
        )

        banco = cursor.fetchone()[0]

        print(
            f"Banco conectado: {banco}"
        )


        cursor.execute(
            """
            SELECT
                id_estagio,
                codigo,
                nome
            FROM estagios_sono
            ORDER BY id_estagio;
            """
        )

        resultados = cursor.fetchall()


        print(
            "\nEstágios encontrados:"
        )


        for (
            id_estagio,
            codigo,
            nome
        ) in resultados:

            print(
                f"{id_estagio} - "
                f"{codigo} - "
                f"{nome}"
            )


        print(
            f"\nQuantidade de estágios: "
            f"{len(resultados)}"
        )


except mysql.connector.Error as erro:

    print(
        "\n❌ Erro ao conectar ao MySQL:"
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
            "\nConexão encerrada."
        )