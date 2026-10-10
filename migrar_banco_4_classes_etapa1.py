from getpass import getpass
from pathlib import Path

import mysql.connector
from mysql.connector import Error


# ============================================================
# CONFIGURAÇÃO
# ============================================================

HOST = "127.0.0.1"
PORT = 3306
DATABASE = "projeto_sono"

TAXONOMIA_4_CLASSES = "sono_4_classes_v1"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def coluna_existe(cursor, tabela, coluna):

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (
            DATABASE,
            tabela,
            coluna
        )
    )

    return cursor.fetchone()[0] > 0


def tabela_existe(cursor, tabela):

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
        """,
        (
            DATABASE,
            tabela
        )
    )

    return cursor.fetchone()[0] > 0


def obter_column_type(
    cursor,
    tabela,
    coluna
):

    cursor.execute(
        """
        SELECT COLUMN_TYPE
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (
            DATABASE,
            tabela,
            coluna
        )
    )

    resultado = cursor.fetchone()

    if resultado is None:

        raise RuntimeError(
            f"Não encontrei {tabela}.{coluna}."
        )

    return resultado[0]


def adicionar_coluna_se_necessario(
    cursor,
    tabela,
    coluna,
    definicao
):

    if coluna_existe(
        cursor,
        tabela,
        coluna
    ):

        print(
            f"  ✅ {tabela}.{coluna} já existe."
        )

        return


    sql = (
        f"ALTER TABLE `{tabela}` "
        f"ADD COLUMN `{coluna}` {definicao}"
    )


    cursor.execute(
        sql
    )


    print(
        f"  ✅ Criada coluna "
        f"{tabela}.{coluna}."
    )


# ============================================================
# CONEXÃO
# ============================================================

print(
    "=" * 100
)

print(
    "MIGRAÇÃO DO BANCO - ETAPA 1 - SUPORTE À TAXONOMIA DE 4 CLASSES"
)

print(
    "=" * 100
)


print(
    "\nEsta etapa NÃO apaga nem altera "
    "as predições antigas de 5 classes."
)


print(
    "Ela apenas cria a estrutura necessária "
    "para o modelo novo."
)


usuario_admin = input(
    "\nUsuário administrador MySQL [root]: "
).strip()


if not usuario_admin:

    usuario_admin = "root"


senha_admin = getpass(
    f"Senha do usuário {usuario_admin}: "
)


conexao = None
cursor = None


try:

    print(
        "\nConectando ao MySQL..."
    )


    conexao = mysql.connector.connect(

        host=HOST,

        port=PORT,

        user=usuario_admin,

        password=senha_admin,

        database=DATABASE
    )


    cursor = conexao.cursor()


    print(
        "✅ Conexão administrativa realizada."
    )


    # ========================================================
    # VALIDAR TABELAS EXISTENTES
    # ========================================================

    print(
        "\nValidando estrutura atual..."
    )


    tabelas_obrigatorias = [
        "estagios_sono",
        "participantes",
        "noites",
        "epocas",
        "modelos_ml",
        "predicoes"
    ]


    faltando = [
        tabela
        for tabela in tabelas_obrigatorias
        if not tabela_existe(
            cursor,
            tabela
        )
    ]


    if faltando:

        raise RuntimeError(
            "Tabelas obrigatórias ausentes:\n"
            + "\n".join(
                faltando
            )
        )


    print(
        "✅ Estrutura antiga encontrada."
    )


    # ========================================================
    # MOSTRAR CONTAGENS ANTES
    # ========================================================

    print(
        "\nEstado atual:"
    )


    for tabela in [
        "participantes",
        "noites",
        "epocas",
        "modelos_ml",
        "predicoes"
    ]:

        cursor.execute(
            f"SELECT COUNT(*) FROM `{tabela}`"
        )

        quantidade = cursor.fetchone()[0]


        print(
            f"  {tabela:<18}: "
            f"{quantidade}"
        )


    # ========================================================
    # ATUALIZAR modelos_ml SEM QUEBRAR O ANTIGO
    # ========================================================

    print(
        "\nPreparando tabela modelos_ml..."
    )


    adicionar_coluna_se_necessario(
        cursor,
        "modelos_ml",
        "balanced_accuracy",
        "DOUBLE NULL"
    )


    adicionar_coluna_se_necessario(
        cursor,
        "modelos_ml",
        "weighted_f1",
        "DOUBLE NULL"
    )


    adicionar_coluna_se_necessario(
        cursor,
        "modelos_ml",
        "taxonomia",
        "VARCHAR(50) NULL"
    )


    adicionar_coluna_se_necessario(
        cursor,
        "modelos_ml",
        "quantidade_classes",
        "TINYINT UNSIGNED NULL"
    )


    # ========================================================
    # CRIAR CATÁLOGO DE CLASSES DO MODELO
    # ========================================================

    print(
        "\nCriando catálogo separado "
        "para classes do modelo..."
    )


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS classes_modelo_ml (

            id_classe_modelo
                TINYINT UNSIGNED
                NOT NULL
                AUTO_INCREMENT,

            taxonomia
                VARCHAR(50)
                NOT NULL,

            rotulo_modelo
                TINYINT UNSIGNED
                NOT NULL,

            codigo
                VARCHAR(20)
                NOT NULL,

            nome
                VARCHAR(60)
                NOT NULL,

            descricao
                VARCHAR(255)
                NULL,

            PRIMARY KEY (
                id_classe_modelo
            ),

            UNIQUE KEY uq_classes_modelo_taxonomia_rotulo (
                taxonomia,
                rotulo_modelo
            ),

            UNIQUE KEY uq_classes_modelo_taxonomia_codigo (
                taxonomia,
                codigo
            )
        )
        ENGINE=InnoDB
        DEFAULT CHARSET=utf8mb4
        COLLATE=utf8mb4_0900_ai_ci
        """
    )


    print(
        "✅ Tabela classes_modelo_ml pronta."
    )


    # ========================================================
    # INSERIR AS 4 CLASSES
    # ========================================================

    print(
        "\nRegistrando taxonomia de 4 classes..."
    )


    classes = [

        (
            TAXONOMIA_4_CLASSES,
            0,
            "WAKE",
            "Vigília",
            "Pessoa acordada."
        ),

        (
            TAXONOMIA_4_CLASSES,
            1,
            "LIGHT",
            "Sono leve",
            "Agrupamento dos estágios N1 e N2."
        ),

        (
            TAXONOMIA_4_CLASSES,
            2,
            "DEEP",
            "Sono profundo",
            "Corresponde ao estágio N3."
        ),

        (
            TAXONOMIA_4_CLASSES,
            3,
            "REM",
            "REM",
            "Estágio REM."
        )
    ]


    for registro in classes:

        cursor.execute(
            """
            INSERT INTO classes_modelo_ml (
                taxonomia,
                rotulo_modelo,
                codigo,
                nome,
                descricao
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s
            )
            ON DUPLICATE KEY UPDATE
                codigo = VALUES(codigo),
                nome = VALUES(nome),
                descricao = VALUES(descricao)
            """,
            registro
        )


    print(
        "✅ Wake / Light / Deep / REM registrados."
    )


    # ========================================================
    # DESCOBRIR TIPOS REAIS DAS CHAVES EXISTENTES
    # ========================================================

    tipo_id_epoca = obter_column_type(
        cursor,
        "epocas",
        "id_epoca"
    )


    tipo_id_modelo = obter_column_type(
        cursor,
        "modelos_ml",
        "id_modelo"
    )


    print(
        "\nTipos detectados:"
    )


    print(
        f"  epocas.id_epoca     = "
        f"{tipo_id_epoca}"
    )


    print(
        f"  modelos_ml.id_modelo = "
        f"{tipo_id_modelo}"
    )


    # ========================================================
    # CRIAR NOVA TABELA DE PREDIÇÕES
    # ========================================================
    #
    # Não usamos a tabela predicoes antiga porque ela aponta
    # para estagios_sono (Wake/N1/N2/N3/REM).
    #
    # Light precisa existir como classe própria do MODELO,
    # sem destruir a referência original.
    # ========================================================

    print(
        "\nCriando tabela de predições "
        "compatível com múltiplas taxonomias..."
    )


    sql_predicoes = f"""
        CREATE TABLE IF NOT EXISTS predicoes_modelo_ml (

            id_predicao
                BIGINT UNSIGNED
                NOT NULL
                AUTO_INCREMENT,

            id_epoca
                {tipo_id_epoca}
                NOT NULL,

            id_modelo
                {tipo_id_modelo}
                NOT NULL,

            id_classe_modelo
                TINYINT UNSIGNED
                NOT NULL,

            confianca
                DOUBLE
                NULL,

            criado_em
                TIMESTAMP
                NOT NULL
                DEFAULT CURRENT_TIMESTAMP,

            PRIMARY KEY (
                id_predicao
            ),

            UNIQUE KEY uq_predicoes_modelo_epoca_modelo (
                id_epoca,
                id_modelo
            ),

            KEY idx_predicoes_modelo_modelo (
                id_modelo
            ),

            KEY idx_predicoes_modelo_classe (
                id_classe_modelo
            ),

            CONSTRAINT fk_predicoes_modelo_epoca
                FOREIGN KEY (
                    id_epoca
                )
                REFERENCES epocas (
                    id_epoca
                )
                ON DELETE CASCADE
                ON UPDATE CASCADE,

            CONSTRAINT fk_predicoes_modelo_modelo
                FOREIGN KEY (
                    id_modelo
                )
                REFERENCES modelos_ml (
                    id_modelo
                )
                ON DELETE CASCADE
                ON UPDATE CASCADE,

            CONSTRAINT fk_predicoes_modelo_classe
                FOREIGN KEY (
                    id_classe_modelo
                )
                REFERENCES classes_modelo_ml (
                    id_classe_modelo
                )
                ON DELETE RESTRICT
                ON UPDATE CASCADE
        )
        ENGINE=InnoDB
        DEFAULT CHARSET=utf8mb4
        COLLATE=utf8mb4_0900_ai_ci
    """


    cursor.execute(
        sql_predicoes
    )


    print(
        "✅ Tabela predicoes_modelo_ml pronta."
    )


    # ========================================================
    # COMMIT
    # ========================================================

    conexao.commit()


    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "VALIDAÇÃO DA MIGRAÇÃO"
    )

    print(
        "=" * 100
    )


    cursor.execute(
        """
        SELECT
            id_classe_modelo,
            taxonomia,
            rotulo_modelo,
            codigo,
            nome
        FROM classes_modelo_ml
        WHERE taxonomia = %s
        ORDER BY rotulo_modelo
        """,
        (
            TAXONOMIA_4_CLASSES,
        )
    )


    registros = cursor.fetchall()


    if len(registros) != 4:

        raise RuntimeError(
            "Esperávamos exatamente "
            "4 classes registradas."
        )


    print(
        "\nClasses cadastradas:"
    )


    for (
        id_classe,
        taxonomia,
        rotulo,
        codigo,
        nome
    ) in registros:

        print(
            f"  id={id_classe} | "
            f"rótulo={rotulo} | "
            f"{codigo:<5} | "
            f"{nome}"
        )


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predicoes_modelo_ml
        """
    )


    quantidade_novas_predicoes = (
        cursor.fetchone()[0]
    )


    print(
        "\nPredições na nova tabela:"
    )


    print(
        f"  {quantidade_novas_predicoes}"
    )


    print(
        "\n✅ estagios_sono foi preservada."
    )


    print(
        "✅ predicoes antiga foi preservada."
    )


    print(
        "✅ classes_modelo_ml representa "
        "Wake / Light / Deep / REM separadamente."
    )


    print(
        "✅ predicoes_modelo_ml está pronta "
        "para receber o modelo final."
    )


    print(
        "\nPróxima etapa:"
    )


    print(
        "Importar Bidslab01, Bidslab35 e Bidslab47 "
        "com suas 15 noites e 12.252 épocas para o MySQL."
    )


except Error as erro:

    print(
        "\n❌ ERRO DO MYSQL:"
    )

    print(
        erro
    )


    if conexao is not None:

        try:

            conexao.rollback()

        except Exception:

            pass


except Exception as erro:

    print(
        "\n❌ ERRO:"
    )

    print(
        f"{type(erro).__name__}: "
        f"{erro}"
    )


    if conexao is not None:

        try:

            conexao.rollback()

        except Exception:

            pass


finally:

    if cursor is not None:

        cursor.close()


    if conexao is not None:

        conexao.close()


    print(
        "\nConexão com MySQL encerrada."
    )
