from getpass import getpass

import mysql.connector
from mysql.connector import Error


# ============================================================
# CONFIGURAÇÃO DO BANCO
# ============================================================

HOST = "127.0.0.1"
PORTA = 3306
BANCO = "projeto_sono"
USUARIO = "projeto_sono_app"


# ============================================================
# INFORMAÇÕES DO MODELO FINAL
# ============================================================

NOME_MODELO = "Ensemble RF + BiGRU + Viterbi"

ALGORITMO = "Random Forest + BiGRU + Viterbi"

VERSAO = "1.0"


# ============================================================
# RESULTADOS DO TESTE FINAL
# ============================================================

ACCURACY = 0.6512
BALANCED_ACCURACY = 0.6015
MACRO_F1 = 0.5954
WEIGHTED_F1 = 0.6561


# ============================================================
# CONFIGURAÇÃO DEFINITIVA
# ============================================================

PESO_RF = 0.75
PESO_BIGRU = 0.25
PESO_VITERBI = 0.60

JANELA_BIGRU_EPOCAS = 21
JANELA_BIGRU_MINUTOS = 10.5

EPOCAS_TREINAMENTO_BIGRU = 8

QUANTIDADE_TESTE = 12061


# ============================================================
# OBSERVAÇÕES
# ============================================================

OBSERVACOES = (
    "Modelo final do sistema experimental de classificação "
    "de estágios do sono. "
    "Ensemble composto por 75% Random Forest e 25% BiGRU, "
    "seguido de pós-processamento sequencial com algoritmo "
    "de Viterbi utilizando peso de transição 0.60. "
    "A BiGRU utiliza janela de 21 épocas de 30 segundos, "
    "equivalente a aproximadamente 10.5 minutos de contexto, "
    "com hidden_size=64 e treinamento final durante 8 épocas. "
    "O Random Forest utiliza 300 árvores, max_depth=16, "
    "min_samples_split=10, min_samples_leaf=5, "
    "max_features=sqrt e class_weight=balanced_subsample. "
    "A configuração foi selecionada durante a etapa de "
    "desenvolvimento utilizando validação cruzada por participante. "
    "O teste final independente utilizou Bidslab16, Bidslab22 "
    "e Bidslab32, totalizando 12061 épocas. "
    "Resultados do teste final: "
    "Accuracy=0.6512; "
    "Balanced Accuracy=0.6015; "
    "Macro F1=0.5954; "
    "Weighted F1=0.6561. "
    "Classes consideradas: Wake, N1, N2, N3 e REM. "
    "Resultado experimental e não destinado a diagnóstico médico."
)


# ============================================================
# CABEÇALHO
# ============================================================

print("=" * 90)
print("REGISTRO DO MODELO FINAL NO MYSQL - V2")
print("=" * 90)

print(
    f"\nBanco: {BANCO}"
)

print(
    f"Usuário: {USUARIO}"
)

print(
    f"Host: {HOST}:{PORTA}"
)


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
            "Não foi possível estabelecer conexão com o MySQL."
        )


    print(
        "✅ Conexão realizada."
    )


    cursor = conexao.cursor(
        dictionary=True
    )


    # ========================================================
    # VALIDAR TABELA
    # ========================================================

    print(
        "\nVerificando tabela modelos_ml..."
    )


    cursor.execute(
        """
        SHOW TABLES LIKE 'modelos_ml'
        """
    )


    tabela = cursor.fetchone()


    if tabela is None:

        raise RuntimeError(
            "A tabela modelos_ml não foi encontrada."
        )


    print(
        "✅ Tabela modelos_ml encontrada."
    )


    # ========================================================
    # VERIFICAR SE JÁ EXISTE
    # ========================================================

    print(
        "\nProcurando modelo e versão..."
    )


    cursor.execute(
        """
        SELECT
            id_modelo,
            nome,
            algoritmo,
            versao,
            data_treinamento,
            accuracy,
            macro_f1,
            observacoes
        FROM modelos_ml
        WHERE nome = %s
          AND versao = %s
        """,
        (
            NOME_MODELO,
            VERSAO
        )
    )


    modelo_existente = cursor.fetchone()


    # ========================================================
    # MODELO JÁ EXISTE
    # ========================================================

    if modelo_existente is not None:

        id_modelo = int(
            modelo_existente[
                "id_modelo"
            ]
        )


        print(
            "\n⚠️ O modelo já está cadastrado."
        )


        print(
            f"ID encontrado: {id_modelo}"
        )


        print(
            "Atualizando o registro existente..."
        )


        cursor.execute(
            """
            UPDATE modelos_ml
            SET
                algoritmo = %s,
                accuracy = %s,
                macro_f1 = %s,
                observacoes = %s
            WHERE id_modelo = %s
            """,
            (
                ALGORITMO,
                ACCURACY,
                MACRO_F1,
                OBSERVACOES,
                id_modelo
            )
        )


        conexao.commit()


        print(
            "✅ Registro existente atualizado."
        )


    # ========================================================
    # MODELO AINDA NÃO EXISTE
    # ========================================================

    else:

        print(
            "\nModelo ainda não está cadastrado."
        )


        print(
            "Inserindo modelo final..."
        )


        cursor.execute(
            """
            INSERT INTO modelos_ml
            (
                nome,
                algoritmo,
                versao,
                data_treinamento,
                accuracy,
                macro_f1,
                observacoes
            )
            VALUES
            (
                %s,
                %s,
                %s,
                NOW(),
                %s,
                %s,
                %s
            )
            """,
            (
                NOME_MODELO,
                ALGORITMO,
                VERSAO,
                ACCURACY,
                MACRO_F1,
                OBSERVACOES
            )
        )


        id_modelo = int(
            cursor.lastrowid
        )


        conexao.commit()


        print(
            "✅ Modelo inserido."
        )


    # ========================================================
    # CONSULTAR REGISTRO FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "VALIDANDO REGISTRO"
    )

    print(
        "=" * 90
    )


    cursor.execute(
        """
        SELECT
            id_modelo,
            nome,
            algoritmo,
            versao,
            data_treinamento,
            accuracy,
            macro_f1,
            observacoes
        FROM modelos_ml
        WHERE id_modelo = %s
        """,
        (
            id_modelo,
        )
    )


    modelo_final = cursor.fetchone()


    if modelo_final is None:

        raise RuntimeError(
            "O registro não foi encontrado após a gravação."
        )


    # ========================================================
    # EXIBIR DADOS
    # ========================================================

    print(
        f"\nID do modelo: "
        f"{modelo_final['id_modelo']}"
    )


    print(
        f"Nome: "
        f"{modelo_final['nome']}"
    )


    print(
        f"Algoritmo: "
        f"{modelo_final['algoritmo']}"
    )


    print(
        f"Versão: "
        f"{modelo_final['versao']}"
    )


    print(
        f"Data de treinamento/registro: "
        f"{modelo_final['data_treinamento']}"
    )


    accuracy_banco = float(
        modelo_final[
            "accuracy"
        ]
    )


    macro_banco = float(
        modelo_final[
            "macro_f1"
        ]
    )


    print(
        f"Accuracy: "
        f"{accuracy_banco:.4f}"
    )


    print(
        f"Macro F1: "
        f"{macro_banco:.4f}"
    )


    # ========================================================
    # CONFERIR MÉTRICAS
    # ========================================================

    if abs(
        accuracy_banco - ACCURACY
    ) > 0.000001:

        raise RuntimeError(
            "A Accuracy gravada não corresponde "
            "ao valor esperado."
        )


    if abs(
        macro_banco - MACRO_F1
    ) > 0.000001:

        raise RuntimeError(
            "O Macro F1 gravado não corresponde "
            "ao valor esperado."
        )


    # ========================================================
    # VERIFICAR DUPLICIDADE
    # ========================================================

    cursor.execute(
        """
        SELECT
            COUNT(*) AS quantidade
        FROM modelos_ml
        WHERE nome = %s
          AND versao = %s
        """,
        (
            NOME_MODELO,
            VERSAO
        )
    )


    quantidade_mesma_versao = int(
        cursor.fetchone()[
            "quantidade"
        ]
    )


    if quantidade_mesma_versao != 1:

        raise RuntimeError(
            "Foi encontrada uma quantidade inesperada "
            "de registros para esta versão."
        )


    # ========================================================
    # TOTAL DE MODELOS
    # ========================================================

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM modelos_ml
        """
    )


    total_modelos = int(
        cursor.fetchone()[
            "total"
        ]
    )


    # ========================================================
    # VALIDAR CHAVES DOS ESTÁGIOS
    # ========================================================

    print(
        "\nVerificando estágios do sono..."
    )


    cursor.execute(
        """
        SELECT
            id_estagio,
            codigo,
            nome
        FROM estagios_sono
        ORDER BY id_estagio
        """
    )


    estagios = cursor.fetchall()


    print(
        "\nEstágios cadastrados:"
    )


    for estagio in estagios:

        print(
            f"  {estagio['id_estagio']} "
            f"-> {estagio['codigo']} "
            f"({estagio['nome']})"
        )


    ids_estagios = {
        int(
            estagio[
                "id_estagio"
            ]
        )
        for estagio in estagios
    }


    ids_esperados = {
        0,
        1,
        2,
        3,
        4
    }


    if ids_estagios != ids_esperados:

        raise RuntimeError(
            "Os IDs de estagios_sono não correspondem "
            "ao esperado {0, 1, 2, 3, 4}."
        )


    print(
        "✅ IDs dos cinco estágios conferidos."
    )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MODELO FINAL REGISTRADO COM SUCESSO"
    )

    print(
        "=" * 90
    )


    print(
        f"\nID DO MODELO FINAL: "
        f"{id_modelo}"
    )


    print(
        f"Modelos cadastrados no banco: "
        f"{total_modelos}"
    )


    print(
        "\nMétricas finais:"
    )


    print(
        f"  Accuracy: "
        f"{ACCURACY:.4f}"
    )


    print(
        f"  Balanced Accuracy: "
        f"{BALANCED_ACCURACY:.4f}"
    )


    print(
        f"  Macro F1: "
        f"{MACRO_F1:.4f}"
    )


    print(
        f"  Weighted F1: "
        f"{WEIGHTED_F1:.4f}"
    )


    print(
        "\nConfiguração:"
    )


    print(
        f"  Random Forest: "
        f"{PESO_RF:.2f}"
    )


    print(
        f"  BiGRU: "
        f"{PESO_BIGRU:.2f}"
    )


    print(
        f"  Viterbi: "
        f"{PESO_VITERBI:.2f}"
    )


    print(
        "\n✅ Modelo registrado."
    )

    print(
        "✅ Métricas conferidas."
    )

    print(
        "✅ Estágios conferidos."
    )

    print(
        "✅ Nenhuma duplicidade criada."
    )


    print(
        "\nPRÓXIMA ETAPA:"
    )


    print(
        "Inserir as 12.061 predições finais "
        "na tabela predicoes."
    )


except Error as erro:

    if (
        conexao is not None
        and conexao.is_connected()
    ):

        conexao.rollback()


    print(
        "\n❌ ERRO DO MYSQL:"
    )

    print(
        erro
    )


except Exception as erro:

    if (
        conexao is not None
        and conexao.is_connected()
    ):

        conexao.rollback()


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