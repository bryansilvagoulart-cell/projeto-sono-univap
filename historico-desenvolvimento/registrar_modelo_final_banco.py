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
# DADOS DEFINITIVOS DO MODELO
# ============================================================

NOME_MODELO = "Ensemble RF + BiGRU + Viterbi"

ALGORITMO = (
    "Random Forest + BiGRU + Viterbi"
)

VERSAO = "1.0"


# ============================================================
# MÉTRICAS DO TESTE FINAL
# ============================================================

ACCURACY = 0.6512
MACRO_F1 = 0.5954


# ============================================================
# OBSERVAÇÕES
# ============================================================

OBSERVACOES = (
    "Modelo final selecionado após validação cruzada por participante. "
    "Ensemble composto por 75% Random Forest e 25% BiGRU, "
    "seguido de pós-processamento sequencial com Viterbi "
    "usando peso de transição 0.60. "
    "Random Forest: 300 árvores, max_depth=16, "
    "min_samples_split=10, min_samples_leaf=5, "
    "max_features=sqrt e class_weight=balanced_subsample. "
    "BiGRU: janela temporal de 21 épocas (10.5 minutos), "
    "hidden_size=64 e treinamento final por 8 épocas. "
    "Teste final independente realizado com Bidslab16, "
    "Bidslab22 e Bidslab32, totalizando 12061 épocas. "
    "Accuracy final=0.6512; Balanced Accuracy=0.6015; "
    "Macro F1=0.5954; Weighted F1=0.6561. "
    "Classificação experimental dos estágios Wake, N1, N2, N3 e REM. "
    "Não se trata de diagnóstico médico."
)


# ============================================================
# CONEXÃO
# ============================================================

print("=" * 90)
print("REGISTRO DO MODELO FINAL NO MYSQL")
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
            "Não foi possível estabelecer conexão."
        )


    print(
        "✅ Conexão realizada."
    )


    cursor = conexao.cursor(
        dictionary=True
    )


    # ========================================================
    # VERIFICAR TABELA
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
            "A tabela modelos_ml não foi encontrada "
            "no banco projeto_sono."
        )


    print(
        "✅ Tabela modelos_ml encontrada."
    )


    # ========================================================
    # VERIFICAR SE MODELO JÁ EXISTE
    # ========================================================

    print(
        "\nProcurando versão já cadastrada..."
    )


    cursor.execute(

        """
        SELECT
            id,
            nome,
            algoritmo,
            versao,
            accuracy,
            macro_f1,
            data_treinamento,
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
    # SE JÁ EXISTE, ATUALIZAR
    # ========================================================

    if modelo_existente is not None:


        print(
            "\n⚠️ Esta versão já existe."
        )


        print(
            f"ID existente: "
            f"{modelo_existente['id']}"
        )


        print(
            "Atualizando os dados para garantir "
            "que correspondam ao resultado final..."
        )


        cursor.execute(

            """
            UPDATE modelos_ml
            SET
                algoritmo = %s,
                accuracy = %s,
                macro_f1 = %s,
                observacoes = %s
            WHERE id = %s
            """,

            (
                ALGORITMO,
                ACCURACY,
                MACRO_F1,
                OBSERVACOES,
                modelo_existente[
                    "id"
                ]
            )
        )


        id_modelo = int(
            modelo_existente[
                "id"
            ]
        )


        conexao.commit()


        print(
            "✅ Registro existente atualizado."
        )


    # ========================================================
    # SENÃO, INSERIR
    # ========================================================

    else:


        print(
            "\nModelo ainda não cadastrado."
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
    # CONSULTAR NOVAMENTE
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
            id,
            nome,
            algoritmo,
            versao,
            data_treinamento,
            accuracy,
            macro_f1,
            observacoes
        FROM modelos_ml
        WHERE id = %s
        """,

        (
            id_modelo,
        )
    )


    modelo_final = cursor.fetchone()


    if modelo_final is None:

        raise RuntimeError(
            "O modelo foi inserido, "
            "mas não foi localizado na validação."
        )


    # ========================================================
    # EXIBIR
    # ========================================================

    print(
        f"\nID: "
        f"{modelo_final['id']}"
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
        f"Data treinamento: "
        f"{modelo_final['data_treinamento']}"
    )


    print(
        f"Accuracy: "
        f"{float(modelo_final['accuracy']):.4f}"
    )


    print(
        f"Macro F1: "
        f"{float(modelo_final['macro_f1']):.4f}"
    )


    # ========================================================
    # VALIDAÇÃO DAS MÉTRICAS
    # ========================================================

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


    if abs(
        accuracy_banco - ACCURACY
    ) > 0.0001:

        raise RuntimeError(
            "Accuracy registrada no banco "
            "não corresponde ao resultado final."
        )


    if abs(
        macro_banco - MACRO_F1
    ) > 0.0001:

        raise RuntimeError(
            "Macro F1 registrada no banco "
            "não corresponde ao resultado final."
        )


    # ========================================================
    # QUANTIDADE TOTAL
    # ========================================================

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM modelos_ml
        """
    )


    total_modelos = cursor.fetchone()[
        "total"
    ]


    # ========================================================
    # FINAL
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
        "\n✅ Accuracy conferida."
    )


    print(
        "✅ Macro F1 conferido."
    )


    print(
        "✅ Configuração final documentada."
    )


    print(
        "\nPróxima etapa:"
    )


    print(
        "Associar as predições finais "
        "ao ID deste modelo."
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