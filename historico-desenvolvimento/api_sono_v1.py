from getpass import getpass

import mysql.connector
from mysql.connector import Error

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn


# ============================================================
# CONFIGURAÇÃO DO BANCO
# ============================================================

HOST = "127.0.0.1"
PORTA = 3306
BANCO = "projeto_sono"
USUARIO = "projeto_sono_app"

ID_MODELO_FINAL = 1


# ============================================================
# SENHA
# ============================================================

print("=" * 90)
print("API - PROJETO SONO")
print("=" * 90)

print(f"\nBanco: {BANCO}")
print(f"Usuário: {USUARIO}")
print(f"Host: {HOST}:{PORTA}")


SENHA_BANCO = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(

    title="API - Sistema de Análise e Classificação do Sono",

    description=(
        "API REST do projeto de Engenharia da Computação "
        "para consulta de participantes, noites, métricas "
        "e classificações experimentais dos estágios do sono."
    ),

    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# FUNÇÃO DE CONEXÃO
# ============================================================

def conectar_banco():

    try:

        conexao = mysql.connector.connect(

            host=HOST,

            port=PORTA,

            user=USUARIO,

            password=SENHA_BANCO,

            database=BANCO
        )


        if not conexao.is_connected():

            raise RuntimeError(
                "Não foi possível estabelecer conexão com o MySQL."
            )


        return conexao


    except Error as erro:

        raise HTTPException(

            status_code=500,

            detail=(
                "Erro ao conectar ao banco de dados: "
                f"{erro}"
            )
        )


# ============================================================
# ROTA RAIZ
# ============================================================

@app.get("/")
def raiz():

    return {

        "sistema":
            "Sistema de Análise e Classificação de Estágios do Sono",

        "api":
            "online",

        "versao":
            "1.0.0",

        "documentacao":
            "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    conexao = None
    cursor = None


    try:

        conexao = conectar_banco()


        cursor = conexao.cursor(
            dictionary=True
        )


        # ----------------------------------------------------
        # PARTICIPANTES
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM participantes
            """
        )


        participantes = int(
            cursor.fetchone()[
                "total"
            ]
        )


        # ----------------------------------------------------
        # NOITES
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM noites
            """
        )


        noites = int(
            cursor.fetchone()[
                "total"
            ]
        )


        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM metricas_noite
            """
        )


        metricas = int(
            cursor.fetchone()[
                "total"
            ]
        )


        # ----------------------------------------------------
        # PREDIÇÕES DO MODELO FINAL
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predicoes
            WHERE id_modelo = %s
            """,
            (
                int(ID_MODELO_FINAL),
            )
        )


        predicoes = int(
            cursor.fetchone()[
                "total"
            ]
        )


        return {

            "status":
                "ok",

            "banco":
                BANCO,

            "participantes":
                participantes,

            "noites":
                noites,

            "metricas_noite":
                metricas,

            "predicoes_modelo_final":
                predicoes
        }


    finally:

        if cursor is not None:

            cursor.close()


        if (
            conexao is not None
            and conexao.is_connected()
        ):

            conexao.close()


# ============================================================
# MODELO FINAL
# ============================================================

@app.get("/modelo")
def obter_modelo():

    conexao = None
    cursor = None


    try:

        conexao = conectar_banco()


        cursor = conexao.cursor(
            dictionary=True
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
                int(ID_MODELO_FINAL),
            )
        )


        modelo = cursor.fetchone()


        if modelo is None:

            raise HTTPException(
                status_code=404,
                detail="Modelo final não encontrado."
            )


        # ----------------------------------------------------
        # QUANTIDADE DE PREDIÇÕES
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predicoes
            WHERE id_modelo = %s
            """,
            (
                int(ID_MODELO_FINAL),
            )
        )


        total_predicoes = int(
            cursor.fetchone()[
                "total"
            ]
        )


        return {

            "id_modelo":
                int(
                    modelo[
                        "id_modelo"
                    ]
                ),

            "nome":
                modelo[
                    "nome"
                ],

            "algoritmo":
                modelo[
                    "algoritmo"
                ],

            "versao":
                modelo[
                    "versao"
                ],

            "data_treinamento":
                modelo[
                    "data_treinamento"
                ],

            "metricas_teste_final": {

                "accuracy":
                    float(
                        modelo[
                            "accuracy"
                        ]
                    ),

                "macro_f1":
                    float(
                        modelo[
                            "macro_f1"
                        ]
                    ),

                "balanced_accuracy":
                    0.6015,

                "weighted_f1":
                    0.6561
            },

            "configuracao": {

                "random_forest":
                    0.75,

                "bigru":
                    0.25,

                "viterbi":
                    0.60,

                "janela_bigru_epocas":
                    21,

                "janela_bigru_minutos":
                    10.5
            },

            "predicoes_registradas":
                total_predicoes,

            "observacoes":
                modelo[
                    "observacoes"
                ]
        }


    finally:

        if cursor is not None:

            cursor.close()


        if (
            conexao is not None
            and conexao.is_connected()
        ):

            conexao.close()


# ============================================================
# PARTICIPANTES
# ============================================================

@app.get("/participantes")
def listar_participantes():

    conexao = None
    cursor = None


    try:

        conexao = conectar_banco()


        cursor = conexao.cursor(
            dictionary=True
        )


        cursor.execute(
            """
            SELECT
                p.id_participante,
                p.codigo,

                COUNT(
                    DISTINCT n.id_noite
                ) AS quantidade_noites,

                COUNT(
                    DISTINCT CASE
                        WHEN pr.id_modelo = %s
                        THEN ep.id_epoca
                    END
                ) AS epocas_com_predicao

            FROM participantes p

            LEFT JOIN noites n
                ON n.id_participante = p.id_participante

            LEFT JOIN epocas ep
                ON ep.id_noite = n.id_noite

            LEFT JOIN predicoes pr
                ON pr.id_epoca = ep.id_epoca
                AND pr.id_modelo = %s

            GROUP BY
                p.id_participante,
                p.codigo

            ORDER BY
                p.codigo
            """,
            (
                int(ID_MODELO_FINAL),
                int(ID_MODELO_FINAL)
            )
        )


        registros = cursor.fetchall()


        participantes = []


        for registro in registros:

            epocas_com_predicao = int(
                registro[
                    "epocas_com_predicao"
                ]
            )


            participantes.append({

                "id_participante":
                    int(
                        registro[
                            "id_participante"
                        ]
                    ),

                "codigo":
                    registro[
                        "codigo"
                    ],

                "quantidade_noites":
                    int(
                        registro[
                            "quantidade_noites"
                        ]
                    ),

                "epocas_com_predicao":
                    epocas_com_predicao,

                "possui_predicoes_modelo_final":
                    epocas_com_predicao > 0
            })


        return {

            "quantidade":
                len(
                    participantes
                ),

            "participantes":
                participantes
        }


    finally:

        if cursor is not None:

            cursor.close()


        if (
            conexao is not None
            and conexao.is_connected()
        ):

            conexao.close()


# ============================================================
# EXECUTAR SERVIDOR
# ============================================================

if __name__ == "__main__":

    print(
        "\nIniciando servidor..."
    )


    print(
        "\nAPI:"
    )

    print(
        "http://127.0.0.1:8000"
    )


    print(
        "\nDocumentação Swagger:"
    )

    print(
        "http://127.0.0.1:8000/docs"
    )


    print(
        "\nPara encerrar:"
    )

    print(
        "CTRL + C"
    )


    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000
    )