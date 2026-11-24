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

print("=" * 95)
print("API - SISTEMA DE ANÁLISE E CLASSIFICAÇÃO DO SONO - V2")
print("=" * 95)

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
        "fisiológicas e classificações experimentais "
        "dos estágios do sono."
    ),

    version="2.0.0"
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
# CONEXÃO
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
# CONVERTER DECIMAL / NONE
# ============================================================

def numero_float(valor):

    if valor is None:

        return None

    return float(valor)


# ============================================================
# /
# ============================================================

@app.get("/")
def raiz():

    return {

        "sistema":
            "Sistema de Análise e Classificação de Estágios do Sono",

        "status":
            "online",

        "versao_api":
            "2.0.0",

        "modelo_final_id":
            ID_MODELO_FINAL,

        "documentacao":
            "/docs"
    }


# ============================================================
# /health
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


        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM participantes
            """
        )

        participantes = int(
            cursor.fetchone()["total"]
        )


        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM noites
            """
        )

        noites = int(
            cursor.fetchone()["total"]
        )


        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM metricas_noite
            """
        )

        metricas = int(
            cursor.fetchone()["total"]
        )


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
            cursor.fetchone()["total"]
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
# /modelo
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
            cursor.fetchone()["total"]
        )


        return {

            "id_modelo":
                int(
                    modelo["id_modelo"]
                ),

            "nome":
                modelo["nome"],

            "algoritmo":
                modelo["algoritmo"],

            "versao":
                modelo["versao"],

            "data_treinamento":
                modelo["data_treinamento"],

            "metricas_teste_final": {

                "accuracy":
                    numero_float(
                        modelo["accuracy"]
                    ),

                "balanced_accuracy":
                    0.6015,

                "macro_f1":
                    numero_float(
                        modelo["macro_f1"]
                    ),

                "weighted_f1":
                    0.6561
            },

            "configuracao": {

                "peso_random_forest":
                    0.75,

                "peso_bigru":
                    0.25,

                "peso_viterbi":
                    0.60,

                "janela_bigru_epocas":
                    21,

                "janela_bigru_minutos":
                    10.5
            },

            "predicoes_registradas":
                predicoes,

            "observacoes":
                modelo["observacoes"]
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
# /participantes
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

                (
                    SELECT COUNT(*)

                    FROM predicoes pr

                    INNER JOIN epocas ep
                        ON ep.id_epoca = pr.id_epoca

                    INNER JOIN noites n2
                        ON n2.id_noite = ep.id_noite

                    WHERE
                        n2.id_participante = p.id_participante
                        AND pr.id_modelo = %s

                ) AS epocas_com_predicao

            FROM participantes p

            LEFT JOIN noites n
                ON n.id_participante = p.id_participante

            GROUP BY
                p.id_participante,
                p.codigo

            ORDER BY
                p.codigo
            """,
            (
                int(ID_MODELO_FINAL),
            )
        )


        registros = cursor.fetchall()

        resultado = []


        for registro in registros:

            epocas_preditas = int(
                registro["epocas_com_predicao"]
            )


            resultado.append({

                "id_participante":
                    int(
                        registro["id_participante"]
                    ),

                "codigo":
                    registro["codigo"],

                "quantidade_noites":
                    int(
                        registro["quantidade_noites"]
                    ),

                "epocas_com_predicao":
                    epocas_preditas,

                "possui_predicoes_modelo_final":
                    epocas_preditas > 0
            })


        return {

            "quantidade":
                len(resultado),

            "participantes":
                resultado
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
# /participantes/{codigo}/noites
# ============================================================

@app.get(
    "/participantes/{codigo}/noites"
)
def listar_noites_participante(
    codigo: str
):

    conexao = None
    cursor = None


    try:

        conexao = conectar_banco()

        cursor = conexao.cursor(
            dictionary=True
        )


        # ----------------------------------------------------
        # CONFIRMAR PARTICIPANTE
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id_participante,
                codigo
            FROM participantes
            WHERE codigo = %s
            """,
            (
                codigo,
            )
        )


        participante = cursor.fetchone()


        if participante is None:

            raise HTTPException(
                status_code=404,
                detail="Participante não encontrado."
            )


        # ----------------------------------------------------
        # NOITES + MÉTRICAS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                n.id_noite,
                n.numero_noite,
                n.inicio_utc,
                n.fim_utc,

                n.epocas_originais,
                n.epocas_unknown,
                n.epocas_validas,
                n.epocas_removidas,
                n.cobertura_percentual,

                m.duracao_analisada_min,
                m.tempo_sono_min,

                m.wake_pct,
                m.n1_pct,
                m.n2_pct,
                m.n3_pct,
                m.rem_pct,

                m.hr_media,
                m.movimento_medio,

                m.transicoes_estagio,
                m.possiveis_despertares,
                m.indice_experimental,

                (
                    SELECT COUNT(*)

                    FROM epocas ep

                    INNER JOIN predicoes pr
                        ON pr.id_epoca = ep.id_epoca

                    WHERE
                        ep.id_noite = n.id_noite
                        AND pr.id_modelo = %s

                ) AS epocas_com_predicao

            FROM noites n

            LEFT JOIN metricas_noite m
                ON m.id_noite = n.id_noite

            WHERE
                n.id_participante = %s

            ORDER BY
                n.numero_noite
            """,
            (
                int(ID_MODELO_FINAL),

                int(
                    participante["id_participante"]
                )
            )
        )


        registros = cursor.fetchall()

        noites = []


        for registro in registros:

            epocas_preditas = int(
                registro["epocas_com_predicao"]
            )


            noites.append({

                "id_noite":
                    int(
                        registro["id_noite"]
                    ),

                "numero_noite":
                    int(
                        registro["numero_noite"]
                    ),

                "inicio_utc":
                    registro["inicio_utc"],

                "fim_utc":
                    registro["fim_utc"],


                "qualidade_dados": {

                    "epocas_originais":
                        registro[
                            "epocas_originais"
                        ],

                    "epocas_unknown":
                        registro[
                            "epocas_unknown"
                        ],

                    "epocas_validas":
                        registro[
                            "epocas_validas"
                        ],

                    "epocas_removidas":
                        registro[
                            "epocas_removidas"
                        ],

                    "cobertura_percentual":
                        numero_float(
                            registro[
                                "cobertura_percentual"
                            ]
                        )
                },


                "duracao": {

                    "duracao_analisada_min":
                        numero_float(
                            registro[
                                "duracao_analisada_min"
                            ]
                        ),

                    "tempo_sono_min":
                        numero_float(
                            registro[
                                "tempo_sono_min"
                            ]
                        )
                },


                "distribuicao_referencia": {

                    "wake_pct":
                        numero_float(
                            registro["wake_pct"]
                        ),

                    "n1_pct":
                        numero_float(
                            registro["n1_pct"]
                        ),

                    "n2_pct":
                        numero_float(
                            registro["n2_pct"]
                        ),

                    "n3_pct":
                        numero_float(
                            registro["n3_pct"]
                        ),

                    "rem_pct":
                        numero_float(
                            registro["rem_pct"]
                        )
                },


                "sinais": {

                    "hr_media":
                        numero_float(
                            registro["hr_media"]
                        ),

                    "movimento_medio":
                        numero_float(
                            registro[
                                "movimento_medio"
                            ]
                        )
                },


                "eventos": {

                    "transicoes_estagio":
                        registro[
                            "transicoes_estagio"
                        ],

                    "possiveis_despertares":
                        registro[
                            "possiveis_despertares"
                        ]
                },


                "indice_experimental":
                    numero_float(
                        registro[
                            "indice_experimental"
                        ]
                    ),


                "modelo_final": {

                    "epocas_com_predicao":
                        epocas_preditas,

                    "possui_predicoes":
                        epocas_preditas > 0
                }
            })


        return {

            "participante": {

                "id_participante":
                    int(
                        participante[
                            "id_participante"
                        ]
                    ),

                "codigo":
                    participante["codigo"]
            },

            "quantidade_noites":
                len(noites),

            "noites":
                noites
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
# /noites/{id_noite}
# ============================================================

@app.get(
    "/noites/{id_noite}"
)
def obter_noite(
    id_noite: int
):

    conexao = None
    cursor = None


    try:

        conexao = conectar_banco()

        cursor = conexao.cursor(
            dictionary=True
        )


        # ----------------------------------------------------
        # DADOS PRINCIPAIS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                n.id_noite,
                n.numero_noite,
                n.inicio_utc,
                n.fim_utc,

                n.epocas_originais,
                n.epocas_unknown,
                n.epocas_validas,
                n.epocas_removidas,
                n.cobertura_percentual,

                p.id_participante,
                p.codigo AS participante,

                m.duracao_analisada_min,
                m.tempo_sono_min,

                m.wake_pct,
                m.n1_pct,
                m.n2_pct,
                m.n3_pct,
                m.rem_pct,

                m.hr_media,
                m.movimento_medio,

                m.transicoes_estagio,
                m.possiveis_despertares,
                m.indice_experimental

            FROM noites n

            INNER JOIN participantes p
                ON p.id_participante = n.id_participante

            LEFT JOIN metricas_noite m
                ON m.id_noite = n.id_noite

            WHERE n.id_noite = %s
            """,
            (
                int(id_noite),
            )
        )


        noite = cursor.fetchone()


        if noite is None:

            raise HTTPException(
                status_code=404,
                detail="Noite não encontrada."
            )


        # ----------------------------------------------------
        # DISTRIBUIÇÃO DO MODELO
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                e.id_estagio,
                e.codigo,
                e.nome,
                COUNT(*) AS quantidade

            FROM epocas ep

            INNER JOIN predicoes pr
                ON pr.id_epoca = ep.id_epoca
                AND pr.id_modelo = %s

            INNER JOIN estagios_sono e
                ON e.id_estagio = pr.id_estagio_previsto

            WHERE ep.id_noite = %s

            GROUP BY
                e.id_estagio,
                e.codigo,
                e.nome

            ORDER BY
                e.id_estagio
            """,
            (
                int(ID_MODELO_FINAL),
                int(id_noite)
            )
        )


        distribuicao_modelo_raw = (
            cursor.fetchall()
        )


        total_preditas = sum(

            int(
                linha["quantidade"]
            )

            for linha in distribuicao_modelo_raw
        )


        distribuicao_modelo = {

            "WAKE": 0.0,
            "N1": 0.0,
            "N2": 0.0,
            "N3": 0.0,
            "REM": 0.0
        }


        if total_preditas > 0:

            for linha in distribuicao_modelo_raw:

                codigo_estagio = (
                    linha["codigo"]
                )

                quantidade = int(
                    linha["quantidade"]
                )

                distribuicao_modelo[
                    codigo_estagio
                ] = round(
                    (
                        quantidade
                        / total_preditas
                    )
                    * 100,
                    4
                )


        # ----------------------------------------------------
        # CONFIANÇA
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                COUNT(*) AS quantidade,
                AVG(pr.confianca) AS media,
                MIN(pr.confianca) AS minima,
                MAX(pr.confianca) AS maxima

            FROM epocas ep

            INNER JOIN predicoes pr
                ON pr.id_epoca = ep.id_epoca

            WHERE
                ep.id_noite = %s
                AND pr.id_modelo = %s
            """,
            (
                int(id_noite),
                int(ID_MODELO_FINAL)
            )
        )


        confianca = cursor.fetchone()


        return {

            "id_noite":
                int(
                    noite["id_noite"]
                ),

            "participante": {

                "id_participante":
                    int(
                        noite[
                            "id_participante"
                        ]
                    ),

                "codigo":
                    noite["participante"]
            },

            "numero_noite":
                int(
                    noite["numero_noite"]
                ),

            "inicio_utc":
                noite["inicio_utc"],

            "fim_utc":
                noite["fim_utc"],


            "qualidade_dados": {

                "epocas_originais":
                    noite[
                        "epocas_originais"
                    ],

                "epocas_unknown":
                    noite[
                        "epocas_unknown"
                    ],

                "epocas_validas":
                    noite[
                        "epocas_validas"
                    ],

                "epocas_removidas":
                    noite[
                        "epocas_removidas"
                    ],

                "cobertura_percentual":
                    numero_float(
                        noite[
                            "cobertura_percentual"
                        ]
                    )
            },


            "metricas_referencia": {

                "duracao_analisada_min":
                    numero_float(
                        noite[
                            "duracao_analisada_min"
                        ]
                    ),

                "tempo_sono_min":
                    numero_float(
                        noite[
                            "tempo_sono_min"
                        ]
                    ),

                "wake_pct":
                    numero_float(
                        noite["wake_pct"]
                    ),

                "n1_pct":
                    numero_float(
                        noite["n1_pct"]
                    ),

                "n2_pct":
                    numero_float(
                        noite["n2_pct"]
                    ),

                "n3_pct":
                    numero_float(
                        noite["n3_pct"]
                    ),

                "rem_pct":
                    numero_float(
                        noite["rem_pct"]
                    ),

                "hr_media":
                    numero_float(
                        noite["hr_media"]
                    ),

                "movimento_medio":
                    numero_float(
                        noite[
                            "movimento_medio"
                        ]
                    ),

                "transicoes_estagio":
                    noite[
                        "transicoes_estagio"
                    ],

                "possiveis_despertares":
                    noite[
                        "possiveis_despertares"
                    ],

                "indice_experimental":
                    numero_float(
                        noite[
                            "indice_experimental"
                        ]
                    )
            },


            "modelo_final": {

                "id_modelo":
                    ID_MODELO_FINAL,

                "possui_predicoes":
                    total_preditas > 0,

                "quantidade_predicoes":
                    total_preditas,

                "distribuicao_percentual":
                    distribuicao_modelo,

                "confianca_experimental": {

                    "media":
                        numero_float(
                            confianca["media"]
                        ),

                    "minima":
                        numero_float(
                            confianca["minima"]
                        ),

                    "maxima":
                        numero_float(
                            confianca["maxima"]
                        )
                }
            }
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
# /noites/{id_noite}/estagios
# ============================================================

@app.get(
    "/noites/{id_noite}/estagios"
)
def listar_estagios_noite(
    id_noite: int
):

    conexao = None
    cursor = None


    try:

        conexao = conectar_banco()

        cursor = conexao.cursor(
            dictionary=True
        )


        # ----------------------------------------------------
        # VALIDAR NOITE
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                n.id_noite,
                n.numero_noite,
                p.codigo AS participante

            FROM noites n

            INNER JOIN participantes p
                ON p.id_participante = n.id_participante

            WHERE n.id_noite = %s
            """,
            (
                int(id_noite),
            )
        )


        noite = cursor.fetchone()


        if noite is None:

            raise HTTPException(
                status_code=404,
                detail="Noite não encontrada."
            )


        # ----------------------------------------------------
        # ÉPOCAS + REFERÊNCIA + PREDIÇÃO
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                ep.id_epoca,
                ep.numero_epoca,
                ep.inicio_utc,
                ep.fim_utc,

                ep.hr_mean,
                ep.motion_mean,

                ref.id_estagio
                    AS id_estagio_referencia,

                ref.codigo
                    AS codigo_referencia,

                ref.nome
                    AS nome_referencia,

                pr.id_estagio_previsto,

                prev.codigo
                    AS codigo_previsto,

                prev.nome
                    AS nome_previsto,

                pr.confianca

            FROM epocas ep

            INNER JOIN estagios_sono ref
                ON ref.id_estagio =
                    ep.id_estagio_referencia

            LEFT JOIN predicoes pr
                ON pr.id_epoca = ep.id_epoca
                AND pr.id_modelo = %s

            LEFT JOIN estagios_sono prev
                ON prev.id_estagio =
                    pr.id_estagio_previsto

            WHERE ep.id_noite = %s

            ORDER BY
                ep.numero_epoca
            """,
            (
                int(ID_MODELO_FINAL),
                int(id_noite)
            )
        )


        registros = cursor.fetchall()

        epocas = []


        for registro in registros:

            previsao_existe = (
                registro[
                    "id_estagio_previsto"
                ]
                is not None
            )


            epocas.append({

                "id_epoca":
                    int(
                        registro["id_epoca"]
                    ),

                "numero_epoca":
                    int(
                        registro["numero_epoca"]
                    ),

                "inicio_utc":
                    registro["inicio_utc"],

                "fim_utc":
                    registro["fim_utc"],


                "sinais": {

                    "hr_mean":
                        numero_float(
                            registro["hr_mean"]
                        ),

                    "motion_mean":
                        numero_float(
                            registro["motion_mean"]
                        )
                },


                "referencia": {

                    "id_estagio":
                        int(
                            registro[
                                "id_estagio_referencia"
                            ]
                        ),

                    "codigo":
                        registro[
                            "codigo_referencia"
                        ],

                    "nome":
                        registro[
                            "nome_referencia"
                        ]
                },


                "predicao": (
                    {

                        "id_estagio":
                            int(
                                registro[
                                    "id_estagio_previsto"
                                ]
                            ),

                        "codigo":
                            registro[
                                "codigo_previsto"
                            ],

                        "nome":
                            registro[
                                "nome_previsto"
                            ],

                        "confianca_experimental":
                            numero_float(
                                registro[
                                    "confianca"
                                ]
                            )
                    }

                    if previsao_existe

                    else None
                )
            })


        predicoes_disponiveis = sum(

            1

            for epoca in epocas

            if epoca["predicao"] is not None
        )


        return {

            "noite": {

                "id_noite":
                    int(
                        noite["id_noite"]
                    ),

                "numero_noite":
                    int(
                        noite["numero_noite"]
                    ),

                "participante":
                    noite["participante"]
            },

            "quantidade_epocas":
                len(epocas),

            "predicoes_disponiveis":
                predicoes_disponiveis,

            "epocas":
                epocas
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
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    print(
        "\nIniciando API V2..."
    )


    print(
        "\nSwagger:"
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