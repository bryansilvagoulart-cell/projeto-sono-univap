from getpass import getpass
import os

import mysql.connector
from mysql.connector import Error

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ============================================================
# CONFIGURAÇÃO
# ============================================================

HOST = "127.0.0.1"
PORT = 3306
DATABASE = "projeto_sono"

USUARIO_PADRAO = "projeto_sono_app"

MODELO_ID = 2
TAXONOMIA_ESPERADA = "sono_4_classes_v1"
VERSAO_ESPERADA = "4_classes_final_1.0"

DB_PASSWORD = os.getenv(
    "PROJETO_SONO_DB_PASSWORD"
)


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Projeto Sono - API 4 Classes",
    description=(
        "API experimental para consulta das classificações "
        "de sono em quatro classes: Wake, Light, Deep e REM."
    ),
    version="3.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=False,
    allow_methods=[
        "*"
    ],
    allow_headers=[
        "*"
    ]
)


# ============================================================
# BANCO
# ============================================================

def conectar_banco():

    if not DB_PASSWORD:

        raise RuntimeError(
            "Senha do banco ainda não configurada."
        )


    return mysql.connector.connect(
        host=HOST,
        port=PORT,
        user=USUARIO_PADRAO,
        password=DB_PASSWORD,
        database=DATABASE
    )


def executar_select(
    sql,
    parametros=None
):

    conexao = None
    cursor = None


    try:

        conexao = conectar_banco()

        cursor = conexao.cursor(
            dictionary=True
        )

        cursor.execute(
            sql,
            parametros or ()
        )

        return cursor.fetchall()


    except Error as erro:

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro ao consultar o banco: "
                f"{erro}"
            )
        )


    finally:

        if cursor is not None:

            cursor.close()


        if conexao is not None:

            conexao.close()


def executar_select_um(
    sql,
    parametros=None
):

    resultados = executar_select(
        sql,
        parametros
    )


    if not resultados:

        return None


    return resultados[0]


def validar_modelo_4_classes():

    modelo = executar_select_um(
        """
        SELECT
            id_modelo,
            nome,
            algoritmo,
            versao,
            data_treinamento,
            accuracy,
            balanced_accuracy,
            macro_f1,
            weighted_f1,
            taxonomia,
            quantidade_classes,
            observacoes
        FROM modelos_ml
        WHERE id_modelo = %s
        """,
        (
            MODELO_ID,
        )
    )


    if modelo is None:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Modelo id={MODELO_ID} "
                "não encontrado."
            )
        )


    if modelo[
        "taxonomia"
    ] != TAXONOMIA_ESPERADA:

        raise HTTPException(
            status_code=500,
            detail=(
                "Taxonomia inesperada no "
                f"modelo id={MODELO_ID}."
            )
        )


    if int(
        modelo[
            "quantidade_classes"
        ]
    ) != 4:

        raise HTTPException(
            status_code=500,
            detail=(
                "O modelo configurado não "
                "possui quatro classes."
            )
        )


    return modelo


# ============================================================
# ROTAS BÁSICAS
# ============================================================

@app.get("/")
def raiz():

    return {
        "projeto":
            "Sistema Experimental de Análise do Sono",

        "api":
            "Projeto Sono - API 4 Classes",

        "versao_api":
            "3.0.0",

        "modelo_id":
            MODELO_ID,

        "taxonomia":
            TAXONOMIA_ESPERADA,

        "classes": [
            "Wake",
            "Light",
            "Deep",
            "REM"
        ],

        "aviso":
            (
                "Sistema experimental para pesquisa e "
                "demonstração de engenharia. "
                "Não é ferramenta de diagnóstico médico."
            ),

        "documentacao":
            "/docs"
    }


@app.get("/health")
def health():

    try:

        modelo = validar_modelo_4_classes()


        contagem_predicoes = executar_select_um(
            """
            SELECT COUNT(*) AS quantidade
            FROM predicoes_modelo_ml
            WHERE id_modelo = %s
            """,
            (
                MODELO_ID,
            )
        )


        contagem_metricas = executar_select_um(
            """
            SELECT COUNT(*) AS quantidade
            FROM metricas_noite_modelo_ml
            WHERE id_modelo = %s
            """,
            (
                MODELO_ID,
            )
        )


        return {
            "status":
                "ok",

            "banco":
                "conectado",

            "modelo_id":
                MODELO_ID,

            "modelo_versao":
                modelo[
                    "versao"
                ],

            "predicoes":
                int(
                    contagem_predicoes[
                        "quantidade"
                    ]
                ),

            "metricas_noite":
                int(
                    contagem_metricas[
                        "quantidade"
                    ]
                )
        }


    except HTTPException:

        raise


    except Exception as erro:

        raise HTTPException(
            status_code=500,
            detail=str(
                erro
            )
        )


# ============================================================
# MODELO
# ============================================================

@app.get("/modelo")
def modelo():

    modelo_db = validar_modelo_4_classes()


    classes = executar_select(
        """
        SELECT
            rotulo_modelo,
            codigo,
            nome,
            descricao
        FROM classes_modelo_ml
        WHERE taxonomia = %s
        ORDER BY rotulo_modelo
        """,
        (
            TAXONOMIA_ESPERADA,
        )
    )


    if len(
        classes
    ) != 4:

        raise HTTPException(
            status_code=500,
            detail=(
                "Catálogo de quatro classes "
                "incompleto."
            )
        )


    return {
        "id_modelo":
            modelo_db[
                "id_modelo"
            ],

        "nome":
            modelo_db[
                "nome"
            ],

        "algoritmo":
            modelo_db[
                "algoritmo"
            ],

        "versao":
            modelo_db[
                "versao"
            ],

        "data_treinamento":
            modelo_db[
                "data_treinamento"
            ],

        "taxonomia":
            modelo_db[
                "taxonomia"
            ],

        "quantidade_classes":
            modelo_db[
                "quantidade_classes"
            ],

        "metricas_teste_final": {
            "accuracy":
                modelo_db[
                    "accuracy"
                ],

            "balanced_accuracy":
                modelo_db[
                    "balanced_accuracy"
                ],

            "macro_f1":
                modelo_db[
                    "macro_f1"
                ],

            "weighted_f1":
                modelo_db[
                    "weighted_f1"
                ]
        },

        "classes":
            classes,

        "configuracao": {
            "random_forest":
                0.75,

            "bigru":
                0.25,

            "viterbi":
                0.80,

            "janela_bigru_epocas":
                21,

            "duracao_janela_min":
                10.5
        },

        "observacoes":
            modelo_db[
                "observacoes"
            ]
    }


# ============================================================
# PARTICIPANTES
# ============================================================

@app.get("/participantes")
def participantes():

    registros = executar_select(
        """
        SELECT
            p.id_participante,
            p.codigo,
            COUNT(
                DISTINCT n.id_noite
            ) AS quantidade_noites,
            COUNT(
                pm.id_predicao
            ) AS quantidade_epocas

        FROM participantes p

        JOIN noites n
          ON n.id_participante =
             p.id_participante

        JOIN epocas e
          ON e.id_noite =
             n.id_noite

        JOIN predicoes_modelo_ml pm
          ON pm.id_epoca =
             e.id_epoca
         AND pm.id_modelo = %s

        GROUP BY
            p.id_participante,
            p.codigo

        ORDER BY
            p.codigo
        """,
        (
            MODELO_ID,
        )
    )


    return {
        "modelo_id":
            MODELO_ID,

        "quantidade":
            len(
                registros
            ),

        "participantes":
            registros
    }


@app.get("/participantes/{codigo}/noites")
def noites_participante(
    codigo: str
):

    participante = executar_select_um(
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


    if participante is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Participante não encontrado."
            )
        )


    noites = executar_select(
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
            m.tempo_sono_estimado_min,
            m.wake_pct,
            m.light_pct,
            m.deep_pct,
            m.rem_pct,
            m.hr_media,
            m.movimento_medio,
            m.transicoes_estagio,
            m.possiveis_despertares,
            m.confianca_media,
            m.indice_experimental

        FROM noites n

        JOIN participantes p
          ON p.id_participante =
             n.id_participante

        JOIN metricas_noite_modelo_ml m
          ON m.id_noite =
             n.id_noite
         AND m.id_modelo = %s

        WHERE p.codigo = %s

        ORDER BY
            n.numero_noite
        """,
        (
            MODELO_ID,
            codigo
        )
    )


    if not noites:

        raise HTTPException(
            status_code=404,
            detail=(
                "O participante existe, mas não "
                "possui noites analisadas pelo "
                "modelo final de quatro classes."
            )
        )


    return {
        "participante":
            participante,

        "modelo_id":
            MODELO_ID,

        "quantidade_noites":
            len(
                noites
            ),

        "noites":
            noites
    }


# ============================================================
# DETALHE DA NOITE
# ============================================================

@app.get("/noites/{id_noite}")
def detalhe_noite(
    id_noite: int
):

    registro = executar_select_um(
        """
        SELECT
            p.codigo AS participante,

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
            m.tempo_sono_estimado_min,
            m.wake_pct,
            m.light_pct,
            m.deep_pct,
            m.rem_pct,
            m.hr_media,
            m.movimento_medio,
            m.transicoes_estagio,
            m.possiveis_despertares,
            m.confianca_media,
            m.indice_experimental

        FROM noites n

        JOIN participantes p
          ON p.id_participante =
             n.id_participante

        JOIN metricas_noite_modelo_ml m
          ON m.id_noite =
             n.id_noite
         AND m.id_modelo = %s

        WHERE n.id_noite = %s
        """,
        (
            MODELO_ID,
            id_noite
        )
    )


    if registro is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Noite não encontrada para "
                "o modelo final de quatro classes."
            )
        )


    return {
        "modelo_id":
            MODELO_ID,

        "participante":
            registro[
                "participante"
            ],

        "noite": {
            "id_noite":
                registro[
                    "id_noite"
                ],

            "numero_noite":
                registro[
                    "numero_noite"
                ],

            "inicio_utc":
                registro[
                    "inicio_utc"
                ],

            "fim_utc":
                registro[
                    "fim_utc"
                ]
        },

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
                registro[
                    "cobertura_percentual"
                ]
        },

        "metricas": {
            "duracao_analisada_min":
                registro[
                    "duracao_analisada_min"
                ],

            "tempo_sono_estimado_min":
                registro[
                    "tempo_sono_estimado_min"
                ],

            "wake_pct":
                registro[
                    "wake_pct"
                ],

            "light_pct":
                registro[
                    "light_pct"
                ],

            "deep_pct":
                registro[
                    "deep_pct"
                ],

            "rem_pct":
                registro[
                    "rem_pct"
                ],

            "hr_media":
                registro[
                    "hr_media"
                ],

            "movimento_medio":
                registro[
                    "movimento_medio"
                ],

            "transicoes_estagio":
                registro[
                    "transicoes_estagio"
                ],

            "possiveis_despertares":
                registro[
                    "possiveis_despertares"
                ],

            "confianca_media":
                registro[
                    "confianca_media"
                ],

            "indice_experimental":
                registro[
                    "indice_experimental"
                ]
        },

        "aviso":
            (
                "As métricas e classificações são "
                "experimentais e não representam "
                "diagnóstico médico."
            )
    }


# ============================================================
# ÉPOCAS / HIPNOGRAMA
# ============================================================

@app.get("/noites/{id_noite}/estagios")
def estagios_noite(
    id_noite: int
):

    noite = executar_select_um(
        """
        SELECT
            p.codigo AS participante,
            n.id_noite,
            n.numero_noite
        FROM noites n
        JOIN participantes p
          ON p.id_participante =
             n.id_participante
        WHERE n.id_noite = %s
        """,
        (
            id_noite,
        )
    )


    if noite is None:

        raise HTTPException(
            status_code=404,
            detail="Noite não encontrada."
        )


    epocas = executar_select(
        """
        SELECT
            e.numero_epoca,
            e.inicio_utc,
            e.fim_utc,

            e.hr_mean,
            e.hr_min,
            e.hr_max,
            e.hr_std,
            e.hr_samples,

            e.motion_mean,
            e.motion_std,
            e.motion_max,
            e.motion_samples,

            e.cobertura_temporal_pct,
            e.maior_gap,

            ref.codigo AS referencia_original,

            CASE e.id_estagio_referencia
                WHEN 0 THEN 'WAKE'
                WHEN 1 THEN 'LIGHT'
                WHEN 2 THEN 'LIGHT'
                WHEN 3 THEN 'DEEP'
                WHEN 4 THEN 'REM'
                ELSE 'UNKNOWN'
            END AS referencia_4_classes,

            c.rotulo_modelo AS rotulo_previsto,
            c.codigo AS classe_prevista,
            c.nome AS nome_classe_prevista,

            pm.confianca

        FROM epocas e

        JOIN estagios_sono ref
          ON ref.id_estagio =
             e.id_estagio_referencia

        JOIN predicoes_modelo_ml pm
          ON pm.id_epoca =
             e.id_epoca
         AND pm.id_modelo = %s

        JOIN classes_modelo_ml c
          ON c.id_classe_modelo =
             pm.id_classe_modelo

        WHERE e.id_noite = %s

        ORDER BY
            e.numero_epoca
        """,
        (
            MODELO_ID,
            id_noite
        )
    )


    if not epocas:

        raise HTTPException(
            status_code=404,
            detail=(
                "A noite existe, mas não possui "
                "predições do modelo final de "
                "quatro classes."
            )
        )


    return {
        "participante":
            noite[
                "participante"
            ],

        "id_noite":
            noite[
                "id_noite"
            ],

        "numero_noite":
            noite[
                "numero_noite"
            ],

        "modelo_id":
            MODELO_ID,

        "quantidade_epocas":
            len(
                epocas
            ),

        "epocas":
            epocas
    }


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 100
    )

    print(
        "PROJETO SONO - API 4 CLASSES"
    )

    print(
        "=" * 100
    )


    print(
        f"\nBanco: "
        f"{DATABASE}"
    )

    print(
        f"Usuário: "
        f"{USUARIO_PADRAO}"
    )

    print(
        f"Modelo: "
        f"id={MODELO_ID}"
    )


    if not DB_PASSWORD:

        DB_PASSWORD = getpass(
            f"\nSenha do usuário "
            f"{USUARIO_PADRAO}: "
        )


    try:

        conexao_teste = conectar_banco()

        cursor_teste = (
            conexao_teste.cursor()
        )

        cursor_teste.execute(
            "SELECT 1"
        )

        cursor_teste.fetchone()

        cursor_teste.close()

        conexao_teste.close()


        print(
            "\n✅ Conexão com MySQL validada."
        )


    except Exception as erro:

        print(
            "\n❌ Não foi possível conectar "
            "ao MySQL:"
        )

        print(
            erro
        )

        raise SystemExit(
            1
        )


    print(
        "\nAPI disponível em:"
    )

    print(
        "  http://127.0.0.1:8000"
    )

    print(
        "\nSwagger:"
    )

    print(
        "  http://127.0.0.1:8000/docs"
    )


    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
