from getpass import getpass

import numpy as np
import pandas as pd

import mysql.connector
from mysql.connector import Error


# ============================================================
# CONFIGURAÇÃO
# ============================================================

HOST = "127.0.0.1"
PORT = 3306
DATABASE = "projeto_sono"

USUARIO_ADMIN_PADRAO = "root"

USUARIO_APP = "projeto_sono_app"
HOST_USUARIO_APP = "localhost"

NOME_MODELO = "Ensemble RF + BiGRU + Viterbi"
VERSAO_MODELO = "4_classes_final_1.0"
TAXONOMIA = "sono_4_classes_v1"

PARTICIPANTES_TESTE = [
    "Bidslab01",
    "Bidslab35",
    "Bidslab47"
]

QUANTIDADE_NOITES_ESPERADA = 15
QUANTIDADE_EPOCAS_ESPERADA = 12252


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def tabela_existe(
    cursor,
    tabela
):

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

    return int(
        cursor.fetchone()[0]
    ) > 0


def obter_tipo_coluna(
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
            f"Coluna não encontrada: "
            f"{tabela}.{coluna}"
        )


    return str(
        resultado[0]
    )


def obter_id_modelo(
    cursor
):

    cursor.execute(
        """
        SELECT
            id_modelo,
            taxonomia,
            quantidade_classes
        FROM modelos_ml
        WHERE nome = %s
          AND versao = %s
        """,
        (
            NOME_MODELO,
            VERSAO_MODELO
        )
    )


    resultado = cursor.fetchone()


    if resultado is None:

        raise RuntimeError(
            "Modelo final de 4 classes "
            "não encontrado em modelos_ml."
        )


    id_modelo = int(
        resultado[0]
    )

    taxonomia = str(
        resultado[1]
    )

    quantidade_classes = int(
        resultado[2]
    )


    if taxonomia != TAXONOMIA:

        raise RuntimeError(
            "Taxonomia inesperada para "
            f"id_modelo={id_modelo}: "
            f"{taxonomia}"
        )


    if quantidade_classes != 4:

        raise RuntimeError(
            "O modelo encontrado não possui "
            "quantidade_classes=4."
        )


    return id_modelo


def calcular_metricas_noite(
    grupo
):

    grupo = (
        grupo
        .sort_values(
            "numero_epoca"
        )
        .reset_index(
            drop=True
        )
    )


    labels = (
        grupo[
            "rotulo_modelo"
        ]
        .astype(int)
        .to_numpy()
    )


    epocas = (
        grupo[
            "numero_epoca"
        ]
        .astype(int)
        .to_numpy()
    )


    total = len(
        grupo
    )


    if total == 0:

        raise RuntimeError(
            "Grupo de noite sem épocas."
        )


    duracao_analisada_min = (
        total
        * 0.5
    )


    tempo_sono_estimado_min = (
        int(
            (
                labels != 0
            ).sum()
        )
        * 0.5
    )


    percentuais = {}


    for classe in [
        0,
        1,
        2,
        3
    ]:

        quantidade = int(
            (
                labels
                == classe
            ).sum()
        )


        percentuais[
            classe
        ] = (
            quantidade
            / total
            * 100.0
        )


    transicoes_estagio = 0

    possiveis_despertares = 0


    for i in range(
        1,
        total
    ):

        if (
            epocas[i]
            - epocas[i - 1]
            != 1
        ):

            continue


        anterior = int(
            labels[
                i - 1
            ]
        )

        atual = int(
            labels[
                i
            ]
        )


        if anterior != atual:

            transicoes_estagio += 1


        if (
            anterior != 0
            and
            atual == 0
        ):

            possiveis_despertares += 1


    hr_media = float(
        grupo[
            "hr_mean"
        ]
        .mean()
    )


    movimento_medio = float(
        grupo[
            "motion_mean"
        ]
        .mean()
    )


    confianca_media = float(
        grupo[
            "confianca"
        ]
        .mean()
    )


    return {
        "duracao_analisada_min":
            duracao_analisada_min,

        "tempo_sono_estimado_min":
            tempo_sono_estimado_min,

        "wake_pct":
            percentuais[0],

        "light_pct":
            percentuais[1],

        "deep_pct":
            percentuais[2],

        "rem_pct":
            percentuais[3],

        "hr_media":
            hr_media,

        "movimento_medio":
            movimento_medio,

        "transicoes_estagio":
            transicoes_estagio,

        "possiveis_despertares":
            possiveis_despertares,

        "confianca_media":
            confianca_media
    }


# ============================================================
# INÍCIO
# ============================================================

print(
    "=" * 100
)

print(
    "MÉTRICAS AGREGADAS POR NOITE - MODELO FINAL DE 4 CLASSES"
)

print(
    "=" * 100
)


print(
    "\nEsta etapa:"
)

print(
    "  - cria uma tabela NOVA de métricas por modelo"
)

print(
    "  - não altera metricas_noite antiga"
)

print(
    "  - calcula métricas para as 15 noites do teste final"
)

print(
    "  - usa somente as predições já congeladas do modelo"
)


usuario_admin = input(
    f"\nUsuário administrador MySQL "
    f"[{USUARIO_ADMIN_PADRAO}]: "
).strip()


if not usuario_admin:

    usuario_admin = USUARIO_ADMIN_PADRAO


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

        database=DATABASE,

        autocommit=False
    )


    cursor = conexao.cursor()


    print(
        "✅ Conexão administrativa realizada."
    )


    tabelas_obrigatorias = [
        "participantes",
        "noites",
        "epocas",
        "modelos_ml",
        "classes_modelo_ml",
        "predicoes_modelo_ml"
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


    id_modelo = obter_id_modelo(
        cursor
    )


    print(
        f"\n✅ Modelo localizado: "
        f"id_modelo={id_modelo}"
    )


    print(
        "\nPreparando tabela "
        "metricas_noite_modelo_ml..."
    )


    tipo_id_noite = obter_tipo_coluna(
        cursor,
        "noites",
        "id_noite"
    )


    tipo_id_modelo = obter_tipo_coluna(
        cursor,
        "modelos_ml",
        "id_modelo"
    )


    sql_criacao = f"""
        CREATE TABLE IF NOT EXISTS metricas_noite_modelo_ml (

            id_metrica_modelo
                BIGINT UNSIGNED
                NOT NULL
                AUTO_INCREMENT,

            id_noite
                {tipo_id_noite}
                NOT NULL,

            id_modelo
                {tipo_id_modelo}
                NOT NULL,

            duracao_analisada_min
                DOUBLE
                NULL,

            tempo_sono_estimado_min
                DOUBLE
                NULL,

            wake_pct
                DOUBLE
                NULL,

            light_pct
                DOUBLE
                NULL,

            deep_pct
                DOUBLE
                NULL,

            rem_pct
                DOUBLE
                NULL,

            hr_media
                DOUBLE
                NULL,

            movimento_medio
                DOUBLE
                NULL,

            transicoes_estagio
                INT UNSIGNED
                NULL,

            possiveis_despertares
                INT UNSIGNED
                NULL,

            confianca_media
                DOUBLE
                NULL,

            indice_experimental
                DOUBLE
                NULL,

            atualizado_em
                TIMESTAMP
                NOT NULL
                DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,

            PRIMARY KEY (
                id_metrica_modelo
            ),

            UNIQUE KEY uq_metricas_noite_modelo (
                id_noite,
                id_modelo
            ),

            KEY idx_metricas_modelo (
                id_modelo
            ),

            CONSTRAINT fk_metricas_modelo_noite
                FOREIGN KEY (
                    id_noite
                )
                REFERENCES noites (
                    id_noite
                )
                ON DELETE CASCADE
                ON UPDATE CASCADE,

            CONSTRAINT fk_metricas_modelo_modelo
                FOREIGN KEY (
                    id_modelo
                )
                REFERENCES modelos_ml (
                    id_modelo
                )
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
        ENGINE=InnoDB
        DEFAULT CHARSET=utf8mb4
        COLLATE=utf8mb4_0900_ai_ci
    """


    cursor.execute(
        sql_criacao
    )


    print(
        "✅ Tabela metricas_noite_modelo_ml pronta."
    )


    try:

        cursor.execute(
            f"""
            GRANT SELECT
            ON `{DATABASE}`.`metricas_noite_modelo_ml`
            TO '{USUARIO_APP}'@'{HOST_USUARIO_APP}'
            """
        )


        print(
            "✅ Permissão SELECT concedida "
            "ao usuário da aplicação."
        )


    except Error as erro_grant:

        print(
            "⚠️ Não foi possível executar GRANT automaticamente."
        )

        print(
            f"   MySQL informou: {erro_grant}"
        )

        print(
            "   A tabela foi criada normalmente; "
            "validaremos o restante."
        )


    placeholders = ", ".join(
        ["%s"] * len(
            PARTICIPANTES_TESTE
        )
    )


    cursor.execute(
        f"""
        SELECT
            p.codigo AS participante,
            n.id_noite,
            n.numero_noite,
            e.numero_epoca,
            e.hr_mean,
            e.motion_mean,
            c.rotulo_modelo,
            pm.confianca

        FROM predicoes_modelo_ml pm

        JOIN epocas e
          ON e.id_epoca = pm.id_epoca

        JOIN noites n
          ON n.id_noite = e.id_noite

        JOIN participantes p
          ON p.id_participante = n.id_participante

        JOIN classes_modelo_ml c
          ON c.id_classe_modelo =
             pm.id_classe_modelo

        WHERE pm.id_modelo = %s
          AND p.codigo IN ({placeholders})

        ORDER BY
            p.codigo,
            n.numero_noite,
            e.numero_epoca
        """,
        (
            id_modelo,
            *PARTICIPANTES_TESTE
        )
    )


    registros = cursor.fetchall()


    colunas = [
        "participante",
        "id_noite",
        "numero_noite",
        "numero_epoca",
        "hr_mean",
        "motion_mean",
        "rotulo_modelo",
        "confianca"
    ]


    dados = pd.DataFrame(
        registros,
        columns=colunas
    )


    if len(
        dados
    ) != QUANTIDADE_EPOCAS_ESPERADA:

        raise RuntimeError(
            f"Esperávamos "
            f"{QUANTIDADE_EPOCAS_ESPERADA} "
            f"predições no banco, "
            f"mas encontramos "
            f"{len(dados)}."
        )


    quantidade_noites = (
        dados[
            "id_noite"
        ]
        .nunique()
    )


    if (
        quantidade_noites
        != QUANTIDADE_NOITES_ESPERADA
    ):

        raise RuntimeError(
            f"Esperávamos "
            f"{QUANTIDADE_NOITES_ESPERADA} noites, "
            f"mas encontramos "
            f"{quantidade_noites}."
        )


    print(
        "\n✅ Predições carregadas:"
    )


    print(
        f"  Épocas: "
        f"{len(dados)}"
    )


    print(
        f"  Noites: "
        f"{quantidade_noites}"
    )


    print(
        "\nCalculando métricas por noite..."
    )


    linhas_banco = []

    resumo_console = []


    grupos = dados.groupby(
        [
            "participante",
            "id_noite",
            "numero_noite"
        ],
        sort=True
    )


    for (
        participante,
        id_noite,
        numero_noite
    ), grupo in grupos:


        metricas = calcular_metricas_noite(
            grupo
        )


        soma_percentuais = (
            metricas[
                "wake_pct"
            ]
            +
            metricas[
                "light_pct"
            ]
            +
            metricas[
                "deep_pct"
            ]
            +
            metricas[
                "rem_pct"
            ]
        )


        if not np.isclose(
            soma_percentuais,
            100.0,
            atol=1e-6
        ):

            raise RuntimeError(
                "Percentuais não somam 100% em "
                f"{participante}, noite "
                f"{numero_noite}."
            )


        linhas_banco.append(
            (
                int(
                    id_noite
                ),

                id_modelo,

                float(
                    metricas[
                        "duracao_analisada_min"
                    ]
                ),

                float(
                    metricas[
                        "tempo_sono_estimado_min"
                    ]
                ),

                float(
                    metricas[
                        "wake_pct"
                    ]
                ),

                float(
                    metricas[
                        "light_pct"
                    ]
                ),

                float(
                    metricas[
                        "deep_pct"
                    ]
                ),

                float(
                    metricas[
                        "rem_pct"
                    ]
                ),

                float(
                    metricas[
                        "hr_media"
                    ]
                ),

                float(
                    metricas[
                        "movimento_medio"
                    ]
                ),

                int(
                    metricas[
                        "transicoes_estagio"
                    ]
                ),

                int(
                    metricas[
                        "possiveis_despertares"
                    ]
                ),

                float(
                    metricas[
                        "confianca_media"
                    ]
                )
            )
        )


        resumo_console.append(
            {
                "participante":
                    participante,

                "noite":
                    int(
                        numero_noite
                    ),

                "epocas":
                    len(
                        grupo
                    ),

                "wake_pct":
                    metricas[
                        "wake_pct"
                    ],

                "light_pct":
                    metricas[
                        "light_pct"
                    ],

                "deep_pct":
                    metricas[
                        "deep_pct"
                    ],

                "rem_pct":
                    metricas[
                        "rem_pct"
                    ],

                "possiveis_despertares":
                    metricas[
                        "possiveis_despertares"
                    ],

                "confianca_media":
                    metricas[
                        "confianca_media"
                    ]
            }
        )


    if len(
        linhas_banco
    ) != QUANTIDADE_NOITES_ESPERADA:

        raise RuntimeError(
            "Quantidade inesperada de "
            "métricas por noite."
        )


    print(
        "✅ 15 noites calculadas."
    )


    print(
        "\nGravando métricas no MySQL..."
    )


    cursor.executemany(
        """
        INSERT INTO metricas_noite_modelo_ml (

            id_noite,
            id_modelo,
            duracao_analisada_min,
            tempo_sono_estimado_min,
            wake_pct,
            light_pct,
            deep_pct,
            rem_pct,
            hr_media,
            movimento_medio,
            transicoes_estagio,
            possiveis_despertares,
            confianca_media,
            indice_experimental

        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, NULL
        )

        ON DUPLICATE KEY UPDATE

            duracao_analisada_min =
                VALUES(duracao_analisada_min),

            tempo_sono_estimado_min =
                VALUES(tempo_sono_estimado_min),

            wake_pct =
                VALUES(wake_pct),

            light_pct =
                VALUES(light_pct),

            deep_pct =
                VALUES(deep_pct),

            rem_pct =
                VALUES(rem_pct),

            hr_media =
                VALUES(hr_media),

            movimento_medio =
                VALUES(movimento_medio),

            transicoes_estagio =
                VALUES(transicoes_estagio),

            possiveis_despertares =
                VALUES(possiveis_despertares),

            confianca_media =
                VALUES(confianca_media)
        """,
        linhas_banco
    )


    print(
        "✅ Métricas enviadas ao banco."
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "VALIDAÇÃO ANTES DO COMMIT"
    )

    print(
        "=" * 100
    )


    cursor.execute(
        f"""
        SELECT COUNT(*)

        FROM metricas_noite_modelo_ml m

        JOIN noites n
          ON n.id_noite = m.id_noite

        JOIN participantes p
          ON p.id_participante =
             n.id_participante

        WHERE m.id_modelo = %s
          AND p.codigo IN ({placeholders})
        """,
        (
            id_modelo,
            *PARTICIPANTES_TESTE
        )
    )


    quantidade_metricas = int(
        cursor.fetchone()[0]
    )


    if (
        quantidade_metricas
        != QUANTIDADE_NOITES_ESPERADA
    ):

        raise RuntimeError(
            f"Esperávamos 15 registros de métricas, "
            f"mas encontramos "
            f"{quantidade_metricas}."
        )


    cursor.execute(
        f"""
        SELECT
            MIN(
                wake_pct
                + light_pct
                + deep_pct
                + rem_pct
            ),
            MAX(
                wake_pct
                + light_pct
                + deep_pct
                + rem_pct
            ),
            MIN(confianca_media),
            MAX(confianca_media)

        FROM metricas_noite_modelo_ml m

        JOIN noites n
          ON n.id_noite = m.id_noite

        JOIN participantes p
          ON p.id_participante =
             n.id_participante

        WHERE m.id_modelo = %s
          AND p.codigo IN ({placeholders})
        """,
        (
            id_modelo,
            *PARTICIPANTES_TESTE
        )
    )


    (
        soma_minima,
        soma_maxima,
        confianca_minima,
        confianca_maxima
    ) = cursor.fetchone()


    if not np.isclose(
        float(
            soma_minima
        ),
        100.0,
        atol=1e-6
    ):

        raise RuntimeError(
            "Soma mínima dos percentuais "
            "é diferente de 100%."
        )


    if not np.isclose(
        float(
            soma_maxima
        ),
        100.0,
        atol=1e-6
    ):

        raise RuntimeError(
            "Soma máxima dos percentuais "
            "é diferente de 100%."
        )


    if (
        float(
            confianca_minima
        ) < 0.0
        or
        float(
            confianca_maxima
        ) > 1.0
    ):

        raise RuntimeError(
            "Confiança média de noite "
            "fora do intervalo [0, 1]."
        )


    conexao.commit()


    print(
        "\n✅ Validação concluída."
    )


    print(
        "✅ COMMIT realizado."
    )


    resumo = pd.DataFrame(
        resumo_console
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "MÉTRICAS POR NOITE REGISTRADAS"
    )

    print(
        "=" * 100
    )


    print(
        "\n"
        + resumo.round(
            {
                "wake_pct": 1,
                "light_pct": 1,
                "deep_pct": 1,
                "rem_pct": 1,
                "confianca_media": 3
            }
        ).to_string(
            index=False
        )
    )


    print(
        "\nRegistros gravados:"
    )


    print(
        f"  {quantidade_metricas}"
    )


    print(
        "\n✅ metricas_noite antiga foi preservada."
    )


    print(
        "✅ indice_experimental ficou NULL "
        "por enquanto."
    )


    print(
        "✅ possiveis_despertares é um "
        "indicador experimental:"
    )


    print(
        "   conta entradas em Wake após "
        "uma classe de sono em épocas consecutivas."
    )


    print(
        "\nPróxima etapa:"
    )


    print(
        "Atualizar a API para utilizar "
        "modelos_ml.id_modelo=2, "
        "predicoes_modelo_ml e "
        "metricas_noite_modelo_ml."
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

            print(
                "↩️ ROLLBACK realizado."
            )

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

            print(
                "↩️ ROLLBACK realizado."
            )

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
