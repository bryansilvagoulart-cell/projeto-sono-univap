from getpass import getpass
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.io import loadmat

import mysql.connector
from mysql.connector import Error


# ============================================================
# CONFIGURAÇÃO
# ============================================================

HOST = "127.0.0.1"
PORT = 3306
DATABASE = "projeto_sono"

USUARIO_PADRAO = "projeto_sono_app"

PARTICIPANTES_TESTE = [
    "Bidslab01",
    "Bidslab35",
    "Bidslab47"
]

NOITES_TESTE = [
    1,
    2,
    3,
    4,
    5
]

QUANTIDADE_EPOCAS_ESPERADA = 12252

ARQUIVO_QC = Path(
    "data/processed/teste_final_4_classes/"
    "dataset_teste_final_qc.csv"
)

PASTA_RAW = Path(
    "data/raw/participantes"
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def para_datetime_mysql(valor):

    instante = pd.to_datetime(
        valor,
        utc=True
    )

    return instante.to_pydatetime().replace(
        tzinfo=None
    )


def carregar_labels_noite(
    participante,
    noite
):

    caminho = (
        PASTA_RAW
        / participante
        / f"noite_{noite}"
        / "labels.mat"
    )


    if not caminho.exists():

        raise FileNotFoundError(
            f"labels.mat não encontrado:\n"
            f"{caminho}"
        )


    dados = loadmat(
        caminho
    )


    if "expert_label" not in dados:

        raise KeyError(
            f"expert_label ausente em:\n"
            f"{caminho}"
        )


    if "recStart" not in dados:

        raise KeyError(
            f"recStart ausente em:\n"
            f"{caminho}"
        )


    labels = (
        dados[
            "expert_label"
        ]
        .flatten()
        .astype(int)
    )


    rec_start_texto = (
        dados[
            "recStart"
        ]
        .item()
    )


    rec_start_local = datetime.strptime(
        rec_start_texto,
        "%Y-%m-%d %H:%M:%S"
    ).replace(
        tzinfo=ZoneInfo(
            "America/New_York"
        )
    )


    rec_start_utc = (
        rec_start_local
        .astimezone(
            ZoneInfo("UTC")
        )
        .replace(
            tzinfo=None
        )
    )


    fim_utc = (
        rec_start_utc
        + timedelta(
            seconds=len(labels) * 30
        )
    )


    return (
        labels,
        rec_start_utc,
        fim_utc
    )


def obter_id_participante(
    cursor,
    codigo
):

    cursor.execute(
        """
        SELECT id_participante
        FROM participantes
        WHERE codigo = %s
        """,
        (
            codigo,
        )
    )


    resultado = cursor.fetchone()


    if resultado is not None:

        return int(
            resultado[0]
        )


    cursor.execute(
        """
        INSERT INTO participantes (
            codigo
        )
        VALUES (%s)
        """,
        (
            codigo,
        )
    )


    return int(
        cursor.lastrowid
    )


def obter_id_noite(
    cursor,
    id_participante,
    numero_noite,
    inicio_utc,
    fim_utc,
    epocas_originais,
    epocas_unknown,
    epocas_validas,
    epocas_removidas,
    cobertura_percentual
):

    cursor.execute(
        """
        SELECT id_noite
        FROM noites
        WHERE id_participante = %s
          AND numero_noite = %s
        """,
        (
            id_participante,
            numero_noite
        )
    )


    resultados = cursor.fetchall()


    if len(resultados) > 1:

        raise RuntimeError(
            "Foram encontradas noites duplicadas "
            f"para id_participante={id_participante}, "
            f"numero_noite={numero_noite}."
        )


    if len(resultados) == 1:

        id_noite = int(
            resultados[0][0]
        )


        cursor.execute(
            """
            UPDATE noites
            SET
                inicio_utc = %s,
                fim_utc = %s,
                epocas_originais = %s,
                epocas_unknown = %s,
                epocas_validas = %s,
                epocas_removidas = %s,
                cobertura_percentual = %s
            WHERE id_noite = %s
            """,
            (
                inicio_utc,
                fim_utc,
                epocas_originais,
                epocas_unknown,
                epocas_validas,
                epocas_removidas,
                cobertura_percentual,
                id_noite
            )
        )


        return id_noite


    cursor.execute(
        """
        INSERT INTO noites (
            id_participante,
            numero_noite,
            inicio_utc,
            fim_utc,
            epocas_originais,
            epocas_unknown,
            epocas_validas,
            epocas_removidas,
            cobertura_percentual
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        """,
        (
            id_participante,
            numero_noite,
            inicio_utc,
            fim_utc,
            epocas_originais,
            epocas_unknown,
            epocas_validas,
            epocas_removidas,
            cobertura_percentual
        )
    )


    return int(
        cursor.lastrowid
    )


def validar_estagios_sono(
    cursor
):

    cursor.execute(
        """
        SELECT
            id_estagio,
            codigo
        FROM estagios_sono
        ORDER BY id_estagio
        """
    )


    registros = cursor.fetchall()


    encontrados = {
        int(
            id_estagio
        ): str(
            codigo
        )
        for (
            id_estagio,
            codigo
        ) in registros
    }


    esperados = {
        0: "WAKE",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM"
    }


    for id_estagio, codigo in esperados.items():

        if id_estagio not in encontrados:

            raise RuntimeError(
                f"Falta id_estagio={id_estagio} "
                "em estagios_sono."
            )


        if encontrados[
            id_estagio
        ].upper() != codigo:

            raise RuntimeError(
                "Mapeamento inesperado em "
                "estagios_sono:\n"
                f"id {id_estagio}: "
                f"{encontrados[id_estagio]}"
            )


# ============================================================
# INÍCIO
# ============================================================

print(
    "=" * 100
)

print(
    "IMPORTAÇÃO DO TESTE FINAL PARA O MYSQL - PARTICIPANTES / NOITES / ÉPOCAS"
)

print(
    "=" * 100
)


print(
    "\nEsta etapa importa SOMENTE:"
)

print(
    "  - Bidslab01"
)

print(
    "  - Bidslab35"
)

print(
    "  - Bidslab47"
)

print(
    "  - 15 noites"
)

print(
    "  - 12.252 épocas aprovadas pelo QC"
)

print(
    "\nAinda NÃO serão importadas "
    "as predições do modelo."
)


# ============================================================
# VALIDAR ARQUIVO
# ============================================================

if not ARQUIVO_QC.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado:\n"
        f"{ARQUIVO_QC}"
    )


dataset = pd.read_csv(
    ARQUIVO_QC
)


colunas_necessarias = [
    "epoca",
    "participante",
    "noite",
    "inicio",
    "fim",
    "label",
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",
    "hr_samples",
    "motion_mean",
    "motion_std",
    "motion_max",
    "motion_samples",
    "duracao_coberta",
    "cobertura_temporal_pct",
    "maior_gap"
]


faltando = [
    coluna
    for coluna in colunas_necessarias
    if coluna not in dataset.columns
]


if faltando:

    raise RuntimeError(
        "Colunas ausentes no dataset:\n"
        + "\n".join(
            faltando
        )
    )


if len(dataset) != QUANTIDADE_EPOCAS_ESPERADA:

    raise RuntimeError(
        f"Esperávamos "
        f"{QUANTIDADE_EPOCAS_ESPERADA} épocas, "
        f"mas encontramos {len(dataset)}."
    )


participantes_encontrados = sorted(
    dataset[
        "participante"
    ]
    .astype(str)
    .unique()
    .tolist()
)


if participantes_encontrados != sorted(
    PARTICIPANTES_TESTE
):

    raise RuntimeError(
        "Os participantes do dataset "
        "não são os participantes congelados.\n"
        f"Encontrados: {participantes_encontrados}"
    )


quantidade_noites = (
    dataset[
        [
            "participante",
            "noite"
        ]
    ]
    .drop_duplicates()
    .shape[
        0
    ]
)


if quantidade_noites != 15:

    raise RuntimeError(
        f"Esperávamos 15 noites, "
        f"mas encontramos {quantidade_noites}."
    )


labels_encontrados = set(
    dataset[
        "label"
    ]
    .astype(int)
    .unique()
)


if not labels_encontrados.issubset(
    {
        0,
        1,
        2,
        3,
        4
    }
):

    raise RuntimeError(
        "Foram encontrados labels inválidos "
        "no dataset de referência."
    )


if (
    dataset[
        colunas_necessarias
    ]
    .isna()
    .sum()
    .sum()
    != 0
):

    raise RuntimeError(
        "Existem valores ausentes "
        "nas colunas necessárias."
    )


duplicadas = (
    dataset
    .duplicated(
        subset=[
            "participante",
            "noite",
            "epoca"
        ]
    )
    .sum()
)


if duplicadas != 0:

    raise RuntimeError(
        f"Foram encontradas "
        f"{duplicadas} épocas duplicadas."
    )


print(
    "\n✅ Dataset validado:"
)

print(
    f"  Participantes: "
    f"{len(participantes_encontrados)}"
)

print(
    f"  Noites: "
    f"{quantidade_noites}"
)

print(
    f"  Épocas: "
    f"{len(dataset)}"
)


# ============================================================
# CONEXÃO
# ============================================================

usuario = input(
    f"\nUsuário MySQL [{USUARIO_PADRAO}]: "
).strip()


if not usuario:

    usuario = USUARIO_PADRAO


senha = getpass(
    f"Senha do usuário {usuario}: "
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

        user=usuario,

        password=senha,

        database=DATABASE,

        autocommit=False
    )


    cursor = conexao.cursor()


    print(
        "✅ Conexão realizada."
    )


    # ========================================================
    # VALIDAR MAPEAMENTO DE REFERÊNCIA
    # ========================================================

    print(
        "\nValidando catálogo original "
        "Wake / N1 / N2 / N3 / REM..."
    )


    validar_estagios_sono(
        cursor
    )


    print(
        "✅ estagios_sono validada."
    )


    # ========================================================
    # ESTADO ANTES
    # ========================================================

    contagens_antes = {}


    for tabela in [
        "participantes",
        "noites",
        "epocas"
    ]:

        cursor.execute(
            f"SELECT COUNT(*) FROM `{tabela}`"
        )

        contagens_antes[
            tabela
        ] = int(
            cursor.fetchone()[0]
        )


    print(
        "\nEstado antes da importação:"
    )


    print(
        f"  participantes: "
        f"{contagens_antes['participantes']}"
    )

    print(
        f"  noites: "
        f"{contagens_antes['noites']}"
    )

    print(
        f"  epocas: "
        f"{contagens_antes['epocas']}"
    )


    # ========================================================
    # IMPORTAR
    # ========================================================

    ids_participantes = {}

    ids_noites = {}

    total_epocas_importadas = 0


    for participante in PARTICIPANTES_TESTE:

        print(
            "\n"
            + "=" * 100
        )

        print(
            participante
        )

        print(
            "=" * 100
        )


        id_participante = (
            obter_id_participante(
                cursor,
                participante
            )
        )


        ids_participantes[
            participante
        ] = id_participante


        print(
            f"ID participante: "
            f"{id_participante}"
        )


        for noite in NOITES_TESTE:

            grupo = dataset[
                (
                    dataset[
                        "participante"
                    ]
                    == participante
                )
                &
                (
                    dataset[
                        "noite"
                    ]
                    == noite
                )
            ].copy()


            if len(grupo) == 0:

                raise RuntimeError(
                    f"Sem épocas para "
                    f"{participante}, noite {noite}."
                )


            (
                labels_originais,
                inicio_utc,
                fim_utc
            ) = carregar_labels_noite(
                participante,
                noite
            )


            epocas_originais = int(
                len(
                    labels_originais
                )
            )


            epocas_unknown = int(
                (
                    labels_originais
                    == 5
                ).sum()
            )


            epocas_utilizaveis = (
                epocas_originais
                - epocas_unknown
            )


            epocas_validas = int(
                len(
                    grupo
                )
            )


            epocas_removidas = (
                epocas_utilizaveis
                - epocas_validas
            )


            if epocas_removidas < 0:

                raise RuntimeError(
                    "Quantidade impossível de épocas "
                    f"em {participante}, noite {noite}."
                )


            if epocas_utilizaveis > 0:

                cobertura_percentual = (
                    epocas_validas
                    / epocas_utilizaveis
                    * 100.0
                )

            else:

                cobertura_percentual = 0.0


            id_noite = obter_id_noite(

                cursor,

                id_participante,

                noite,

                inicio_utc,

                fim_utc,

                epocas_originais,

                epocas_unknown,

                epocas_validas,

                epocas_removidas,

                round(
                    cobertura_percentual,
                    3
                )
            )


            ids_noites[
                (
                    participante,
                    noite
                )
            ] = id_noite


            print(
                f"\n  Noite {noite}"
            )


            print(
                f"    id_noite: "
                f"{id_noite}"
            )


            print(
                f"    épocas originais: "
                f"{epocas_originais}"
            )


            print(
                f"    unknown: "
                f"{epocas_unknown}"
            )


            print(
                f"    válidas após QC: "
                f"{epocas_validas}"
            )


            print(
                f"    removidas pelo QC: "
                f"{epocas_removidas}"
            )


            print(
                f"    cobertura: "
                f"{cobertura_percentual:.2f}%"
            )


            # =================================================
            # IMPORTAR ÉPOCAS
            # =================================================

            linhas = []


            grupo = (
                grupo
                .sort_values(
                    "epoca"
                )
                .reset_index(
                    drop=True
                )
            )


            for _, linha in grupo.iterrows():

                linhas.append(
                    (
                        id_noite,

                        int(
                            linha[
                                "epoca"
                            ]
                        ),

                        para_datetime_mysql(
                            linha[
                                "inicio"
                            ]
                        ),

                        para_datetime_mysql(
                            linha[
                                "fim"
                            ]
                        ),

                        float(
                            linha[
                                "hr_mean"
                            ]
                        ),

                        float(
                            linha[
                                "hr_min"
                            ]
                        ),

                        float(
                            linha[
                                "hr_max"
                            ]
                        ),

                        float(
                            linha[
                                "hr_std"
                            ]
                        ),

                        int(
                            linha[
                                "hr_samples"
                            ]
                        ),

                        float(
                            linha[
                                "motion_mean"
                            ]
                        ),

                        float(
                            linha[
                                "motion_std"
                            ]
                        ),

                        float(
                            linha[
                                "motion_max"
                            ]
                        ),

                        int(
                            linha[
                                "motion_samples"
                            ]
                        ),

                        float(
                            linha[
                                "duracao_coberta"
                            ]
                        ),

                        float(
                            linha[
                                "cobertura_temporal_pct"
                            ]
                        ),

                        float(
                            linha[
                                "maior_gap"
                            ]
                        ),

                        int(
                            linha[
                                "label"
                            ]
                        )
                    )
                )


            cursor.executemany(
                """
                INSERT INTO epocas (

                    id_noite,
                    numero_epoca,
                    inicio_utc,
                    fim_utc,

                    hr_mean,
                    hr_min,
                    hr_max,
                    hr_std,
                    hr_samples,

                    motion_mean,
                    motion_std,
                    motion_max,
                    motion_samples,

                    duracao_coberta,
                    cobertura_temporal_pct,
                    maior_gap,

                    id_estagio_referencia
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s
                )

                ON DUPLICATE KEY UPDATE

                    inicio_utc =
                        VALUES(inicio_utc),

                    fim_utc =
                        VALUES(fim_utc),

                    hr_mean =
                        VALUES(hr_mean),

                    hr_min =
                        VALUES(hr_min),

                    hr_max =
                        VALUES(hr_max),

                    hr_std =
                        VALUES(hr_std),

                    hr_samples =
                        VALUES(hr_samples),

                    motion_mean =
                        VALUES(motion_mean),

                    motion_std =
                        VALUES(motion_std),

                    motion_max =
                        VALUES(motion_max),

                    motion_samples =
                        VALUES(motion_samples),

                    duracao_coberta =
                        VALUES(duracao_coberta),

                    cobertura_temporal_pct =
                        VALUES(cobertura_temporal_pct),

                    maior_gap =
                        VALUES(maior_gap),

                    id_estagio_referencia =
                        VALUES(id_estagio_referencia)
                """,
                linhas
            )


            total_epocas_importadas += len(
                linhas
            )


            print(
                f"    épocas enviadas ao MySQL: "
                f"{len(linhas)}"
            )


    # ========================================================
    # VALIDAR ANTES DO COMMIT
    # ========================================================

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


    placeholders = ", ".join(
        ["%s"] * len(
            PARTICIPANTES_TESTE
        )
    )


    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM participantes
        WHERE codigo IN ({placeholders})
        """,
        tuple(
            PARTICIPANTES_TESTE
        )
    )


    qtd_participantes_banco = int(
        cursor.fetchone()[0]
    )


    if qtd_participantes_banco != 3:

        raise RuntimeError(
            f"Esperávamos 3 participantes novos, "
            f"mas encontramos "
            f"{qtd_participantes_banco}."
        )


    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM noites n
        JOIN participantes p
          ON p.id_participante = n.id_participante
        WHERE p.codigo IN ({placeholders})
        """,
        tuple(
            PARTICIPANTES_TESTE
        )
    )


    qtd_noites_banco = int(
        cursor.fetchone()[0]
    )


    if qtd_noites_banco != 15:

        raise RuntimeError(
            f"Esperávamos 15 noites no banco, "
            f"mas encontramos "
            f"{qtd_noites_banco}."
        )


    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM epocas e
        JOIN noites n
          ON n.id_noite = e.id_noite
        JOIN participantes p
          ON p.id_participante = n.id_participante
        WHERE p.codigo IN ({placeholders})
        """,
        tuple(
            PARTICIPANTES_TESTE
        )
    )


    qtd_epocas_banco = int(
        cursor.fetchone()[0]
    )


    if qtd_epocas_banco != QUANTIDADE_EPOCAS_ESPERADA:

        raise RuntimeError(
            f"Esperávamos "
            f"{QUANTIDADE_EPOCAS_ESPERADA} "
            f"épocas no banco, "
            f"mas encontramos "
            f"{qtd_epocas_banco}."
        )


    if (
        total_epocas_importadas
        != QUANTIDADE_EPOCAS_ESPERADA
    ):

        raise RuntimeError(
            f"Foram processadas "
            f"{total_epocas_importadas} épocas, "
            "quantidade inesperada."
        )


    # ========================================================
    # COMMIT
    # ========================================================

    conexao.commit()


    print(
        "\n✅ Validação concluída."
    )


    print(
        "✅ COMMIT realizado."
    )


    # ========================================================
    # ESTADO FINAL
    # ========================================================

    contagens_depois = {}


    for tabela in [
        "participantes",
        "noites",
        "epocas"
    ]:

        cursor.execute(
            f"SELECT COUNT(*) FROM `{tabela}`"
        )

        contagens_depois[
            tabela
        ] = int(
            cursor.fetchone()[0]
        )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "IMPORTAÇÃO CONCLUÍDA"
    )

    print(
        "=" * 100
    )


    print(
        "\nBanco antes → depois:"
    )


    print(
        f"  participantes: "
        f"{contagens_antes['participantes']} "
        f"→ "
        f"{contagens_depois['participantes']}"
    )


    print(
        f"  noites: "
        f"{contagens_antes['noites']} "
        f"→ "
        f"{contagens_depois['noites']}"
    )


    print(
        f"  epocas: "
        f"{contagens_antes['epocas']} "
        f"→ "
        f"{contagens_depois['epocas']}"
    )


    print(
        "\nConjunto novo confirmado:"
    )


    print(
        "  Participantes: 3"
    )


    print(
        "  Noites: 15"
    )


    print(
        f"  Épocas: "
        f"{qtd_epocas_banco}"
    )


    print(
        "\n✅ Referência original "
        "Wake/N1/N2/N3/REM preservada em epocas."
    )


    print(
        "✅ Ainda não foram inseridas "
        "predições do modelo de 4 classes."
    )


    print(
        "\nPróxima etapa:"
    )


    print(
        "Registrar o modelo final de 4 classes "
        "em modelos_ml e importar as 12.252 predições."
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
