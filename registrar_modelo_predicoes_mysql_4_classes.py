from getpass import getpass
from pathlib import Path
from datetime import datetime, timezone

import json
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

USUARIO_PADRAO = "projeto_sono_app"

TAXONOMIA = "sono_4_classes_v1"

NOME_MODELO = "Ensemble RF + BiGRU + Viterbi"
VERSAO_MODELO = "4_classes_final_1.0"
ALGORITMO_MODELO = "Random Forest + BiGRU + Viterbi"

QUANTIDADE_CLASSES = 4

PARTICIPANTES_TESTE = [
    "Bidslab01",
    "Bidslab35",
    "Bidslab47"
]

QUANTIDADE_PREDICOES_ESPERADA = 12252


# ============================================================
# ARQUIVOS
# ============================================================

ARQUIVO_PREDICOES = Path(
    "data/processed/resultados_ml_4_classes_final/"
    "teste_final_4_classes_predicoes.csv"
)

ARQUIVO_RESUMO = Path(
    "data/processed/resultados_ml_4_classes_final/"
    "teste_final_4_classes_resumo.csv"
)

ARQUIVO_MANIFESTO_MODELO = Path(
    "models/4_classes/"
    "manifesto_modelo_final_4_classes.json"
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def carregar_data_treinamento():

    if not ARQUIVO_MANIFESTO_MODELO.exists():

        return datetime.now(
            timezone.utc
        ).replace(
            tzinfo=None
        )


    with open(
        ARQUIVO_MANIFESTO_MODELO,
        "r",
        encoding="utf-8"
    ) as arquivo:

        manifesto = json.load(
            arquivo
        )


    texto = manifesto.get(
        "criado_em_utc"
    )


    if not texto:

        return datetime.now(
            timezone.utc
        ).replace(
            tzinfo=None
        )


    try:

        instante = datetime.fromisoformat(
            texto.replace(
                "Z",
                "+00:00"
            )
        )

        if instante.tzinfo is not None:

            instante = instante.astimezone(
                timezone.utc
            ).replace(
                tzinfo=None
            )


        return instante


    except Exception:

        return datetime.now(
            timezone.utc
        ).replace(
            tzinfo=None
        )


def validar_arquivos():

    arquivos = [
        ARQUIVO_PREDICOES,
        ARQUIVO_RESUMO
    ]


    faltando = [
        str(
            arquivo
        )
        for arquivo in arquivos
        if not arquivo.exists()
    ]


    if faltando:

        raise FileNotFoundError(
            "Arquivos necessários ausentes:\n"
            + "\n".join(
                faltando
            )
        )


def carregar_metricas():

    resumo = pd.read_csv(
        ARQUIVO_RESUMO
    )


    if len(resumo) != 1:

        raise RuntimeError(
            "O arquivo de resumo final deveria "
            "possuir exatamente uma linha."
        )


    linha = resumo.iloc[
        0
    ]


    return {
        "accuracy":
            float(
                linha[
                    "accuracy"
                ]
            ),

        "balanced_accuracy":
            float(
                linha[
                    "balanced_accuracy"
                ]
            ),

        "macro_f1":
            float(
                linha[
                    "macro_f1"
                ]
            ),

        "weighted_f1":
            float(
                linha[
                    "weighted_f1"
                ]
            )
    }


def validar_predicoes(
    predicoes
):

    colunas_necessarias = [
        "participante",
        "noite",
        "epoca",
        "label",
        "predicao",
        "classe_prevista",
        "confianca_experimental"
    ]


    faltando = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in predicoes.columns
    ]


    if faltando:

        raise RuntimeError(
            "Colunas ausentes no arquivo "
            "de predições:\n"
            + "\n".join(
                faltando
            )
        )


    if len(
        predicoes
    ) != QUANTIDADE_PREDICOES_ESPERADA:

        raise RuntimeError(
            f"Esperávamos "
            f"{QUANTIDADE_PREDICOES_ESPERADA} "
            f"predições, mas encontramos "
            f"{len(predicoes)}."
        )


    participantes = sorted(
        predicoes[
            "participante"
        ]
        .astype(str)
        .unique()
        .tolist()
    )


    if participantes != sorted(
        PARTICIPANTES_TESTE
    ):

        raise RuntimeError(
            "Participantes inesperados no "
            "arquivo de predições:\n"
            f"{participantes}"
        )


    if (
        predicoes[
            [
                "participante",
                "noite",
                "epoca"
            ]
        ]
        .duplicated()
        .sum()
        != 0
    ):

        raise RuntimeError(
            "Existem épocas duplicadas "
            "no arquivo de predições."
        )


    classes = set(
        predicoes[
            "predicao"
        ]
        .astype(int)
        .unique()
    )


    if not classes.issubset(
        {
            0,
            1,
            2,
            3
        }
    ):

        raise RuntimeError(
            "Foram encontradas predições "
            "fora das quatro classes."
        )


    confiancas = (
        predicoes[
            "confianca_experimental"
        ]
        .astype(float)
    )


    if (
        confiancas.isna().any()
        or
        (~np.isfinite(
            confiancas.to_numpy()
        )).any()
    ):

        raise RuntimeError(
            "Há confiança ausente ou inválida."
        )


    if (
        (confiancas < 0.0).any()
        or
        (confiancas > 1.0).any()
    ):

        raise RuntimeError(
            "Há confiança fora do intervalo [0, 1]."
        )


def obter_mapa_classes(
    cursor
):

    cursor.execute(
        """
        SELECT
            rotulo_modelo,
            id_classe_modelo
        FROM classes_modelo_ml
        WHERE taxonomia = %s
        ORDER BY rotulo_modelo
        """,
        (
            TAXONOMIA,
        )
    )


    registros = cursor.fetchall()


    mapa = {
        int(
            rotulo
        ): int(
            id_classe
        )
        for (
            rotulo,
            id_classe
        ) in registros
    }


    esperado = {
        0,
        1,
        2,
        3
    }


    if set(
        mapa.keys()
    ) != esperado:

        raise RuntimeError(
            "A taxonomia de 4 classes "
            "não está completa no banco."
        )


    return mapa


def obter_ou_registrar_modelo(
    cursor,
    metricas,
    data_treinamento
):

    cursor.execute(
        """
        SELECT id_modelo
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


    observacoes = (
        "Modelo final oficial de 4 classes. "
        "Treinado nos 15 participantes de desenvolvimento "
        "(60.165 épocas) com configuração congelada por "
        "validação cruzada: RF 0.75 + BiGRU 0.25 + "
        "Viterbi 0.80, janela BiGRU de 21 épocas e "
        "8 épocas de treinamento. Teste independente em "
        "Bidslab01, Bidslab35 e Bidslab47, totalizando "
        "12.252 épocas."
    )


    if resultado is not None:

        id_modelo = int(
            resultado[0]
        )


        cursor.execute(
            """
            UPDATE modelos_ml
            SET
                algoritmo = %s,
                data_treinamento = %s,
                accuracy = %s,
                balanced_accuracy = %s,
                macro_f1 = %s,
                weighted_f1 = %s,
                taxonomia = %s,
                quantidade_classes = %s,
                observacoes = %s
            WHERE id_modelo = %s
            """,
            (
                ALGORITMO_MODELO,
                data_treinamento,
                metricas[
                    "accuracy"
                ],
                metricas[
                    "balanced_accuracy"
                ],
                metricas[
                    "macro_f1"
                ],
                metricas[
                    "weighted_f1"
                ],
                TAXONOMIA,
                QUANTIDADE_CLASSES,
                observacoes,
                id_modelo
            )
        )


        return (
            id_modelo,
            False
        )


    cursor.execute(
        """
        INSERT INTO modelos_ml (
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
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        """,
        (
            NOME_MODELO,
            ALGORITMO_MODELO,
            VERSAO_MODELO,
            data_treinamento,
            metricas[
                "accuracy"
            ],
            metricas[
                "balanced_accuracy"
            ],
            metricas[
                "macro_f1"
            ],
            metricas[
                "weighted_f1"
            ],
            TAXONOMIA,
            QUANTIDADE_CLASSES,
            observacoes
        )
    )


    return (
        int(
            cursor.lastrowid
        ),
        True
    )


def carregar_mapa_epocas(
    cursor
):

    placeholders = ", ".join(
        ["%s"] * len(
            PARTICIPANTES_TESTE
        )
    )


    cursor.execute(
        f"""
        SELECT
            p.codigo,
            n.numero_noite,
            e.numero_epoca,
            e.id_epoca
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


    registros = cursor.fetchall()


    mapa = {}


    for (
        participante,
        noite,
        epoca,
        id_epoca
    ) in registros:

        chave = (
            str(
                participante
            ),
            int(
                noite
            ),
            int(
                epoca
            )
        )


        if chave in mapa:

            raise RuntimeError(
                "Chave de época duplicada "
                "no banco:\n"
                f"{chave}"
            )


        mapa[
            chave
        ] = int(
            id_epoca
        )


    if len(
        mapa
    ) != QUANTIDADE_PREDICOES_ESPERADA:

        raise RuntimeError(
            f"Esperávamos "
            f"{QUANTIDADE_PREDICOES_ESPERADA} "
            f"épocas do teste no banco, "
            f"mas encontramos {len(mapa)}."
        )


    return mapa


# ============================================================
# INÍCIO
# ============================================================

print(
    "=" * 100
)

print(
    "REGISTRO DO MODELO FINAL + PREDIÇÕES - 4 CLASSES"
)

print(
    "=" * 100
)


print(
    "\nModelo:"
)

print(
    f"  {NOME_MODELO}"
)

print(
    f"  versão: {VERSAO_MODELO}"
)

print(
    f"  taxonomia: {TAXONOMIA}"
)

print(
    "\nPredições esperadas:"
)

print(
    f"  {QUANTIDADE_PREDICOES_ESPERADA}"
)


# ============================================================
# VALIDAR ARQUIVOS LOCAIS
# ============================================================

validar_arquivos()


metricas = carregar_metricas()


predicoes = pd.read_csv(
    ARQUIVO_PREDICOES
)


validar_predicoes(
    predicoes
)


data_treinamento = (
    carregar_data_treinamento()
)


print(
    "\n✅ Arquivos locais validados."
)


print(
    "\nMétricas finais:"
)


print(
    f"  Accuracy: "
    f"{metricas['accuracy']:.4f}"
)


print(
    f"  Balanced Accuracy: "
    f"{metricas['balanced_accuracy']:.4f}"
)


print(
    f"  Macro F1: "
    f"{metricas['macro_f1']:.4f}"
)


print(
    f"  Weighted F1: "
    f"{metricas['weighted_f1']:.4f}"
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
    # MAPEAR CLASSES
    # ========================================================

    print(
        "\nCarregando catálogo "
        "Wake / Light / Deep / REM..."
    )


    mapa_classes = (
        obter_mapa_classes(
            cursor
        )
    )


    print(
        "✅ Catálogo carregado:"
    )


    for rotulo in sorted(
        mapa_classes
    ):

        print(
            f"  rótulo {rotulo} "
            f"→ id_classe_modelo "
            f"{mapa_classes[rotulo]}"
        )


    # ========================================================
    # REGISTRAR MODELO
    # ========================================================

    print(
        "\nRegistrando modelo final..."
    )


    (
        id_modelo,
        modelo_novo
    ) = obter_ou_registrar_modelo(

        cursor,

        metricas,

        data_treinamento
    )


    if modelo_novo:

        print(
            f"✅ Modelo inserido "
            f"com id_modelo={id_modelo}."
        )

    else:

        print(
            f"✅ Modelo já existia; "
            f"registro id_modelo={id_modelo} "
            f"foi atualizado."
        )


    # ========================================================
    # MAPEAR ÉPOCAS
    # ========================================================

    print(
        "\nMapeando as 12.252 épocas "
        "do teste para id_epoca..."
    )


    mapa_epocas = (
        carregar_mapa_epocas(
            cursor
        )
    )


    print(
        f"✅ {len(mapa_epocas)} épocas mapeadas."
    )


    # ========================================================
    # MONTAR LINHAS DE PREDIÇÃO
    # ========================================================

    linhas = []


    for _, linha in predicoes.iterrows():

        chave = (
            str(
                linha[
                    "participante"
                ]
            ),

            int(
                linha[
                    "noite"
                ]
            ),

            int(
                linha[
                    "epoca"
                ]
            )
        )


        if chave not in mapa_epocas:

            raise RuntimeError(
                "Época não encontrada "
                "no banco:\n"
                f"{chave}"
            )


        rotulo_previsto = int(
            linha[
                "predicao"
            ]
        )


        id_classe_modelo = (
            mapa_classes[
                rotulo_previsto
            ]
        )


        confianca = float(
            linha[
                "confianca_experimental"
            ]
        )


        linhas.append(
            (
                mapa_epocas[
                    chave
                ],

                id_modelo,

                id_classe_modelo,

                confianca
            )
        )


    if len(
        linhas
    ) != QUANTIDADE_PREDICOES_ESPERADA:

        raise RuntimeError(
            "Quantidade inesperada de "
            "linhas de predição."
        )


    # ========================================================
    # INSERIR PREDIÇÕES
    # ========================================================

    print(
        "\nInserindo predições..."
    )


    cursor.executemany(
        """
        INSERT INTO predicoes_modelo_ml (
            id_epoca,
            id_modelo,
            id_classe_modelo,
            confianca
        )
        VALUES (
            %s,
            %s,
            %s,
            %s
        )
        ON DUPLICATE KEY UPDATE
            id_classe_modelo =
                VALUES(id_classe_modelo),
            confianca =
                VALUES(confianca)
        """,
        linhas
    )


    print(
        "✅ Predições enviadas ao MySQL."
    )


    # ========================================================
    # VALIDAÇÃO ANTES DO COMMIT
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


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predicoes_modelo_ml
        WHERE id_modelo = %s
        """,
        (
            id_modelo,
        )
    )


    qtd_predicoes_modelo = int(
        cursor.fetchone()[0]
    )


    if (
        qtd_predicoes_modelo
        != QUANTIDADE_PREDICOES_ESPERADA
    ):

        raise RuntimeError(
            f"Esperávamos "
            f"{QUANTIDADE_PREDICOES_ESPERADA} "
            f"predições para o modelo, "
            f"mas encontramos "
            f"{qtd_predicoes_modelo}."
        )


    cursor.execute(
        """
        SELECT
            c.rotulo_modelo,
            c.codigo,
            COUNT(*) AS quantidade
        FROM predicoes_modelo_ml p
        JOIN classes_modelo_ml c
          ON c.id_classe_modelo =
             p.id_classe_modelo
        WHERE p.id_modelo = %s
        GROUP BY
            c.rotulo_modelo,
            c.codigo
        ORDER BY
            c.rotulo_modelo
        """,
        (
            id_modelo,
        )
    )


    distribuicao = cursor.fetchall()


    total_distribuicao = sum(
        int(
            linha[
                2
            ]
        )
        for linha in distribuicao
    )


    if (
        total_distribuicao
        != QUANTIDADE_PREDICOES_ESPERADA
    ):

        raise RuntimeError(
            "A distribuição das classes "
            "não fecha com o total."
        )


    cursor.execute(
        """
        SELECT
            MIN(confianca),
            MAX(confianca),
            AVG(confianca)
        FROM predicoes_modelo_ml
        WHERE id_modelo = %s
        """,
        (
            id_modelo,
        )
    )


    (
        confianca_min,
        confianca_max,
        confianca_media
    ) = cursor.fetchone()


    if (
        confianca_min is None
        or
        confianca_max is None
    ):

        raise RuntimeError(
            "Confiâncias não foram gravadas."
        )


    if (
        float(
            confianca_min
        ) < 0.0
        or
        float(
            confianca_max
        ) > 1.0
    ):

        raise RuntimeError(
            "Confiiança fora do intervalo [0, 1]."
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
    # RESULTADO
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MODELO FINAL E PREDIÇÕES REGISTRADOS"
    )

    print(
        "=" * 100
    )


    print(
        f"\nid_modelo: "
        f"{id_modelo}"
    )


    print(
        f"Nome: "
        f"{NOME_MODELO}"
    )


    print(
        f"Versão: "
        f"{VERSAO_MODELO}"
    )


    print(
        f"Taxonomia: "
        f"{TAXONOMIA}"
    )


    print(
        "\nMétricas armazenadas:"
    )


    print(
        f"  Accuracy: "
        f"{metricas['accuracy']:.4f}"
    )


    print(
        f"  Balanced Accuracy: "
        f"{metricas['balanced_accuracy']:.4f}"
    )


    print(
        f"  Macro F1: "
        f"{metricas['macro_f1']:.4f}"
    )


    print(
        f"  Weighted F1: "
        f"{metricas['weighted_f1']:.4f}"
    )


    print(
        "\nDistribuição das predições:"
    )


    for (
        rotulo,
        codigo,
        quantidade
    ) in distribuicao:

        print(
            f"  {codigo:<5} "
            f"(rótulo {rotulo}): "
            f"{quantidade}"
        )


    print(
        "\nConfiança experimental:"
    )


    print(
        f"  mínima: "
        f"{float(confianca_min):.4f}"
    )


    print(
        f"  média: "
        f"{float(confianca_media):.4f}"
    )


    print(
        f"  máxima: "
        f"{float(confianca_max):.4f}"
    )


    print(
        "\nPredições gravadas:"
    )


    print(
        f"  {qtd_predicoes_modelo}"
    )


    print(
        "\n✅ O modelo antigo de 5 classes "
        "permanece preservado."
    )


    print(
        "✅ As novas predições estão em "
        "predicoes_modelo_ml."
    )


    print(
        "\nPróxima etapa:"
    )


    print(
        "Criar as métricas agregadas por noite "
        "para Wake / Light / Deep / REM e "
        "depois atualizar a API."
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
