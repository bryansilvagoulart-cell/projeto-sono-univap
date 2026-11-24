from pathlib import Path
from getpass import getpass

import pandas as pd

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
# MODELO FINAL
# ============================================================

ID_MODELO_FINAL = 1

NOME_MODELO_ESPERADO = (
    "Ensemble RF + BiGRU + Viterbi"
)

VERSAO_MODELO_ESPERADA = "1.0"


# ============================================================
# ARQUIVO DE PREDIÇÕES
# ============================================================

ARQUIVO_PREDICOES = Path(
    "data/processed/resultados_ml/teste_final_predicoes.csv"
)


# ============================================================
# QUANTIDADE ESPERADA
# ============================================================

QUANTIDADE_ESPERADA = 12061


# ============================================================
# TAMANHO DOS LOTES DE INSERT
# ============================================================

TAMANHO_LOTE = 1000


# ============================================================
# INÍCIO
# ============================================================

print("=" * 95)
print("INSERÇÃO DAS PREDIÇÕES FINAIS NO MYSQL")
print("=" * 95)


print(
    f"\nModelo final ID: "
    f"{ID_MODELO_FINAL}"
)


print(
    f"Arquivo:"
)

print(
    ARQUIVO_PREDICOES
)


# ============================================================
# 1. VALIDAR ARQUIVO
# ============================================================

if not ARQUIVO_PREDICOES.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado:\n"
        f"{ARQUIVO_PREDICOES}"
    )


print(
    "\nCarregando predições..."
)


predicoes = pd.read_csv(
    ARQUIVO_PREDICOES
)


print(
    f"Linhas encontradas: "
    f"{len(predicoes)}"
)


if len(predicoes) != QUANTIDADE_ESPERADA:

    raise ValueError(
        f"Esperávamos {QUANTIDADE_ESPERADA} predições, "
        f"mas encontramos {len(predicoes)}."
    )


# ============================================================
# 2. VALIDAR COLUNAS
# ============================================================

COLUNAS_NECESSARIAS = [
    "participante",
    "noite",
    "epoca",
    "id_noite",
    "segmento",
    "label",
    "predicao",
    "confianca"
]


faltando = [
    coluna
    for coluna in COLUNAS_NECESSARIAS
    if coluna not in predicoes.columns
]


if faltando:

    raise ValueError(
        "Colunas ausentes no CSV:\n"
        + "\n".join(
            faltando
        )
    )


print(
    "✅ Colunas validadas."
)


# ============================================================
# 3. CONVERTER TIPOS
# ============================================================

predicoes[
    "id_noite"
] = predicoes[
    "id_noite"
].astype(
    int
)


predicoes[
    "epoca"
] = predicoes[
    "epoca"
].astype(
    int
)


predicoes[
    "predicao"
] = predicoes[
    "predicao"
].astype(
    int
)


predicoes[
    "label"
] = predicoes[
    "label"
].astype(
    int
)


predicoes[
    "confianca"
] = predicoes[
    "confianca"
].astype(
    float
)


# ============================================================
# 4. VALIDAR ESTÁGIOS PREVISTOS
# ============================================================

ids_previstos = set(
    predicoes[
        "predicao"
    ].unique()
)


ids_esperados = {
    0,
    1,
    2,
    3,
    4
}


if not ids_previstos.issubset(
    ids_esperados
):

    raise ValueError(
        "Foram encontradas previsões "
        "fora das classes 0, 1, 2, 3 e 4."
    )


print(
    "✅ Classes previstas validadas."
)


# ============================================================
# 5. VALIDAR CONFIANÇA
# ============================================================

if predicoes[
    "confianca"
].isna().any():

    raise ValueError(
        "Existem valores de confiança ausentes."
    )


if (
    (
        predicoes[
            "confianca"
        ] < 0
    )
    |
    (
        predicoes[
            "confianca"
        ] > 1
    )
).any():

    raise ValueError(
        "Existem valores de confiança "
        "fora do intervalo [0, 1]."
    )


print(
    "✅ Confianças validadas."
)


# ============================================================
# 6. VALIDAR DUPLICIDADE NO CSV
# ============================================================

duplicadas_csv = int(

    predicoes.duplicated(
        subset=[
            "id_noite",
            "epoca"
        ]
    ).sum()
)


if duplicadas_csv != 0:

    raise ValueError(
        f"Foram encontradas {duplicadas_csv} "
        f"predições duplicadas no CSV."
    )


print(
    "✅ Nenhuma duplicidade no arquivo."
)


# ============================================================
# 7. RESUMO DO ARQUIVO
# ============================================================

print(
    "\nDistribuição das previsões:"
)


print(

    predicoes[
        "predicao"
    ]

    .value_counts()

    .sort_index()

    .to_string()
)


print(
    "\nConfiança experimental:"
)


print(

    predicoes[
        "confianca"
    ]

    .describe()

    .round(
        4
    )

    .to_string()
)


# ============================================================
# 8. SENHA
# ============================================================

senha = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


conexao = None
cursor = None


try:

    # ========================================================
    # 9. CONECTAR
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
    # 10. VALIDAR MODELO FINAL
    # ========================================================

    print(
        "\nValidando modelo final no banco..."
    )


    cursor.execute(
        """
        SELECT
            id_modelo,
            nome,
            versao,
            accuracy,
            macro_f1
        FROM modelos_ml
        WHERE id_modelo = %s
        """,
        (
            ID_MODELO_FINAL,
        )
    )


    modelo = cursor.fetchone()


    if modelo is None:

        raise RuntimeError(
            f"O modelo ID {ID_MODELO_FINAL} "
            f"não existe em modelos_ml."
        )


    if modelo[
        "nome"
    ] != NOME_MODELO_ESPERADO:

        raise RuntimeError(
            "O nome do modelo encontrado "
            "não corresponde ao modelo final esperado."
        )


    if modelo[
        "versao"
    ] != VERSAO_MODELO_ESPERADA:

        raise RuntimeError(
            "A versão do modelo encontrado "
            "não corresponde à versão esperada."
        )


    print(
        "✅ Modelo final confirmado."
    )


    print(
        f"  ID: "
        f"{modelo['id_modelo']}"
    )


    print(
        f"  Nome: "
        f"{modelo['nome']}"
    )


    print(
        f"  Versão: "
        f"{modelo['versao']}"
    )


    # ========================================================
    # 11. VALIDAR ESTÁGIOS NO BANCO
    # ========================================================

    cursor.execute(
        """
        SELECT
            id_estagio,
            codigo
        FROM estagios_sono
        ORDER BY id_estagio
        """
    )


    estagios_banco = cursor.fetchall()


    ids_banco = {
        int(
            linha[
                "id_estagio"
            ]
        )
        for linha in estagios_banco
    }


    if ids_banco != ids_esperados:

        raise RuntimeError(
            "Os IDs dos estágios cadastrados "
            "no banco não correspondem a "
            "{0, 1, 2, 3, 4}."
        )


    print(
        "✅ Estágios do banco confirmados."
    )


    # ========================================================
    # 12. VERIFICAR PREDIÇÕES JÁ EXISTENTES
    # ========================================================

    cursor.execute(
        """
        SELECT
            COUNT(*) AS quantidade
        FROM predicoes
        WHERE id_modelo = %s
        """,
        (
            ID_MODELO_FINAL,
        )
    )


    existentes = int(
        cursor.fetchone()[
            "quantidade"
        ]
    )


    print(
        f"\nPredições já existentes para este modelo: "
        f"{existentes}"
    )


    if existentes not in [
        0,
        QUANTIDADE_ESPERADA
    ]:

        raise RuntimeError(
            "Existe uma quantidade parcial inesperada "
            "de predições deste modelo no banco. "
            "A inserção foi cancelada para evitar inconsistência."
        )


    # ========================================================
    # 13. SE JÁ ESTIVER COMPLETO, APENAS VALIDAR
    # ========================================================

    if existentes == QUANTIDADE_ESPERADA:

        print(
            "\n⚠️ As 12.061 predições deste modelo "
            "já estão cadastradas."
        )


        print(
            "Nenhuma nova linha será inserida."
        )


    # ========================================================
    # 14. SE NÃO EXISTIR, MAPEAR ÉPOCAS
    # ========================================================

    else:

        print(
            "\nBuscando as épocas correspondentes no banco..."
        )


        # ----------------------------------------------------
        # Buscar somente os IDs de noite presentes no teste.
        # ----------------------------------------------------

        ids_noites_teste = sorted(

            predicoes[
                "id_noite"
            ].unique()
        )


        placeholders = ",".join(
            ["%s"] * len(
                ids_noites_teste
            )
        )


        consulta_epocas = f"""
        SELECT
            id_epoca,
            id_noite,
            numero_epoca,
            id_estagio_referencia
        FROM epocas
        WHERE id_noite IN ({placeholders})
        """


        cursor.execute(
            consulta_epocas,
            tuple(
                ids_noites_teste
            )
        )


        epocas_banco = (
            cursor.fetchall()
        )


        print(
            f"Épocas retornadas pelo banco: "
            f"{len(epocas_banco)}"
        )


        # ====================================================
        # 15. CRIAR MAPA
        # ====================================================

        mapa_epocas = {}


        for linha in epocas_banco:


            chave = (
                int(
                    linha[
                        "id_noite"
                    ]
                ),

                int(
                    linha[
                        "numero_epoca"
                    ]
                )
            )


            if chave in mapa_epocas:

                raise RuntimeError(
                    "Foi encontrada uma época duplicada "
                    "no banco."
                )


            mapa_epocas[
                chave
            ] = {
                "id_epoca":
                    int(
                        linha[
                            "id_epoca"
                        ]
                    ),

                "referencia":
                    int(
                        linha[
                            "id_estagio_referencia"
                        ]
                    )
            }


        # ====================================================
        # 16. ASSOCIAR CADA LINHA DO CSV AO BANCO
        # ====================================================

        print(
            "\nAssociando CSV às épocas do banco..."
        )


        registros_insert = []


        nao_encontradas = []

        referencias_diferentes = []


        for linha in predicoes.itertuples(
            index=False
        ):


            chave = (
                int(
                    linha.id_noite
                ),

                int(
                    linha.epoca
                )
            )


            dados_epoca = mapa_epocas.get(
                chave
            )


            if dados_epoca is None:

                nao_encontradas.append(
                    chave
                )

                continue


            # ------------------------------------------------
            # Segurança adicional:
            #
            # a referência do CSV deve ser igual à
            # referência armazenada em epocas.
            # ------------------------------------------------

            if (
                int(
                    linha.label
                )
                !=
                dados_epoca[
                    "referencia"
                ]
            ):

                referencias_diferentes.append(
                    (
                        chave,
                        int(
                            linha.label
                        ),
                        dados_epoca[
                            "referencia"
                        ]
                    )
                )


            registros_insert.append(
                (
                    dados_epoca[
                        "id_epoca"
                    ],

                    ID_MODELO_FINAL,

                    int(
                        linha.predicao
                    ),

                    float(
                        linha.confianca
                    )
                )
            )


        # ====================================================
        # 17. VALIDAR MAPEAMENTO
        # ====================================================

        print(
            f"Predições associadas: "
            f"{len(registros_insert)}"
        )


        print(
            f"Épocas não encontradas: "
            f"{len(nao_encontradas)}"
        )


        print(
            f"Referências divergentes: "
            f"{len(referencias_diferentes)}"
        )


        if nao_encontradas:

            print(
                "\nPrimeiras épocas não encontradas:"
            )


            for item in nao_encontradas[
                :10
            ]:

                print(
                    f"  {item}"
                )


            raise RuntimeError(
                "Nem todas as épocas do CSV "
                "foram encontradas no banco."
            )


        if referencias_diferentes:

            print(
                "\nPrimeiras divergências:"
            )


            for item in referencias_diferentes[
                :10
            ]:

                print(
                    f"  {item}"
                )


            raise RuntimeError(
                "A referência do CSV não corresponde "
                "à referência armazenada no banco."
            )


        if (
            len(
                registros_insert
            )
            != QUANTIDADE_ESPERADA
        ):

            raise RuntimeError(
                "A quantidade de registros preparados "
                "não corresponde a 12061."
            )


        print(
            "✅ Todas as épocas foram associadas corretamente."
        )


        # ====================================================
        # 18. INSERT
        # ====================================================

        print(
            "\nInserindo predições no banco..."
        )


        sql_insert = """
        INSERT INTO predicoes
        (
            id_epoca,
            id_modelo,
            id_estagio_previsto,
            confianca
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s
        )
        """


        inseridas = 0


        for inicio in range(
            0,
            len(registros_insert),
            TAMANHO_LOTE
        ):


            lote = registros_insert[
                inicio:
                inicio + TAMANHO_LOTE
            ]


            cursor.executemany(
                sql_insert,
                lote
            )


            inseridas += len(
                lote
            )


            print(
                f"  {inseridas}/"
                f"{QUANTIDADE_ESPERADA}"
                f" inseridas..."
            )


        # ====================================================
        # 19. COMMIT SOMENTE DEPOIS DE TUDO
        # ====================================================

        conexao.commit()


        print(
            "✅ Inserção concluída e confirmada."
        )


    # ========================================================
    # 20. VALIDAÇÃO FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "VALIDAÇÃO FINAL DA TABELA PREDICOES"
    )

    print(
        "=" * 95
    )


    cursor.execute(
        """
        SELECT
            COUNT(*) AS quantidade
        FROM predicoes
        WHERE id_modelo = %s
        """,
        (
            ID_MODELO_FINAL,
        )
    )


    total_predicoes = int(
        cursor.fetchone()[
            "quantidade"
        ]
    )


    print(
        f"\nPredições do modelo {ID_MODELO_FINAL}: "
        f"{total_predicoes}"
    )


    if total_predicoes != QUANTIDADE_ESPERADA:

        raise RuntimeError(
            f"Esperávamos {QUANTIDADE_ESPERADA} "
            f"predições no banco."
        )


    # ========================================================
    # 21. DISTRIBUIÇÃO POR ESTÁGIO
    # ========================================================

    cursor.execute(
        """
        SELECT
            e.id_estagio,
            e.codigo,
            e.nome,
            COUNT(*) AS quantidade
        FROM predicoes p
        INNER JOIN estagios_sono e
            ON e.id_estagio = p.id_estagio_previsto
        WHERE p.id_modelo = %s
        GROUP BY
            e.id_estagio,
            e.codigo,
            e.nome
        ORDER BY
            e.id_estagio
        """,
        (
            ID_MODELO_FINAL,
        )
    )


    distribuicao = cursor.fetchall()


    print(
        "\nDistribuição das previsões no banco:"
    )


    soma_distribuicao = 0


    for linha in distribuicao:


        quantidade = int(
            linha[
                "quantidade"
            ]
        )


        soma_distribuicao += quantidade


        percentual = (
            quantidade
            / total_predicoes
            * 100
        )


        print(
            f"  {linha['id_estagio']} "
            f"{linha['codigo']:<5} "
            f"-> "
            f"{quantidade:5d} "
            f"({percentual:6.2f}%)"
        )


    if soma_distribuicao != total_predicoes:

        raise RuntimeError(
            "A soma da distribuição não corresponde "
            "ao total de predições."
        )


    # ========================================================
    # 22. CONFIANÇA
    # ========================================================

    cursor.execute(
        """
        SELECT
            MIN(confianca) AS minimo,
            AVG(confianca) AS media,
            MAX(confianca) AS maximo
        FROM predicoes
        WHERE id_modelo = %s
        """,
        (
            ID_MODELO_FINAL,
        )
    )


    confianca_banco = cursor.fetchone()


    print(
        "\nConfiança experimental:"
    )


    print(
        f"  Mínima: "
        f"{float(confianca_banco['minimo']):.4f}"
    )


    print(
        f"  Média: "
        f"{float(confianca_banco['media']):.4f}"
    )


    print(
        f"  Máxima: "
        f"{float(confianca_banco['maximo']):.4f}"
    )


    # ========================================================
    # 23. CONFERIR QUANTIDADE DE PARTICIPANTES
    # ========================================================

    cursor.execute(
        """
        SELECT
            pa.codigo AS participante,
            COUNT(*) AS quantidade
        FROM predicoes pr
        INNER JOIN epocas ep
            ON ep.id_epoca = pr.id_epoca
        INNER JOIN noites n
            ON n.id_noite = ep.id_noite
        INNER JOIN participantes pa
            ON pa.id_participante = n.id_participante
        WHERE pr.id_modelo = %s
        GROUP BY pa.codigo
        ORDER BY pa.codigo
        """,
        (
            ID_MODELO_FINAL,
        )
    )


    participantes_banco = (
        cursor.fetchall()
    )


    print(
        "\nParticipantes com predições:"
    )


    for linha in participantes_banco:

        print(
            f"  {linha['participante']}: "
            f"{linha['quantidade']} épocas"
        )


    participantes_encontrados = {
        linha[
            "participante"
        ]
        for linha in participantes_banco
    }


    participantes_esperados = {
        "Bidslab16",
        "Bidslab22",
        "Bidslab32"
    }


    if (
        participantes_encontrados
        != participantes_esperados
    ):

        raise RuntimeError(
            "Os participantes presentes nas predições "
            "não correspondem ao conjunto de teste final."
        )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "PREDIÇÕES FINAIS REGISTRADAS COM SUCESSO"
    )

    print(
        "=" * 95
    )


    print(
        f"\nModelo: "
        f"{ID_MODELO_FINAL}"
    )


    print(
        f"Predições cadastradas: "
        f"{total_predicoes}"
    )


    print(
        f"Participantes: "
        f"{len(participantes_encontrados)}"
    )


    print(
        "\n✅ Épocas associadas."
    )

    print(
        "✅ Referências conferidas."
    )

    print(
        "✅ Estágios previstos gravados."
    )

    print(
        "✅ Confianças gravadas."
    )

    print(
        "✅ Quantidades conferidas."
    )


    print(
        "\nPRÓXIMA ETAPA:"
    )


    print(
        "Consultar as predições junto com noites, "
        "participantes e estágios para preparar a API."
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