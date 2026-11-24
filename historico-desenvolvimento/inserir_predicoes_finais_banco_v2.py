from pathlib import Path
from getpass import getpass

import pandas as pd
import mysql.connector
from mysql.connector import Error


# ============================================================
# CONFIGURAÇÃO
# ============================================================

HOST = "127.0.0.1"
PORTA = 3306
BANCO = "projeto_sono"
USUARIO = "projeto_sono_app"

ID_MODELO_FINAL = 1

QUANTIDADE_ESPERADA = 12061

TAMANHO_LOTE = 1000

ARQUIVO_PREDICOES = Path(
    "data/processed/resultados_ml/teste_final_predicoes.csv"
)


# ============================================================
# INÍCIO
# ============================================================

print("=" * 95)
print("INSERÇÃO DAS PREDIÇÕES FINAIS NO MYSQL - V2")
print("=" * 95)


# ============================================================
# 1. CARREGAR CSV
# ============================================================

if not ARQUIVO_PREDICOES.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado:\n"
        f"{ARQUIVO_PREDICOES}"
    )


print(
    "\nCarregando predições..."
)


df = pd.read_csv(
    ARQUIVO_PREDICOES
)


print(
    f"Linhas encontradas: "
    f"{len(df)}"
)


if len(df) != QUANTIDADE_ESPERADA:

    raise ValueError(
        f"Esperávamos {QUANTIDADE_ESPERADA} linhas, "
        f"mas encontramos {len(df)}."
    )


# ============================================================
# 2. VALIDAR COLUNAS
# ============================================================

COLUNAS = [
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
    for coluna in COLUNAS
    if coluna not in df.columns
]


if faltando:

    raise ValueError(
        "Colunas ausentes:\n"
        + "\n".join(faltando)
    )


print(
    "✅ Colunas validadas."
)


# ============================================================
# 3. VALIDAR DADOS
# ============================================================

if df[
    [
        "id_noite",
        "epoca",
        "label",
        "predicao",
        "confianca"
    ]
].isna().any().any():

    raise ValueError(
        "Existem valores ausentes "
        "nas colunas essenciais."
    )


# Conversões explícitas

df["id_noite"] = df["id_noite"].astype(int)
df["epoca"] = df["epoca"].astype(int)
df["label"] = df["label"].astype(int)
df["predicao"] = df["predicao"].astype(int)
df["confianca"] = df["confianca"].astype(float)


classes = set(
    int(valor)
    for valor in df[
        "predicao"
    ].unique()
)


if not classes.issubset(
    {0, 1, 2, 3, 4}
):

    raise ValueError(
        f"Classes inválidas encontradas: "
        f"{classes}"
    )


if (
    (df["confianca"] < 0)
    |
    (df["confianca"] > 1)
).any():

    raise ValueError(
        "Existem valores de confiança "
        "fora do intervalo [0,1]."
    )


duplicadas = int(

    df.duplicated(
        subset=[
            "id_noite",
            "epoca"
        ]
    ).sum()
)


if duplicadas != 0:

    raise ValueError(
        f"Foram encontradas "
        f"{duplicadas} linhas duplicadas."
    )


print(
    "✅ Dados validados."
)


# ============================================================
# 4. RESUMO
# ============================================================

print(
    "\nDistribuição das previsões:"
)


print(

    df[
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

    df[
        "confianca"
    ]
    .describe()
    .round(4)
    .to_string()
)


# ============================================================
# 5. SENHA
# ============================================================

senha = getpass(
    "\nDigite a senha do usuário projeto_sono_app: "
)


conexao = None
cursor = None


try:

    # ========================================================
    # 6. CONEXÃO
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
            "Não foi possível conectar."
        )


    print(
        "✅ Conexão realizada."
    )


    cursor = conexao.cursor(
        dictionary=True
    )


    # ========================================================
    # 7. VALIDAR MODELO
    # ========================================================

    cursor.execute(
        """
        SELECT
            id_modelo,
            nome,
            versao
        FROM modelos_ml
        WHERE id_modelo = %s
        """,
        (
            int(ID_MODELO_FINAL),
        )
    )


    modelo = cursor.fetchone()


    if modelo is None:

        raise RuntimeError(
            f"Modelo {ID_MODELO_FINAL} "
            f"não encontrado."
        )


    print(
        "\nModelo confirmado:"
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
    # 8. VERIFICAR PREDIÇÕES EXISTENTES
    # ========================================================

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


    existentes = int(
        cursor.fetchone()[
            "total"
        ]
    )


    print(
        f"\nPredições já cadastradas: "
        f"{existentes}"
    )


    if existentes not in [
        0,
        QUANTIDADE_ESPERADA
    ]:

        raise RuntimeError(
            "Existe uma quantidade parcial inesperada "
            "de predições no banco."
        )


    # ========================================================
    # 9. CASO JÁ ESTEJA COMPLETO
    # ========================================================

    if existentes == QUANTIDADE_ESPERADA:

        print(
            "\n⚠️ As predições já foram cadastradas."
        )


        print(
            "Nenhum INSERT será executado."
        )


    # ========================================================
    # 10. CASO AINDA NÃO TENHA SIDO INSERIDO
    # ========================================================

    else:

        # ====================================================
        # IDs DAS NOITES
        # ====================================================
        #
        # CORREÇÃO IMPORTANTE:
        #
        # Cada valor vira explicitamente int do Python.
        #
        # ====================================================

        ids_noites = sorted(

            int(valor)

            for valor in df[
                "id_noite"
            ].unique()
        )


        print(
            f"\nNoites do teste encontradas no CSV: "
            f"{len(ids_noites)}"
        )


        print(
            "IDs:"
        )


        print(
            ids_noites
        )


        # ====================================================
        # 11. BUSCAR ÉPOCAS
        # ====================================================

        placeholders = ",".join(
            ["%s"] * len(ids_noites)
        )


        sql_epocas = f"""
        SELECT
            id_epoca,
            id_noite,
            numero_epoca,
            id_estagio_referencia
        FROM epocas
        WHERE id_noite IN ({placeholders})
        """


        print(
            "\nBuscando épocas correspondentes..."
        )


        cursor.execute(

            sql_epocas,

            tuple(
                int(valor)
                for valor in ids_noites
            )
        )


        linhas_epocas = cursor.fetchall()


        print(
            f"Épocas retornadas pelo banco: "
            f"{len(linhas_epocas)}"
        )


        # ====================================================
        # 12. MAPEAR
        # ====================================================

        mapa_epocas = {}


        for linha in linhas_epocas:


            id_noite = int(
                linha[
                    "id_noite"
                ]
            )


            numero_epoca = int(
                linha[
                    "numero_epoca"
                ]
            )


            id_epoca = int(
                linha[
                    "id_epoca"
                ]
            )


            referencia = int(
                linha[
                    "id_estagio_referencia"
                ]
            )


            chave = (
                id_noite,
                numero_epoca
            )


            if chave in mapa_epocas:

                raise RuntimeError(
                    f"Época duplicada no banco: "
                    f"{chave}"
                )


            mapa_epocas[
                chave
            ] = (
                id_epoca,
                referencia
            )


        # ====================================================
        # 13. PREPARAR INSERT
        # ====================================================

        print(
            "\nAssociando predições às épocas..."
        )


        registros = []

        nao_encontradas = []

        labels_divergentes = []


        for linha in df.itertuples(
            index=False
        ):


            # ------------------------------------------------
            # CONVERTER TUDO PARA TIPOS NATIVOS
            # ------------------------------------------------

            id_noite = int(
                linha.id_noite
            )


            numero_epoca = int(
                linha.epoca
            )


            label = int(
                linha.label
            )


            predicao = int(
                linha.predicao
            )


            confianca = float(
                linha.confianca
            )


            chave = (
                id_noite,
                numero_epoca
            )


            dados_banco = mapa_epocas.get(
                chave
            )


            if dados_banco is None:

                nao_encontradas.append(
                    chave
                )

                continue


            id_epoca = int(
                dados_banco[0]
            )


            referencia_banco = int(
                dados_banco[1]
            )


            # ------------------------------------------------
            # Conferir ground truth
            # ------------------------------------------------

            if label != referencia_banco:

                labels_divergentes.append(
                    (
                        chave,
                        label,
                        referencia_banco
                    )
                )


            registros.append(
                (
                    int(id_epoca),
                    int(ID_MODELO_FINAL),
                    int(predicao),
                    float(confianca)
                )
            )


        # ====================================================
        # 14. RESULTADO DO MAPEAMENTO
        # ====================================================

        print(
            f"Predições associadas: "
            f"{len(registros)}"
        )


        print(
            f"Não encontradas: "
            f"{len(nao_encontradas)}"
        )


        print(
            f"Labels divergentes: "
            f"{len(labels_divergentes)}"
        )


        if nao_encontradas:

            print(
                "\nPrimeiras épocas não encontradas:"
            )


            for chave in nao_encontradas[
                :10
            ]:

                print(
                    f"  {chave}"
                )


            raise RuntimeError(
                "Nem todas as épocas foram encontradas."
            )


        if labels_divergentes:

            print(
                "\nPrimeiras divergências:"
            )


            for item in labels_divergentes[
                :10
            ]:

                print(
                    f"  {item}"
                )


            raise RuntimeError(
                "Existem labels divergentes entre "
                "CSV e banco."
            )


        if len(registros) != QUANTIDADE_ESPERADA:

            raise RuntimeError(
                f"Esperávamos preparar "
                f"{QUANTIDADE_ESPERADA} registros, "
                f"mas foram preparados "
                f"{len(registros)}."
            )


        print(
            "✅ Todas as épocas foram associadas."
        )


        # ====================================================
        # 15. INSERT
        # ====================================================

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


        print(
            "\nInserindo no banco..."
        )


        total_inserido = 0


        for inicio in range(
            0,
            len(registros),
            TAMANHO_LOTE
        ):


            lote = registros[
                inicio:
                inicio + TAMANHO_LOTE
            ]


            cursor.executemany(
                sql_insert,
                lote
            )


            total_inserido += len(
                lote
            )


            print(
                f"  {total_inserido}/"
                f"{QUANTIDADE_ESPERADA}"
            )


        # ====================================================
        # COMMIT SOMENTE AGORA
        # ====================================================

        conexao.commit()


        print(
            "✅ Commit realizado."
        )


    # ========================================================
    # 16. VALIDAR TOTAL FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 95
    )

    print(
        "VALIDAÇÃO FINAL"
    )

    print(
        "=" * 95
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


    total_final = int(
        cursor.fetchone()[
            "total"
        ]
    )


    print(
        f"\nPredições cadastradas: "
        f"{total_final}"
    )


    if total_final != QUANTIDADE_ESPERADA:

        raise RuntimeError(
            f"Quantidade final incorreta: "
            f"{total_final}"
        )


    # ========================================================
    # 17. DISTRIBUIÇÃO
    # ========================================================

    cursor.execute(
        """
        SELECT
            e.id_estagio,
            e.codigo,
            COUNT(*) AS quantidade
        FROM predicoes p
        INNER JOIN estagios_sono e
            ON e.id_estagio = p.id_estagio_previsto
        WHERE p.id_modelo = %s
        GROUP BY
            e.id_estagio,
            e.codigo
        ORDER BY
            e.id_estagio
        """,
        (
            int(ID_MODELO_FINAL),
        )
    )


    distribuicao = cursor.fetchall()


    print(
        "\nDistribuição no banco:"
    )


    soma = 0


    for linha in distribuicao:


        quantidade = int(
            linha[
                "quantidade"
            ]
        )


        soma += quantidade


        percentual = (
            quantidade
            / total_final
            * 100
        )


        print(
            f"  {linha['id_estagio']} "
            f"{linha['codigo']:<5} "
            f"{quantidade:5d} "
            f"({percentual:6.2f}%)"
        )


    if soma != total_final:

        raise RuntimeError(
            "A soma por estágio está incorreta."
        )


    # ========================================================
    # 18. CONFIANÇA
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
            int(ID_MODELO_FINAL),
        )
    )


    confianca = cursor.fetchone()


    print(
        "\nConfiança experimental:"
    )


    print(
        f"  Mínima: "
        f"{float(confianca['minimo']):.4f}"
    )


    print(
        f"  Média: "
        f"{float(confianca['media']):.4f}"
    )


    print(
        f"  Máxima: "
        f"{float(confianca['maximo']):.4f}"
    )


    # ========================================================
    # 19. PARTICIPANTES
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

        GROUP BY
            pa.codigo

        ORDER BY
            pa.codigo
        """,
        (
            int(ID_MODELO_FINAL),
        )
    )


    participantes = (
        cursor.fetchall()
    )


    print(
        "\nParticipantes:"
    )


    for linha in participantes:

        print(
            f"  {linha['participante']}: "
            f"{linha['quantidade']} épocas"
        )


    encontrados = {
        linha[
            "participante"
        ]
        for linha in participantes
    }


    esperados = {
        "Bidslab16",
        "Bidslab22",
        "Bidslab32"
    }


    if encontrados != esperados:

        raise RuntimeError(
            "Os participantes encontrados "
            "não correspondem ao teste final."
        )


    # ========================================================
    # 20. FINAL
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
        f"\nModelo ID: "
        f"{ID_MODELO_FINAL}"
    )


    print(
        f"Predições: "
        f"{total_final}"
    )


    print(
        f"Participantes: "
        f"{len(encontrados)}"
    )


    print(
        "\n✅ CSV validado."
    )

    print(
        "✅ Épocas associadas."
    )

    print(
        "✅ Labels conferidos."
    )

    print(
        "✅ Predições inseridas."
    )

    print(
        "✅ Confianças inseridas."
    )

    print(
        "✅ Quantidade final conferida."
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