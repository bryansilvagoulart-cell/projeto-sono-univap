from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import time

import pandas as pd

from processar_noite import processar_noite


# ============================================================
# CONFIGURAÇÃO OFICIAL DO TESTE FINAL
# ============================================================

PARTICIPANTES_TESTE = [
    "Bidslab01",
    "Bidslab35",
    "Bidslab47"
]

NOITES_OFICIAIS = [
    1,
    2,
    3,
    4,
    5
]

PASTA_RAW = Path(
    "data/raw/participantes"
)

PASTA_PROCESSED = Path(
    "data/processed/participantes"
)

ARQUIVO_RESUMO = Path(
    "data/processed/resumo_processamento_base_teste_final.csv"
)


# ============================================================
# INÍCIO
# ============================================================

print("=" * 100)
print("PROCESSAMENTO BASE DO TESTE FINAL - 4 CLASSES")
print("=" * 100)

print(
    "\nEste processo NÃO treina o modelo "
    "e NÃO calcula métricas."
)

print(
    "Ele apenas aplica o mesmo processamento "
    "base usado anteriormente."
)


# ============================================================
# VALIDAR ARQUIVOS DE ENTRADA
# ============================================================

print(
    "\nVerificando arquivos..."
)

arquivos_faltando = []


for participante in PARTICIPANTES_TESTE:

    for noite in NOITES_OFICIAIS:

        pasta_noite = (
            PASTA_RAW
            / participante
            / f"noite_{noite}"
        )

        for nome_arquivo in [
            "hr.csv",
            "motion.csv",
            "labels.mat"
        ]:

            arquivo = (
                pasta_noite
                / nome_arquivo
            )

            if not arquivo.exists():

                arquivos_faltando.append(
                    str(arquivo)
                )


if arquivos_faltando:

    print(
        "\n❌ Existem arquivos faltando:"
    )

    for arquivo in arquivos_faltando:

        print(
            f"  {arquivo}"
        )

    raise FileNotFoundError(
        "O teste final está incompleto."
    )


print(
    "✅ Todos os 45 arquivos foram encontrados."
)


# ============================================================
# PROCESSAR AS 15 NOITES
# ============================================================

resultados = []

contador = 0

total_noites = (
    len(PARTICIPANTES_TESTE)
    * len(NOITES_OFICIAIS)
)

inicio_total = time.time()


for participante in PARTICIPANTES_TESTE:

    for noite in NOITES_OFICIAIS:

        contador += 1

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"NOITE {contador}/{total_noites}"
        )

        print(
            f"Participante: {participante}"
        )

        print(
            f"Noite: {noite}"
        )

        print(
            "=" * 100
        )


        inicio_noite = time.time()


        # ====================================================
        # IMPORTANTE
        # ====================================================
        #
        # processar_noite.py imprime também a distribuição
        # dos labels.
        #
        # Como este é nosso teste final congelado,
        # vamos ocultar essa saída para evitar ficar
        # inspecionando o conjunto antes da avaliação.
        #
        # O processamento realizado é o mesmo.
        # ====================================================

        saida_oculta = StringIO()


        try:

            with redirect_stdout(
                saida_oculta
            ):

                dataset = processar_noite(
                    participante,
                    noite
                )


            tempo_noite = (
                time.time()
                - inicio_noite
            )


            quantidade_epocas = len(
                dataset
            )


            arquivo_saida = (
                PASTA_PROCESSED
                / participante
                / f"noite_{noite}.csv"
            )


            if not arquivo_saida.exists():

                raise FileNotFoundError(
                    f"O processamento terminou, "
                    f"mas o arquivo não foi criado:\n"
                    f"{arquivo_saida}"
                )


            # =================================================
            # VALIDAR ESTRUTURA
            # =================================================

            colunas_esperadas = [
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
                "motion_samples"
            ]


            if list(
                dataset.columns
            ) != colunas_esperadas:

                raise RuntimeError(
                    f"Estrutura inesperada em "
                    f"{participante}, noite {noite}."
                )


            print(
                f"✅ Processamento concluído."
            )

            print(
                f"Épocas base criadas: "
                f"{quantidade_epocas}"
            )

            print(
                f"Colunas: "
                f"{len(dataset.columns)}"
            )

            print(
                f"Tempo: "
                f"{tempo_noite:.1f} s"
            )


            resultados.append(
                {
                    "participante":
                        participante,

                    "noite":
                        noite,

                    "epocas_base":
                        quantidade_epocas,

                    "colunas":
                        len(dataset.columns),

                    "tempo_segundos":
                        round(
                            tempo_noite,
                            2
                        ),

                    "arquivo":
                        str(
                            arquivo_saida
                        ),

                    "status":
                        "OK"
                }
            )


        except Exception as erro:

            print(
                "\n❌ ERRO NO PROCESSAMENTO"
            )

            print(
                f"{type(erro).__name__}: "
                f"{erro}"
            )


            # Mostrar a saída interna somente em caso
            # de erro, para conseguirmos diagnosticar.

            texto_interno = (
                saida_oculta
                .getvalue()
            )


            if texto_interno.strip():

                print(
                    "\nDetalhes internos:"
                )

                print(
                    texto_interno
                )


            resultados.append(
                {
                    "participante":
                        participante,

                    "noite":
                        noite,

                    "epocas_base":
                        None,

                    "colunas":
                        None,

                    "tempo_segundos":
                        round(
                            time.time()
                            - inicio_noite,
                            2
                        ),

                    "arquivo":
                        None,

                    "status":
                        "ERRO"
                }
            )


            raise


# ============================================================
# SALVAR RESUMO
# ============================================================

df_resultados = pd.DataFrame(
    resultados
)


df_resultados.to_csv(
    ARQUIVO_RESUMO,
    index=False
)


# ============================================================
# VALIDAÇÃO FINAL
# ============================================================

erros = df_resultados[
    df_resultados[
        "status"
    ] != "OK"
]


if len(erros) > 0:

    raise RuntimeError(
        "Existem noites com erro "
        "no processamento."
    )


if len(df_resultados) != 15:

    raise RuntimeError(
        "Esperávamos exatamente "
        "15 noites processadas."
    )


arquivos_saida = []


for participante in PARTICIPANTES_TESTE:

    for noite in NOITES_OFICIAIS:

        arquivo = (
            PASTA_PROCESSED
            / participante
            / f"noite_{noite}.csv"
        )

        if not arquivo.exists():

            raise FileNotFoundError(
                arquivo
            )

        arquivos_saida.append(
            arquivo
        )


if len(arquivos_saida) != 15:

    raise RuntimeError(
        "Quantidade incorreta "
        "de arquivos processados."
    )


# ============================================================
# RESUMO
# ============================================================

tempo_total = (
    time.time()
    - inicio_total
)


print(
    "\n"
    + "=" * 100
)

print(
    "PROCESSAMENTO BASE CONCLUÍDO"
)

print(
    "=" * 100
)


print(
    f"\nParticipantes processados: "
    f"{len(PARTICIPANTES_TESTE)}"
)

print(
    f"Noites processadas: "
    f"{len(df_resultados)}"
)

print(
    f"Arquivos CSV criados: "
    f"{len(arquivos_saida)}"
)

print(
    f"Total de épocas base: "
    f"{int(df_resultados['epocas_base'].sum())}"
)

print(
    f"Tempo total: "
    f"{tempo_total:.1f} s"
)


print(
    "\nÉpocas por participante:"
)


resumo_participante = (
    df_resultados
    .groupby(
        "participante"
    )[
        "epocas_base"
    ]
    .sum()
)


for participante, quantidade in (
    resumo_participante.items()
):

    print(
        f"  {participante}: "
        f"{int(quantidade)}"
    )


print(
    "\nArquivo de controle:"
)

print(
    ARQUIVO_RESUMO
)


print(
    "\n✅ Nenhuma métrica do modelo foi calculada."
)

print(
    "✅ Nenhum hiperparâmetro foi alterado."
)

print(
    "✅ Os participantes continuam sendo "
    "um teste final independente."
)


print(
    "\nPróxima etapa:"
)

print(
    "Aplicar exatamente o mesmo controle "
    "de qualidade usado no desenvolvimento."
)