from pathlib import Path

import gc
import time
import pandas as pd

from processar_noite import processar_noite


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PARTICIPANTES = [
    "Bidslab02",
    "Bidslab07",
    "Bidslab09",
    "Bidslab11",
    "Bidslab13",
    "Bidslab14",
    "Bidslab16",
    "Bidslab17",
    "Bidslab18",
    "Bidslab19",
    "Bidslab20",
    "Bidslab22",
    "Bidslab30",
    "Bidslab31",
    "Bidslab32"
]

NOITES = [
    1,
    2,
    3,
    4,
    5
]


PASTA_PROCESSADOS = Path(
    "data/processed/participantes"
)


# ============================================================
# COLUNAS ESPERADAS
# ============================================================

COLUNAS_ESPERADAS = {
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
}


# ============================================================
# VERIFICAR SE UMA NOITE JÁ FOI PROCESSADA
# ============================================================

def arquivo_processado_valido(
    participante,
    noite
):

    arquivo = (
        PASTA_PROCESSADOS
        / participante
        / f"noite_{noite}.csv"
    )


    # --------------------------------------------------------
    # ARQUIVO NÃO EXISTE
    # --------------------------------------------------------

    if not arquivo.exists():
        return False


    # --------------------------------------------------------
    # ARQUIVO VAZIO
    # --------------------------------------------------------

    if arquivo.stat().st_size == 0:
        return False


    # --------------------------------------------------------
    # VALIDAR ESTRUTURA
    # --------------------------------------------------------

    try:

        # Lemos apenas algumas linhas.
        # Não precisamos carregar o CSV inteiro.
        df = pd.read_csv(
            arquivo,
            nrows=5
        )


        colunas = set(
            df.columns
        )


        if not COLUNAS_ESPERADAS.issubset(
            colunas
        ):
            return False


        return True


    except Exception:

        return False


# ============================================================
# PROCESSAR AMOSTRA COMPLETA
# ============================================================

def processar_amostra():

    total_noites = (
        len(PARTICIPANTES)
        * len(NOITES)
    )


    contador = 0

    processadas_agora = 0
    puladas = 0

    falhas = []

    tempos = []


    print(
        "=" * 70
    )

    print(
        "PROCESSAMENTO DA AMOSTRA COMPLETA"
    )

    print(
        "=" * 70
    )


    print(
        f"\nParticipantes: "
        f"{len(PARTICIPANTES)}"
    )

    print(
        f"Noites por participante: "
        f"{len(NOITES)}"
    )

    print(
        f"Total de noites: "
        f"{total_noites}"
    )


    # ========================================================
    # PERCORRER PARTICIPANTES
    # ========================================================

    for participante in PARTICIPANTES:


        # ====================================================
        # PERCORRER NOITES
        # ====================================================

        for noite in NOITES:

            contador += 1


            print(
                "\n"
                + "#" * 70
            )

            print(
                f"NOITE "
                f"{contador}/{total_noites}"
            )

            print(
                f"Participante: "
                f"{participante}"
            )

            print(
                f"Noite: "
                f"{noite}"
            )

            print(
                "#" * 70
            )


            # =================================================
            # VERIFICAR SE JÁ ESTÁ PROCESSADA
            # =================================================

            if arquivo_processado_valido(
                participante,
                noite
            ):

                arquivo_existente = (
                    PASTA_PROCESSADOS
                    / participante
                    / f"noite_{noite}.csv"
                )


                try:

                    quantidade_epocas = len(
                        pd.read_csv(
                            arquivo_existente,
                            usecols=["epoca"]
                        )
                    )

                except Exception:

                    quantidade_epocas = "?"


                print(
                    "\n✅ Já processada."
                )

                print(
                    f"Épocas salvas: "
                    f"{quantidade_epocas}"
                )

                print(
                    "Pulando esta noite..."
                )


                puladas += 1

                continue


            # =================================================
            # PROCESSAR NOITE
            # =================================================

            inicio = (
                time.perf_counter()
            )


            try:

                dataset = processar_noite(
                    participante,
                    noite
                )


                duracao = (
                    time.perf_counter()
                    - inicio
                )


                tempos.append(
                    duracao
                )


                processadas_agora += 1


                print(
                    f"\nTempo dessa noite: "
                    f"{duracao:.2f} segundos"
                )


                print(
                    f"Épocas válidas: "
                    f"{len(dataset)}"
                )


                # ------------------------------------------------
                # LIBERAR MEMÓRIA
                # ------------------------------------------------

                del dataset

                gc.collect()


            # =================================================
            # CTRL + C
            # =================================================

            except KeyboardInterrupt:

                print(
                    "\n\n⚠️ PROCESSAMENTO "
                    "INTERROMPIDO PELO USUÁRIO"
                )


                print(
                    "As noites que já foram "
                    "processadas continuam salvas."
                )


                print(
                    "Quando executar novamente, "
                    "elas serão puladas."
                )


                return


            # =================================================
            # ERRO DE UMA NOITE
            # =================================================

            except Exception as erro:

                duracao = (
                    time.perf_counter()
                    - inicio
                )


                print(
                    "\n❌ ERRO NO PROCESSAMENTO"
                )


                print(
                    f"Participante: "
                    f"{participante}"
                )


                print(
                    f"Noite: "
                    f"{noite}"
                )


                print(
                    f"Erro: "
                    f"{erro}"
                )


                falhas.append(
                    (
                        participante,
                        noite,
                        str(erro)
                    )
                )


                # Mesmo com erro,
                # continuamos para a próxima noite.
                continue


            # =================================================
            # ESTIMATIVA DE TEMPO
            # =================================================

            if len(tempos) > 0:

                tempo_medio = (
                    sum(tempos)
                    / len(tempos)
                )


                noites_restantes = (
                    total_noites
                    - contador
                )


                segundos_restantes = (
                    tempo_medio
                    * noites_restantes
                )


                minutos_restantes = (
                    segundos_restantes
                    / 60
                )


                print(
                    f"\nTempo médio/noite: "
                    f"{tempo_medio:.2f} s"
                )


                print(
                    f"Estimativa restante: "
                    f"{minutos_restantes:.1f} min"
                )


    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESSAMENTO DA AMOSTRA FINALIZADO"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTotal previsto: "
        f"{total_noites}"
    )


    print(
        f"Processadas nesta execução: "
        f"{processadas_agora}"
    )


    print(
        f"Já existentes / puladas: "
        f"{puladas}"
    )


    print(
        f"Falhas: "
        f"{len(falhas)}"
    )


    # ========================================================
    # MOSTRAR FALHAS
    # ========================================================

    if len(falhas) > 0:

        print(
            "\n⚠️ NOITES COM PROBLEMAS"
        )


        for (
            participante,
            noite,
            erro
        ) in falhas:

            print(
                "\n"
                + "-" * 50
            )

            print(
                f"Participante: "
                f"{participante}"
            )

            print(
                f"Noite: "
                f"{noite}"
            )

            print(
                f"Erro: "
                f"{erro}"
            )


    else:

        print(
            "\n✅ TODAS AS 75 NOITES "
            "ESTÃO PROCESSADAS"
        )


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    processar_amostra()