from pathlib import Path

import urllib.request
import ssl
import certifi
import time


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

ARQUIVOS = [
    "hr.csv",
    "motion.csv",
    "labels.mat"
]

BASE_URL = (
    "https://physionet.org/files/"
    "bidsleep-dataset/1.0.0"
)

MAX_TENTATIVAS = 5

TEMPO_ENTRE_TENTATIVAS = 5


# ============================================================
# CONFIGURAÇÃO SSL
# ============================================================

context = ssl.create_default_context(
    cafile=certifi.where()
)


# ============================================================
# FUNÇÃO PARA BAIXAR UM ARQUIVO
# ============================================================

def baixar_arquivo(url, destino):

    # --------------------------------------------------------
    # 1. VERIFICAR SE O ARQUIVO COMPLETO JÁ EXISTE
    # --------------------------------------------------------

    if destino.exists():

        tamanho_mb = (
            destino.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"Já existe: {destino.name} "
            f"({tamanho_mb:.2f} MB)"
        )

        return True


    # --------------------------------------------------------
    # 2. CAMINHO DO DOWNLOAD PARCIAL
    # --------------------------------------------------------

    arquivo_temporario = Path(
        str(destino) + ".part"
    )


    # --------------------------------------------------------
    # 3. TENTATIVAS
    # --------------------------------------------------------

    for tentativa in range(
        1,
        MAX_TENTATIVAS + 1
    ):

        print(
            f"Tentativa "
            f"{tentativa}/{MAX_TENTATIVAS}"
        )

        try:

            # ------------------------------------------------
            # DESCOBRIR QUANTO JÁ FOI BAIXADO
            # ------------------------------------------------

            tamanho_existente = 0

            if arquivo_temporario.exists():

                tamanho_existente = (
                    arquivo_temporario.stat().st_size
                )

                print(
                    f"Download parcial encontrado: "
                    f"{tamanho_existente / (1024 * 1024):.2f} MB"
                )


            # ------------------------------------------------
            # MONTAR CABEÇALHO HTTP RANGE
            # ------------------------------------------------

            headers = {}

            if tamanho_existente > 0:

                headers["Range"] = (
                    f"bytes={tamanho_existente}-"
                )

                print(
                    "Tentando continuar "
                    "do ponto onde parou..."
                )


            # ------------------------------------------------
            # CRIAR REQUISIÇÃO
            # ------------------------------------------------

            requisicao = urllib.request.Request(
                url,
                headers=headers
            )


            # ------------------------------------------------
            # ABRIR CONEXÃO
            # ------------------------------------------------

            with urllib.request.urlopen(
                requisicao,
                context=context,
                timeout=120
            ) as resposta:

                status = resposta.status


                # ------------------------------------------------
                # SERVIDOR ACEITOU O RANGE
                # ------------------------------------------------

                if (
                    tamanho_existente > 0
                    and status == 206
                ):

                    modo = "ab"

                    print(
                        "Servidor aceitou continuação."
                    )


                # ------------------------------------------------
                # SERVIDOR NÃO ACEITOU RANGE
                # ------------------------------------------------

                else:

                    if tamanho_existente > 0:

                        print(
                            "Servidor não aceitou "
                            "a continuação."
                        )

                        print(
                            "O download será reiniciado "
                            "desse arquivo."
                        )

                    tamanho_existente = 0

                    modo = "wb"


                # ------------------------------------------------
                # DESCOBRIR TAMANHO TOTAL
                # ------------------------------------------------

                content_range = (
                    resposta.headers.get(
                        "Content-Range"
                    )
                )


                if content_range:

                    # Exemplo:
                    #
                    # bytes 1000000-5000000/5000001

                    tamanho_total = int(
                        content_range
                        .split("/")[-1]
                    )

                else:

                    content_length = (
                        resposta.headers.get(
                            "Content-Length"
                        )
                    )

                    if content_length:

                        tamanho_total = (
                            tamanho_existente
                            + int(content_length)
                        )

                    else:

                        tamanho_total = None


                if tamanho_total:

                    print(
                        f"Tamanho total: "
                        f"{tamanho_total / (1024 * 1024):.2f} MB"
                    )


                # ------------------------------------------------
                # BAIXAR EM BLOCOS DE 1 MB
                # ------------------------------------------------

                total = tamanho_existente


                with open(
                    arquivo_temporario,
                    modo
                ) as arquivo:

                    while True:

                        bloco = resposta.read(
                            1024 * 1024
                        )

                        if not bloco:
                            break


                        arquivo.write(
                            bloco
                        )

                        total += len(
                            bloco
                        )


                        if tamanho_total:

                            porcentagem = (
                                total
                                / tamanho_total
                            ) * 100


                            print(
                                f"\rProgresso: "
                                f"{porcentagem:.1f}%",
                                end=""
                            )


                # ------------------------------------------------
                # VALIDAR SE TERMINOU MESMO
                # ------------------------------------------------

                if (
                    tamanho_total
                    and total < tamanho_total
                ):

                    raise IOError(
                        "A conexão terminou antes "
                        "do arquivo ser concluído."
                    )


            # ------------------------------------------------
            # TRANSFORMAR .PART EM ARQUIVO FINAL
            # ------------------------------------------------

            arquivo_temporario.replace(
                destino
            )


            tamanho_final_mb = (
                destino.stat().st_size
                / (1024 * 1024)
            )


            print(
                f"\nDownload concluído: "
                f"{tamanho_final_mb:.2f} MB"
            )

            return True


        # ====================================================
        # USUÁRIO APERTOU CTRL + C
        # ====================================================

        except KeyboardInterrupt:

            print(
                "\n\nDownload interrompido "
                "pelo usuário."
            )


            if arquivo_temporario.exists():

                tamanho_salvo = (
                    arquivo_temporario.stat().st_size
                    / (1024 * 1024)
                )

                print(
                    f"Download parcial preservado: "
                    f"{tamanho_salvo:.2f} MB"
                )

                print(
                    "Na próxima execução, "
                    "o programa tentará continuar."
                )


            raise


        # ====================================================
        # ERRO DE REDE
        # ====================================================

        except Exception as erro:

            print(
                f"\nErro na tentativa "
                f"{tentativa}:"
            )

            print(
                erro
            )


            if arquivo_temporario.exists():

                tamanho_salvo = (
                    arquivo_temporario.stat().st_size
                    / (1024 * 1024)
                )

                print(
                    f"Progresso preservado: "
                    f"{tamanho_salvo:.2f} MB"
                )


            if tentativa < MAX_TENTATIVAS:

                print(
                    f"Aguardando "
                    f"{TEMPO_ENTRE_TENTATIVAS} segundos..."
                )

                time.sleep(
                    TEMPO_ENTRE_TENTATIVAS
                )


    print(
        f"\nFALHA DEFINITIVA:"
    )

    print(
        url
    )

    return False


# ============================================================
# PROCESSAMENTO DA AMOSTRA COMPLETA
# ============================================================

def baixar_amostra():

    total_noites = (
        len(PARTICIPANTES)
        * len(NOITES)
    )

    contador_noite = 0

    falhas = []


    print(
        "=" * 70
    )

    print(
        "DOWNLOAD DA AMOSTRA DO PROJETO"
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
    # PARTICIPANTES
    # ========================================================

    for participante in PARTICIPANTES:

        # ====================================================
        # NOITES
        # ====================================================

        for noite in NOITES:

            contador_noite += 1


            print(
                "\n"
                + "=" * 70
            )

            print(
                f"NOITE "
                f"{contador_noite}/{total_noites}"
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
                "=" * 70
            )


            # ------------------------------------------------
            # CRIAR PASTA
            # ------------------------------------------------

            pasta = Path(
                "data"
            ) / "raw" / "participantes" / participante / (
                f"noite_{noite}"
            )


            pasta.mkdir(
                parents=True,
                exist_ok=True
            )


            noite_ok = True


            # ------------------------------------------------
            # BAIXAR OS 3 ARQUIVOS
            # ------------------------------------------------

            for nome_arquivo in ARQUIVOS:

                print(
                    f"\nArquivo: "
                    f"{nome_arquivo}"
                )


                url = (
                    f"{BASE_URL}/"
                    f"{participante}/"
                    f"{noite}/"
                    f"{nome_arquivo}"
                )


                destino = (
                    pasta
                    / nome_arquivo
                )


                sucesso = baixar_arquivo(
                    url,
                    destino
                )


                if not sucesso:

                    noite_ok = False


            # ------------------------------------------------
            # REGISTRAR NOITES COM PROBLEMA
            # ------------------------------------------------

            if not noite_ok:

                falhas.append(
                    (
                        participante,
                        noite
                    )
                )


    # ========================================================
    # RESUMO
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DOWNLOAD DA AMOSTRA FINALIZADO"
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
        f"Total previsto: "
        f"{total_noites}"
    )


    if len(falhas) == 0:

        print(
            "\nTodos os downloads "
            "foram concluídos com sucesso."
        )

    else:

        print(
            f"\nNoites com problemas: "
            f"{len(falhas)}"
        )


        for participante, noite in falhas:

            print(
                f"- {participante} "
                f"/ noite {noite}"
            )


# ============================================================
# INICIAR O PROGRAMA
# ============================================================

if __name__ == "__main__":

    baixar_amostra()