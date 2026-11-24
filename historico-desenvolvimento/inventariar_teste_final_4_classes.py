from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_URL = (
    "https://physionet.org/files/"
    "bidsleep-dataset/1.0.0"
)

ARQUIVO_DIVISAO = Path(
    "data/processed/divisao_final_4_classes.csv"
)

ARQUIVO_MANIFESTO = Path(
    "data/processed/manifesto_teste_final_4_classes.csv"
)

ARQUIVOS_OBRIGATORIOS = [
    "hr.csv",
    "motion.csv",
    "labels.mat"
]

MAX_NOITES = 7

TIMEOUT = 30


# ============================================================
# FUNÇÕES
# ============================================================

def tamanho_legivel(bytes_total):

    unidades = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    tamanho = float(
        bytes_total
    )

    for unidade in unidades:

        if tamanho < 1024:

            return (
                f"{tamanho:.2f} "
                f"{unidade}"
            )

        tamanho /= 1024

    return (
        f"{tamanho:.2f} PB"
    )


def verificar_arquivo(url):

    # --------------------------------------------------------
    # Primeiro tenta HEAD para não baixar o arquivo.
    # --------------------------------------------------------

    requisicao = Request(
        url,
        method="HEAD",
        headers={
            "User-Agent":
                "Projeto-Sono-UNIVAP/1.0"
        }
    )

    try:

        with urlopen(
            requisicao,
            timeout=TIMEOUT
        ) as resposta:

            status = resposta.status

            tamanho = resposta.headers.get(
                "Content-Length"
            )

            if tamanho is not None:

                tamanho = int(
                    tamanho
                )

            return (
                status == 200,
                tamanho,
                status
            )

    except HTTPError as erro:

        # ----------------------------------------------------
        # Alguns servidores podem não gostar de HEAD.
        # Se for esse o caso, tenta uma requisição mínima.
        # ----------------------------------------------------

        if erro.code in [
            403,
            405
        ]:

            requisicao_range = Request(
                url,
                method="GET",
                headers={
                    "User-Agent":
                        "Projeto-Sono-UNIVAP/1.0",

                    "Range":
                        "bytes=0-0"
                }
            )

            try:

                with urlopen(
                    requisicao_range,
                    timeout=TIMEOUT
                ) as resposta:

                    tamanho = None

                    content_range = (
                        resposta.headers.get(
                            "Content-Range"
                        )
                    )

                    if (
                        content_range is not None
                        and "/"
                        in content_range
                    ):

                        total = (
                            content_range
                            .split("/")[-1]
                        )

                        if total.isdigit():

                            tamanho = int(
                                total
                            )

                    return (
                        resposta.status
                        in [
                            200,
                            206
                        ],
                        tamanho,
                        resposta.status
                    )

            except Exception:

                return (
                    False,
                    None,
                    erro.code
                )

        return (
            False,
            None,
            erro.code
        )

    except URLError:

        return (
            False,
            None,
            None
        )

    except Exception:

        return (
            False,
            None,
            None
        )


# ============================================================
# INÍCIO
# ============================================================

print("=" * 100)
print("INVENTÁRIO REMOTO DO TESTE FINAL - 4 CLASSES")
print("=" * 100)


# ============================================================
# CARREGAR DIVISÃO CONGELADA
# ============================================================

if not ARQUIVO_DIVISAO.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado:\n"
        f"{ARQUIVO_DIVISAO}"
    )


divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


teste = divisao[
    divisao[
        "grupo"
    ] == "teste_final_4_classes"
].copy()


participantes = sorted(
    teste[
        "participante"
    ]
    .astype(str)
    .tolist()
)


if len(participantes) != 3:

    raise RuntimeError(
        "Esperávamos exatamente "
        "3 participantes de teste."
    )


print(
    "\nParticipantes congelados:"
)


for participante in participantes:

    print(
        f"  🔒 {participante}"
    )


# ============================================================
# INVENTÁRIO
# ============================================================

registros = []

total_bytes = 0

total_noites = 0


for participante in participantes:

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


    noites_encontradas = 0


    for numero_noite in range(
        1,
        MAX_NOITES + 1
    ):

        print(
            f"\nVerificando noite "
            f"{numero_noite}..."
        )


        arquivos_noite = []

        noite_completa = True

        tamanho_noite = 0


        for nome_arquivo in (
            ARQUIVOS_OBRIGATORIOS
        ):

            url = (
                f"{BASE_URL}/"
                f"{participante}/"
                f"{numero_noite}/"
                f"{nome_arquivo}"
            )


            existe, tamanho, status = (
                verificar_arquivo(
                    url
                )
            )


            if not existe:

                noite_completa = False


            if tamanho is not None:

                tamanho_noite += (
                    tamanho
                )


            arquivos_noite.append(
                {
                    "nome":
                        nome_arquivo,

                    "url":
                        url,

                    "existe":
                        existe,

                    "tamanho_bytes":
                        tamanho,

                    "status_http":
                        status
                }
            )


            simbolo = (
                "✅"
                if existe
                else "❌"
            )


            tamanho_texto = (

                tamanho_legivel(
                    tamanho
                )

                if tamanho is not None

                else "desconhecido"
            )


            print(
                f"  {simbolo} "
                f"{nome_arquivo:<12} "
                f"{tamanho_texto}"
            )


        # ----------------------------------------------------
        # Só consideramos uma noite válida se os 3 arquivos
        # estiverem disponíveis.
        # ----------------------------------------------------

        if noite_completa:

            noites_encontradas += 1

            total_noites += 1

            total_bytes += (
                tamanho_noite
            )


            print(
                f"  ✅ Noite completa "
                f"({tamanho_legivel(tamanho_noite)})"
            )

        else:

            print(
                "  ⛔ Noite não disponível "
                "ou incompleta."
            )


        for arquivo in arquivos_noite:

            registros.append(
                {
                    "participante":
                        participante,

                    "numero_noite":
                        numero_noite,

                    "arquivo":
                        arquivo[
                            "nome"
                        ],

                    "existe":
                        arquivo[
                            "existe"
                        ],

                    "tamanho_bytes":
                        arquivo[
                            "tamanho_bytes"
                        ],

                    "status_http":
                        arquivo[
                            "status_http"
                        ],

                    "url":
                        arquivo[
                            "url"
                        ],

                    "noite_completa":
                        noite_completa
                }
            )


    print(
        f"\nNoites completas de "
        f"{participante}: "
        f"{noites_encontradas}"
    )


# ============================================================
# SALVAR MANIFESTO
# ============================================================

manifesto = pd.DataFrame(
    registros
)


manifesto.to_csv(
    ARQUIVO_MANIFESTO,
    index=False
)


# ============================================================
# RESUMO
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "RESUMO DO TESTE FINAL"
)

print(
    "=" * 100
)


for participante in participantes:

    noites = manifesto[
        (
            manifesto[
                "participante"
            ] == participante
        )
        &
        (
            manifesto[
                "noite_completa"
            ] == True
        )
    ][
        "numero_noite"
    ].unique()


    noites = sorted(
        int(x)
        for x in noites
    )


    print(
        f"\n{participante}"
    )

    print(
        f"  Noites disponíveis: "
        f"{len(noites)}"
    )

    print(
        f"  Números: "
        f"{noites}"
    )


print(
    "\n"
    + "-" * 100
)


print(
    f"Total de participantes: "
    f"{len(participantes)}"
)


print(
    f"Total de noites completas: "
    f"{total_noites}"
)


if total_bytes > 0:

    print(
        f"Tamanho aproximado do download: "
        f"{tamanho_legivel(total_bytes)}"
    )

else:

    print(
        "Tamanho aproximado do download: "
        "não informado pelo servidor"
    )


print(
    "\nManifesto salvo em:"
)

print(
    ARQUIVO_MANIFESTO
)


print(
    "\n"
    + "=" * 100
)

print(
    "INVENTÁRIO CONCLUÍDO"
)

print(
    "=" * 100
)


print(
    "\n✅ Nenhum arquivo de dados foi baixado."
)

print(
    "✅ Nenhum resultado do teste final foi analisado."
)

print(
    "✅ Apenas verificamos a disponibilidade dos arquivos."
)