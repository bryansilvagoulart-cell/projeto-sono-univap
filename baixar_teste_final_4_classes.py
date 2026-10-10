from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import shutil
import time

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_MANIFESTO = Path(
    "data/processed/manifesto_teste_final_4_classes.csv"
)

ARQUIVO_PLANO = Path(
    "data/processed/plano_download_teste_final_4_classes.csv"
)

PASTA_DESTINO = Path(
    "data/raw/participantes"
)

PARTICIPANTES_TESTE = [
    "Bidslab01",
    "Bidslab35",
    "Bidslab47"
]

# O desenvolvimento utilizou 5 noites por participante.
# Para manter o teste equilibrado, usaremos exatamente
# 5 noites de cada participante.
NOITES_OFICIAIS = [
    1,
    2,
    3,
    4,
    5
]

ARQUIVOS_OBRIGATORIOS = [
    "hr.csv",
    "motion.csv",
    "labels.mat"
]

CHUNK_SIZE = 1024 * 1024

TIMEOUT = 60

MAX_TENTATIVAS = 5


# ============================================================
# FUNÇÕES AUXILIARES
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


def barra_progresso(
    atual,
    total,
    largura=35
):

    if total <= 0:

        return (
            "[???????????????????????????????????]"
        )

    proporcao = min(
        atual / total,
        1.0
    )

    preenchido = int(
        largura * proporcao
    )

    vazio = (
        largura
        - preenchido
    )

    barra = (
        "█" * preenchido
        + "-" * vazio
    )

    percentual = (
        proporcao
        * 100
    )

    return (
        f"[{barra}] "
        f"{percentual:6.2f}%"
    )


def baixar_arquivo(
    url,
    destino,
    tamanho_esperado
):

    destino.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporario = destino.with_suffix(
        destino.suffix + ".part"
    )

    # --------------------------------------------------------
    # Arquivo já existe completo
    # --------------------------------------------------------

    if destino.exists():

        tamanho_atual = (
            destino.stat().st_size
        )

        if (
            tamanho_esperado is None
            or tamanho_atual
            == tamanho_esperado
        ):

            print(
                f"    ✅ Já existe: "
                f"{destino.name}"
            )

            return True

        else:

            print(
                f"    ⚠️ Arquivo existente "
                f"com tamanho incorreto."
            )

            destino.unlink()


    # --------------------------------------------------------
    # Tentativas de download
    # --------------------------------------------------------

    for tentativa in range(
        1,
        MAX_TENTATIVAS + 1
    ):

        try:

            inicio = 0

            if temporario.exists():

                inicio = (
                    temporario
                    .stat()
                    .st_size
                )


            headers = {
                "User-Agent":
                    "Projeto-Sono-UNIVAP/1.0"
            }


            if inicio > 0:

                headers[
                    "Range"
                ] = (
                    f"bytes={inicio}-"
                )


            requisicao = Request(
                url,
                headers=headers
            )


            with urlopen(
                requisicao,
                timeout=TIMEOUT
            ) as resposta:

                status = (
                    resposta.status
                )


                # ------------------------------------------------
                # Se pedimos continuação e o servidor ignorou Range
                # retornando 200, reiniciamos o download.
                # ------------------------------------------------

                if (
                    inicio > 0
                    and status == 200
                ):

                    print(
                        "\n    ⚠️ Servidor não aceitou "
                        "retomada. Reiniciando arquivo."
                    )

                    inicio = 0

                    if temporario.exists():

                        temporario.unlink()


                modo = (
                    "ab"
                    if inicio > 0
                    else "wb"
                )


                total = (
                    tamanho_esperado
                    if tamanho_esperado
                    is not None
                    else 0
                )


                baixado = inicio


                with open(
                    temporario,
                    modo
                ) as arquivo:

                    while True:

                        bloco = resposta.read(
                            CHUNK_SIZE
                        )

                        if not bloco:

                            break


                        arquivo.write(
                            bloco
                        )


                        baixado += len(
                            bloco
                        )


                        if total > 0:

                            barra = (
                                barra_progresso(
                                    baixado,
                                    total
                                )
                            )

                            texto_tamanho = (
                                f"{tamanho_legivel(baixado)}"
                                f" / "
                                f"{tamanho_legivel(total)}"
                            )

                        else:

                            barra = (
                                "[download em andamento]"
                            )

                            texto_tamanho = (
                                tamanho_legivel(
                                    baixado
                                )
                            )


                        print(
                            f"\r    {barra} "
                            f"{texto_tamanho}",
                            end="",
                            flush=True
                        )


                print()


            # ------------------------------------------------
            # Validar tamanho final
            # ------------------------------------------------

            tamanho_final = (
                temporario
                .stat()
                .st_size
            )


            if (
                tamanho_esperado
                is not None
                and tamanho_final
                != tamanho_esperado
            ):

                raise RuntimeError(
                    f"Tamanho incorreto. "
                    f"Esperado: "
                    f"{tamanho_esperado} bytes | "
                    f"Obtido: "
                    f"{tamanho_final} bytes"
                )


            shutil.move(
                temporario,
                destino
            )


            print(
                f"    ✅ Download concluído: "
                f"{destino.name}"
            )


            return True


        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
            RuntimeError
        ) as erro:

            print(
                f"\n    ⚠️ Tentativa "
                f"{tentativa}/"
                f"{MAX_TENTATIVAS} "
                f"falhou:"
            )

            print(
                f"       {erro}"
            )


            if tentativa < MAX_TENTATIVAS:

                print(
                    "       Aguardando 5 segundos..."
                )

                time.sleep(
                    5
                )


    return False


# ============================================================
# INÍCIO
# ============================================================

print(
    "=" * 100
)

print(
    "DOWNLOAD DO TESTE FINAL CONGELADO - 4 CLASSES"
)

print(
    "=" * 100
)


# ============================================================
# VALIDAR MANIFESTO
# ============================================================

if not ARQUIVO_MANIFESTO.exists():

    raise FileNotFoundError(
        f"Manifesto não encontrado:\n"
        f"{ARQUIVO_MANIFESTO}"
    )


manifesto = pd.read_csv(
    ARQUIVO_MANIFESTO
)


# ============================================================
# MONTAR PLANO OFICIAL
# ============================================================

registros_plano = []


for participante in PARTICIPANTES_TESTE:

    for numero_noite in NOITES_OFICIAIS:

        linhas = manifesto[
            (
                manifesto[
                    "participante"
                ] == participante
            )
            &
            (
                manifesto[
                    "numero_noite"
                ] == numero_noite
            )
            &
            (
                manifesto[
                    "noite_completa"
                ] == True
            )
        ].copy()


        if len(linhas) != 3:

            raise RuntimeError(
                f"Noite incompleta no manifesto:\n"
                f"{participante} "
                f"- noite {numero_noite}"
            )


        arquivos_encontrados = set(
            linhas[
                "arquivo"
            ].tolist()
        )


        if arquivos_encontrados != set(
            ARQUIVOS_OBRIGATORIOS
        ):

            raise RuntimeError(
                f"Arquivos inesperados em "
                f"{participante}, "
                f"noite {numero_noite}"
            )


        for _, linha in linhas.iterrows():

            registros_plano.append(
                {
                    "participante":
                        participante,

                    "numero_noite":
                        numero_noite,

                    "arquivo":
                        linha[
                            "arquivo"
                        ],

                    "url":
                        linha[
                            "url"
                        ],

                    "tamanho_bytes":
                        linha[
                            "tamanho_bytes"
                        ],

                    "usar_teste_final":
                        True
                }
            )


plano = pd.DataFrame(
    registros_plano
)


plano.to_csv(
    ARQUIVO_PLANO,
    index=False
)


# ============================================================
# RESUMO DO PLANO
# ============================================================

print(
    "\nPROTOCOLO CONGELADO:"
)


for participante in PARTICIPANTES_TESTE:

    print(
        f"  🔒 {participante}: "
        f"noites 1 a 5"
    )


print(
    "\nBidslab35 noite 6:"
)

print(
    "  🚫 Não será utilizada no teste oficial."
)


total_bytes = (
    pd.to_numeric(
        plano[
            "tamanho_bytes"
        ],
        errors="coerce"
    )
    .fillna(0)
    .sum()
)


print(
    f"\nParticipantes: "
    f"{len(PARTICIPANTES_TESTE)}"
)

print(
    f"Noites: "
    f"{len(PARTICIPANTES_TESTE) * len(NOITES_OFICIAIS)}"
)

print(
    f"Arquivos: "
    f"{len(plano)}"
)

print(
    f"Tamanho aproximado: "
    f"{tamanho_legivel(total_bytes)}"
)


print(
    "\nPlano salvo em:"
)

print(
    ARQUIVO_PLANO
)


# ============================================================
# DOWNLOAD
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "INICIANDO DOWNLOAD"
)

print(
    "=" * 100
)


arquivos_ok = 0

arquivos_erro = []


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


    for numero_noite in NOITES_OFICIAIS:

        print(
            f"\n  NOITE {numero_noite}"
        )


        linhas = plano[
            (
                plano[
                    "participante"
                ] == participante
            )
            &
            (
                plano[
                    "numero_noite"
                ] == numero_noite
            )
        ]


        for _, linha in linhas.iterrows():

            nome_arquivo = (
                linha[
                    "arquivo"
                ]
            )

            url = (
                linha[
                    "url"
                ]
            )


            tamanho = (
                linha[
                    "tamanho_bytes"
                ]
            )


            if pd.isna(
                tamanho
            ):

                tamanho = None

            else:

                tamanho = int(
                    tamanho
                )


            destino = (
                PASTA_DESTINO
                / participante
                / f"noite_{numero_noite}"
                / nome_arquivo
            )


            print(
                f"\n    {nome_arquivo}"
            )


            sucesso = baixar_arquivo(
                url=url,
                destino=destino,
                tamanho_esperado=tamanho
            )


            if sucesso:

                arquivos_ok += 1

            else:

                arquivos_erro.append(
                    str(
                        destino
                    )
                )


# ============================================================
# RESULTADO FINAL
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "RESULTADO DO DOWNLOAD"
)

print(
    "=" * 100
)


print(
    f"\nArquivos esperados: "
    f"{len(plano)}"
)

print(
    f"Arquivos concluídos: "
    f"{arquivos_ok}"
)

print(
    f"Arquivos com erro: "
    f"{len(arquivos_erro)}"
)


if arquivos_erro:

    print(
        "\nArquivos que precisam "
        "ser retomados:"
    )

    for caminho in arquivos_erro:

        print(
            f"  ❌ {caminho}"
        )


    print(
        "\nExecute novamente este mesmo "
        "script para continuar."
    )


else:

    print(
        "\n✅ Todos os arquivos do teste "
        "final foram baixados."
    )


    print(
        "\nEstrutura criada:"
    )

    for participante in PARTICIPANTES_TESTE:

        print(
            f"\n  {participante}/"
        )

        for numero_noite in NOITES_OFICIAIS:

            print(
                f"    noite_{numero_noite}/"
            )

            print(
                "      hr.csv"
            )

            print(
                "      motion.csv"
            )

            print(
                "      labels.mat"
            )


    print(
        "\n🔒 O conjunto de teste continua "
        "sem ter sido utilizado para "
        "ajustar o modelo."
    )


print(
    "\n"
    + "=" * 100
)

print(
    "PROCESSO FINALIZADO"
)

print(
    "=" * 100
)