import re
import ssl
import certifi
import urllib.request


BASE_URL = (
    "https://physionet.org/files/"
    "bidsleep-dataset/1.0.0/"
)

context = ssl.create_default_context(
    cafile=certifi.where()
)


def acessar_pagina(url):

    with urllib.request.urlopen(
        url,
        context=context
    ) as resposta:

        return resposta.read().decode()


# ============================================================
# 1. LISTAR PARTICIPANTES
# ============================================================

print("=" * 70)
print("LISTANDO PARTICIPANTES E NOITES DISPONÍVEIS")
print("=" * 70)

html_principal = acessar_pagina(
    BASE_URL
)

participantes = re.findall(
    r'href="(Bidslab\d+)/"',
    html_principal
)

participantes = sorted(
    set(participantes)
)

print(
    f"\nParticipantes encontrados: "
    f"{len(participantes)}"
)

print()


# ============================================================
# 2. VERIFICAR NOITES DE CADA PARTICIPANTE
# ============================================================

resultado = []

for numero, participante in enumerate(
    participantes,
    start=1
):

    print(
        f"[{numero}/{len(participantes)}] "
        f"Verificando {participante}...",
        end=" "
    )

    url_participante = (
        f"{BASE_URL}{participante}/"
    )

    try:

        html = acessar_pagina(
            url_participante
        )

        noites = re.findall(
            r'href="(\d+)/"',
            html
        )

        noites = sorted(
            set(int(n) for n in noites)
        )

        resultado.append(
            (
                participante,
                noites
            )
        )

        print(
            f"{len(noites)} noites: "
            f"{noites}"
        )

    except Exception as erro:

        print(
            f"ERRO: {erro}"
        )


# ============================================================
# 3. MOSTRAR PARTICIPANTES COM PELO MENOS 5 NOITES
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANTES COM PELO MENOS 5 NOITES")
print("=" * 70)

aptos = []

for participante, noites in resultado:

    if len(noites) >= 5:

        aptos.append(
            (
                participante,
                noites
            )
        )

        print(
            f"{participante}: "
            f"{noites}"
        )


print("\n" + "=" * 70)

print(
    f"Total de participantes com "
    f"5 ou mais noites: {len(aptos)}"
)

print("=" * 70)