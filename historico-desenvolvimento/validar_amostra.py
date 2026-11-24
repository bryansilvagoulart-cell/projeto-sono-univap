from pathlib import Path

from scipy.io import loadmat


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

NOITES = [1, 2, 3, 4, 5]

BASE = Path(
    "data/raw/participantes"
)


# ============================================================
# CONTADORES
# ============================================================

total_esperado = (
    len(PARTICIPANTES)
    * len(NOITES)
)

noites_validas = 0
noites_invalidas = []

arquivos_part = []


# ============================================================
# FUNÇÃO PARA VALIDAR UMA NOITE
# ============================================================

def validar_noite(participante, noite):

    pasta = (
        BASE
        / participante
        / f"noite_{noite}"
    )

    problemas = []

    # --------------------------------------------------------
    # 1. VERIFICAR PASTA
    # --------------------------------------------------------

    if not pasta.exists():

        problemas.append(
            "Pasta não existe"
        )

        return problemas


    # --------------------------------------------------------
    # 2. VERIFICAR ARQUIVOS
    # --------------------------------------------------------

    hr = pasta / "hr.csv"
    motion = pasta / "motion.csv"
    labels = pasta / "labels.mat"


    for arquivo in [
        hr,
        motion,
        labels
    ]:

        if not arquivo.exists():

            problemas.append(
                f"{arquivo.name} não existe"
            )

        elif arquivo.stat().st_size == 0:

            problemas.append(
                f"{arquivo.name} está vazio"
            )


    # Se algum arquivo nem existe,
    # não adianta tentar validar conteúdo.
    if problemas:
        return problemas


    # --------------------------------------------------------
    # 3. VALIDAR HR.CSV
    # --------------------------------------------------------

    try:

        with open(
            hr,
            "r",
            encoding="utf-8"
        ) as arquivo:

            primeira_linha = (
                arquivo.readline()
                .strip()
            )

        partes = (
            primeira_linha.split(",")
        )

        if len(partes) != 2:

            problemas.append(
                "hr.csv com formato inesperado"
            )

        else:

            # O HR não possui cabeçalho.
            # Esperamos dois números:
            # timestamp e frequência cardíaca.
            float(partes[0])
            float(partes[1])

    except Exception as erro:

        problemas.append(
            f"Erro ao validar hr.csv: {erro}"
        )


    # --------------------------------------------------------
    # 4. VALIDAR MOTION.CSV
    # --------------------------------------------------------

    try:

        with open(
            motion,
            "r",
            encoding="utf-8"
        ) as arquivo:

            cabecalho = (
                arquivo.readline()
                .strip()
            )

        colunas = [
            coluna.strip()
            for coluna
            in cabecalho.split(",")
        ]

        esperadas = [
            "Timestamp",
            "x",
            "y",
            "z"
        ]

        if colunas != esperadas:

            problemas.append(
                "motion.csv com cabeçalho inesperado: "
                f"{colunas}"
            )

    except Exception as erro:

        problemas.append(
            f"Erro ao validar motion.csv: {erro}"
        )


    # --------------------------------------------------------
    # 5. VALIDAR LABELS.MAT
    # --------------------------------------------------------

    try:

        dados = loadmat(
            labels
        )

        if "expert_label" not in dados:

            problemas.append(
                "expert_label não encontrado"
            )

        if "recStart" not in dados:

            problemas.append(
                "recStart não encontrado"
            )

        if "expert_label" in dados:

            quantidade = (
                dados["expert_label"]
                .flatten()
                .size
            )

            if quantidade == 0:

                problemas.append(
                    "expert_label está vazio"
                )

    except Exception as erro:

        problemas.append(
            f"Erro ao validar labels.mat: {erro}"
        )


    return problemas


# ============================================================
# PROCURAR .PART
# ============================================================

for arquivo in BASE.rglob("*.part"):

    arquivos_part.append(
        arquivo
    )


# ============================================================
# VALIDAR TODAS AS 75 NOITES
# ============================================================

print("=" * 70)
print("VALIDAÇÃO DA AMOSTRA")
print("=" * 70)

print(
    f"\nParticipantes: "
    f"{len(PARTICIPANTES)}"
)

print(
    f"Noites por participante: "
    f"{len(NOITES)}"
)

print(
    f"Total esperado: "
    f"{total_esperado}"
)

print()


contador = 0


for participante in PARTICIPANTES:

    for noite in NOITES:

        contador += 1

        problemas = validar_noite(
            participante,
            noite
        )


        if len(problemas) == 0:

            noites_validas += 1

            print(
                f"[{contador:02d}/{total_esperado}] "
                f"{participante} / noite {noite} "
                f"-> OK"
            )

        else:

            noites_invalidas.append(
                (
                    participante,
                    noite,
                    problemas
                )
            )

            print(
                f"[{contador:02d}/{total_esperado}] "
                f"{participante} / noite {noite} "
                f"-> PROBLEMA"
            )

            for problema in problemas:

                print(
                    f"    - {problema}"
                )


# ============================================================
# RESUMO FINAL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RESULTADO DA VALIDAÇÃO"
)

print(
    "=" * 70
)

print(
    f"\nNoites esperadas: "
    f"{total_esperado}"
)

print(
    f"Noites válidas: "
    f"{noites_validas}"
)

print(
    f"Noites com problemas: "
    f"{len(noites_invalidas)}"
)


# ------------------------------------------------------------
# ARQUIVOS .PART
# ------------------------------------------------------------

print(
    f"\nArquivos .part encontrados: "
    f"{len(arquivos_part)}"
)

for arquivo in arquivos_part:

    print(
        f"- {arquivo}"
    )


# ------------------------------------------------------------
# RESULTADO
# ------------------------------------------------------------

if (
    noites_validas == total_esperado
    and len(arquivos_part) == 0
):

    print(
        "\n✅ AMOSTRA VALIDADA COM SUCESSO"
    )

    print(
        "As 75 noites estão prontas "
        "para processamento."
    )

else:

    print(
        "\n⚠️ A AMOSTRA PRECISA DE CORREÇÕES"
    )

    print(
        "Não inicie o processamento "
        "completo ainda."
    )