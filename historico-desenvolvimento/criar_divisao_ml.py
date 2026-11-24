from pathlib import Path

import pandas as pd


# ============================================================
# DIVISÃO FIXA DOS PARTICIPANTES
# ============================================================

TREINO = [
    "Bidslab02",
    "Bidslab09",
    "Bidslab11",
    "Bidslab13",
    "Bidslab14",
    "Bidslab17",
    "Bidslab19",
    "Bidslab20",
    "Bidslab30",
    "Bidslab31"
]


VALIDACAO = [
    "Bidslab07",
    "Bidslab18"
]


TESTE = [
    "Bidslab16",
    "Bidslab22",
    "Bidslab32"
]


# ============================================================
# CRIAR REGISTROS
# ============================================================

registros = []


for participante in TREINO:

    registros.append({
        "participante": participante,
        "grupo": "treino"
    })


for participante in VALIDACAO:

    registros.append({
        "participante": participante,
        "grupo": "validacao"
    })


for participante in TESTE:

    registros.append({
        "participante": participante,
        "grupo": "teste"
    })


# ============================================================
# DATAFRAME
# ============================================================

divisao = pd.DataFrame(
    registros
)


# ============================================================
# VALIDAÇÕES
# ============================================================

print("=" * 70)
print("DIVISÃO DOS PARTICIPANTES PARA MACHINE LEARNING")
print("=" * 70)


duplicados = (
    divisao["participante"]
    .duplicated()
    .sum()
)


if duplicados > 0:

    raise ValueError(
        "Existe participante em mais de um grupo."
    )


if len(divisao) != 15:

    raise ValueError(
        f"Foram encontrados {len(divisao)} participantes, "
        "mas eram esperados 15."
    )


print(
    f"\nParticipantes de treino: "
    f"{len(TREINO)}"
)

print(
    f"Participantes de validação: "
    f"{len(VALIDACAO)}"
)

print(
    f"Participantes de teste: "
    f"{len(TESTE)}"
)


# ============================================================
# MOSTRAR DIVISÃO
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TREINO"
)

print(
    "=" * 70
)

for participante in TREINO:
    print(participante)


print(
    "\n"
    + "=" * 70
)

print(
    "VALIDAÇÃO"
)

print(
    "=" * 70
)

for participante in VALIDACAO:
    print(participante)


print(
    "\n"
    + "=" * 70
)

print(
    "TESTE"
)

print(
    "=" * 70
)

for participante in TESTE:
    print(participante)


# ============================================================
# SALVAR
# ============================================================

arquivo_saida = Path(
    "data/processed/divisao_participantes_ml.csv"
)


divisao.to_csv(
    arquivo_saida,
    index=False
)


print(
    "\n"
    + "=" * 70
)

print(
    "DIVISÃO SALVA"
)

print(
    "=" * 70
)


print(
    f"\nArquivo criado:"
)

print(
    arquivo_saida
)


print(
    "\n✅ DIVISÃO DE MACHINE LEARNING DEFINIDA"
)