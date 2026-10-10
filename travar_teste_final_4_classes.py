from pathlib import Path
import random

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEED = 42

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_4_classes.csv"
)

ARQUIVO_SAIDA = Path(
    "data/processed/divisao_final_4_classes.csv"
)


# ============================================================
# PARTICIPANTES DISPONÍVEIS NO BIDSLEEP
# ============================================================

PARTICIPANTES_BIDSLEEP = [
    "Bidslab00",
    "Bidslab01",
    "Bidslab02",
    "Bidslab06",
    "Bidslab07",
    "Bidslab08",
    "Bidslab09",
    "Bidslab10",
    "Bidslab11",
    "Bidslab13",
    "Bidslab14",
    "Bidslab15",
    "Bidslab16",
    "Bidslab17",
    "Bidslab18",
    "Bidslab19",
    "Bidslab20",
    "Bidslab22",
    "Bidslab30",
    "Bidslab31",
    "Bidslab32",
    "Bidslab34",
    "Bidslab35",
    "Bidslab36",
    "Bidslab38",
    "Bidslab39",
    "Bidslab40",
    "Bidslab41",
    "Bidslab42",
    "Bidslab43",
    "Bidslab44",
    "Bidslab45",
    "Bidslab47",
    "Bidslab49",
    "Bidslab50",
    "Bidslab51",
    "Bidslab52",
    "Bidslab53",
    "Bidslab55",
    "Bidslab56",
    "Bidslab60",
    "Bidslab62",
    "Bidslab63",
    "Bidslab64",
    "Bidslab65",
    "Bidslab66",
    "Bidslab68"
]


# ============================================================
# INÍCIO
# ============================================================

print("=" * 95)
print("TRAVAMENTO DO NOVO TESTE FINAL - MODELO 4 CLASSES")
print("=" * 95)


# ============================================================
# VALIDAR DATASET
# ============================================================

if not ARQUIVO_DATASET.exists():

    raise FileNotFoundError(
        f"Arquivo não encontrado:\n"
        f"{ARQUIVO_DATASET}"
    )


# ============================================================
# CARREGAR DATASET ATUAL
# ============================================================

print(
    "\nCarregando dataset de desenvolvimento..."
)


df = pd.read_csv(
    ARQUIVO_DATASET
)


participantes_desenvolvimento = sorted(

    df[
        "participante"
    ]
    .astype(str)
    .unique()
    .tolist()
)


print(
    f"Participantes encontrados no desenvolvimento: "
    f"{len(participantes_desenvolvimento)}"
)


if len(participantes_desenvolvimento) != 15:

    raise ValueError(
        "Esperávamos exatamente 15 participantes "
        "no conjunto de desenvolvimento."
    )


# ============================================================
# MOSTRAR PARTICIPANTES JÁ USADOS
# ============================================================

print(
    "\nPARTICIPANTES JÁ UTILIZADOS:"
)


for participante in participantes_desenvolvimento:

    print(
        f"  - {participante}"
    )


# ============================================================
# VALIDAR NOMES
# ============================================================

nao_reconhecidos = [

    participante

    for participante in participantes_desenvolvimento

    if participante not in PARTICIPANTES_BIDSLEEP
]


if nao_reconhecidos:

    raise RuntimeError(
        "Foram encontrados participantes "
        "não reconhecidos:\n"
        + "\n".join(
            nao_reconhecidos
        )
    )


# ============================================================
# IDENTIFICAR PARTICIPANTES NUNCA USADOS
# ============================================================

participantes_disponiveis = [

    participante

    for participante in PARTICIPANTES_BIDSLEEP

    if participante not in participantes_desenvolvimento
]


print(
    f"\nParticipantes ainda nunca utilizados: "
    f"{len(participantes_disponiveis)}"
)


if len(participantes_disponiveis) != 32:

    raise RuntimeError(
        "Quantidade inesperada de participantes disponíveis."
    )


# ============================================================
# SELEÇÃO REPRODUTÍVEL
# ============================================================

gerador = random.Random(
    SEED
)


participantes_teste = gerador.sample(
    participantes_disponiveis,
    3
)


participantes_teste = sorted(
    participantes_teste
)


# ============================================================
# EXIBIR TESTE FINAL
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "NOVO TESTE FINAL CONGELADO"
)

print(
    "=" * 95
)


print(
    f"\nSeed utilizada: "
    f"{SEED}"
)


for participante in participantes_teste:

    print(
        f"  🔒 {participante}"
    )


# ============================================================
# GARANTIR QUE NÃO EXISTE SOBREPOSIÇÃO
# ============================================================

sobreposicao = set(
    participantes_desenvolvimento
).intersection(
    participantes_teste
)


if sobreposicao:

    raise RuntimeError(
        "ERRO: existe participante do desenvolvimento "
        "dentro do teste final."
    )


print(
    "\n✅ Nenhuma sobreposição encontrada."
)


# ============================================================
# CRIAR ARQUIVO DE DIVISÃO
# ============================================================

registros = []


for participante in participantes_desenvolvimento:

    registros.append(
        {
            "participante":
                participante,

            "grupo":
                "desenvolvimento",

            "seed":
                SEED
        }
    )


for participante in participantes_teste:

    registros.append(
        {
            "participante":
                participante,

            "grupo":
                "teste_final_4_classes",

            "seed":
                SEED
        }
    )


df_divisao = pd.DataFrame(
    registros
)


df_divisao.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


# ============================================================
# VALIDAR O ARQUIVO SALVO
# ============================================================

df_validacao = pd.read_csv(
    ARQUIVO_SAIDA
)


teste_salvo = sorted(

    df_validacao[
        df_validacao[
            "grupo"
        ] == "teste_final_4_classes"
    ][
        "participante"
    ]
    .tolist()
)


if teste_salvo != participantes_teste:

    raise RuntimeError(
        "Erro ao salvar a divisão final."
    )


# ============================================================
# MOSTRAR CONFIGURAÇÃO CONGELADA
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "CONFIGURAÇÃO DO MODELO CONGELADA"
)

print(
    "=" * 95
)


print(
    "\nClasses:"
)

print(
    "  0 -> Wake"
)

print(
    "  1 -> Light (N1 + N2)"
)

print(
    "  2 -> Deep (N3)"
)

print(
    "  3 -> REM"
)


print(
    "\nModelo:"
)

print(
    "  Random Forest = 0.75"
)

print(
    "  BiGRU         = 0.25"
)

print(
    "  Viterbi       = 0.80"
)

print(
    "  Janela BiGRU  = 21 épocas"
)

print(
    "  Treinamento   = 8 épocas"
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 95
)

print(
    "DIVISÃO FINAL SALVA COM SUCESSO"
)

print(
    "=" * 95
)


print(
    "\nArquivo:"
)

print(
    ARQUIVO_SAIDA
)


print(
    "\nDESENVOLVIMENTO:"
)

print(
    "15 participantes"
)


print(
    "\nTESTE FINAL:"
)


for participante in participantes_teste:

    print(
        f"  {participante}"
    )


print(
    "\n⚠️ IMPORTANTE:"
)

print(
    "A partir deste momento esses participantes "
    "não devem ser usados para ajustar pesos, "
    "features ou hiperparâmetros."
)

print(
    "Eles serão usados somente no teste final."
)