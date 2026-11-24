from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CAMINHO DOS DADOS
# ============================================================

PASTA_PARTICIPANTES = Path(
    "data/raw/participantes"
)


# ============================================================
# RESULTADOS
# ============================================================

resultados = []


print("=" * 90)
print("ANÁLISE DOS SINAIS BRUTOS PARA REDE NEURAL")
print("=" * 90)


# ============================================================
# LOCALIZAR TODAS AS NOITES
# ============================================================

arquivos_motion = sorted(
    PASTA_PARTICIPANTES.glob(
        "*/noite_*/motion.csv"
    )
)


arquivos_hr = sorted(
    PASTA_PARTICIPANTES.glob(
        "*/noite_*/hr.csv"
    )
)


print(
    f"\nArquivos motion encontrados: "
    f"{len(arquivos_motion)}"
)


print(
    f"Arquivos HR encontrados: "
    f"{len(arquivos_hr)}"
)


if len(arquivos_motion) == 0:

    raise FileNotFoundError(
        "Nenhum motion.csv encontrado."
    )


if len(arquivos_hr) == 0:

    raise FileNotFoundError(
        "Nenhum hr.csv encontrado."
    )


# ============================================================
# ANALISAR MOVIMENTO
# ============================================================

print(
    "\nAnalisando frequência do acelerômetro..."
)


for numero, arquivo in enumerate(
    arquivos_motion,
    start=1
):

    participante = arquivo.parts[-3]

    noite = arquivo.parts[-2]


    motion = pd.read_csv(
        arquivo
    )


    # --------------------------------------------------------
    # Descobrir nome da coluna de tempo.
    # --------------------------------------------------------

    coluna_tempo = motion.columns[
        0
    ]


    timestamps = pd.to_numeric(
        motion[
            coluna_tempo
        ],
        errors="coerce"
    ).dropna().to_numpy()


    if len(timestamps) < 2:

        continue


    diferencas = np.diff(
        timestamps
    )


    diferencas = diferencas[
        diferencas > 0
    ]


    if len(diferencas) == 0:

        continue


    intervalo_mediano = np.median(
        diferencas
    )


    frequencia_mediana = (
        1.0
        / intervalo_mediano
    )


    resultados.append({

        "participante":
            participante,

        "noite":
            noite,

        "amostras_motion":
            len(motion),

        "intervalo_mediano_seg":
            intervalo_mediano,

        "frequencia_motion_hz":
            frequencia_mediana
    })


    print(
        f"[{numero:02d}/{len(arquivos_motion)}] "
        f"{participante} {noite} "
        f"-> "
        f"{frequencia_mediana:.2f} Hz "
        f"| "
        f"{len(motion):,} amostras"
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    resultados
)


if len(df) == 0:

    raise RuntimeError(
        "Não foi possível calcular "
        "nenhuma frequência."
    )


# ============================================================
# RESUMO
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "RESUMO DO ACELERÔMETRO"
)

print(
    "=" * 90
)


print(
    "\nFrequência de amostragem (Hz):"
)


print(
    df[
        "frequencia_motion_hz"
    ]
    .describe()
    .round(4)
    .to_string()
)


print(
    "\nQuantidade de amostras por noite:"
)


print(
    df[
        "amostras_motion"
    ]
    .describe()
    .round(2)
    .to_string()
)


# ============================================================
# ESTIMATIVA POR ÉPOCA DE 30 SEGUNDOS
# ============================================================

frequencia_mediana_global = (

    df[
        "frequencia_motion_hz"
    ]
    .median()
)


amostras_30s = (

    frequencia_mediana_global
    * 30
)


print(
    "\n"
    + "=" * 90
)

print(
    "ESTIMATIVA PARA UMA ÉPOCA DE 30 SEGUNDOS"
)

print(
    "=" * 90
)


print(
    f"\nFrequência mediana global: "
    f"{frequencia_mediana_global:.2f} Hz"
)


print(
    f"Amostras esperadas em 30 segundos: "
    f"{amostras_30s:.0f}"
)


# ============================================================
# SALVAR
# ============================================================

ARQUIVO_SAIDA = Path(
    "data/processed/analise_sinais_brutos_rede.csv"
)


df.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


print(
    "\nArquivo salvo:"
)


print(
    ARQUIVO_SAIDA
)


print(
    "\n✅ ANÁLISE CONCLUÍDA"
)