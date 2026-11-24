from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

FREQUENCIA_ALVO = 10.0
DURACAO_EPOCA_SEG = 30.0

AMOSTRAS_POR_EPOCA = int(
    FREQUENCIA_ALVO * DURACAO_EPOCA_SEG
)

CANAIS = [
    "x",
    "y",
    "z",
    "magnitude"
]


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_DATASET = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
)

ARQUIVO_DIVISAO = Path(
    "data/processed/divisao_participantes_ml.csv"
)

PASTA_RAW = Path(
    "data/raw/participantes"
)

ARQUIVO_SINAIS = Path(
    "data/processed/motion_cnn_dev_10hz.npy"
)

ARQUIVO_METADATA = Path(
    "data/processed/motion_cnn_dev_metadata.csv"
)


# ============================================================
# FUNÇÃO - NOME DA PASTA DA NOITE
# ============================================================

def nome_pasta_noite(valor):

    texto = str(valor)

    if texto.startswith("noite_"):
        return texto

    numero = int(float(valor))

    return f"noite_{numero}"


# ============================================================
# FUNÇÃO - ENCONTRAR COLUNAS
# ============================================================

def encontrar_colunas_motion(arquivo):

    cabecalho = pd.read_csv(
        arquivo,
        nrows=0
    )

    colunas = list(
        cabecalho.columns
    )

    mapa = {
        coluna.lower().strip(): coluna
        for coluna in colunas
    }

    coluna_timestamp = None

    for candidato in [
        "timestamp",
        "time",
        "tempo"
    ]:

        if candidato in mapa:

            coluna_timestamp = mapa[
                candidato
            ]

            break


    if coluna_timestamp is None:

        coluna_timestamp = colunas[0]


    for eixo in [
        "x",
        "y",
        "z"
    ]:

        if eixo not in mapa:

            raise ValueError(
                f"Coluna '{eixo}' não encontrada em:\n"
                f"{arquivo}"
            )


    return (
        coluna_timestamp,
        mapa["x"],
        mapa["y"],
        mapa["z"]
    )


# ============================================================
# FUNÇÃO - NORMALIZAR TIMESTAMP DO MOTION
# ============================================================

def normalizar_timestamp_motion(timestamps):

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    validos = timestamps[
        np.isfinite(timestamps)
    ]


    if len(validos) == 0:

        raise ValueError(
            "Nenhum timestamp válido foi encontrado."
        )


    referencia = np.median(
        np.abs(validos)
    )


    # Microssegundos Unix
    if referencia > 1e14:

        timestamps = (
            timestamps
            / 1_000_000.0
        )


    # Milissegundos Unix
    elif referencia > 1e11:

        timestamps = (
            timestamps
            / 1_000.0
        )


    # Caso contrário:
    # segundos Unix.


    return timestamps


# ============================================================
# INÍCIO
# ============================================================

print("=" * 90)
print("DATASET DE MOVIMENTO PARA CNN - 10 Hz")
print("=" * 90)


print(
    f"\nFrequência alvo: "
    f"{FREQUENCIA_ALVO:.1f} Hz"
)

print(
    f"Duração de cada época: "
    f"{DURACAO_EPOCA_SEG:.0f} segundos"
)

print(
    f"Amostras por canal: "
    f"{AMOSTRAS_POR_EPOCA}"
)

print(
    f"Canais: "
    f"{', '.join(CANAIS)}"
)


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

print(
    "\nCarregando dataset principal..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


if len(dataset) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


# ============================================================
# 2. CARREGAR DIVISÃO
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


participantes_teste = (

    divisao[
        divisao["grupo"] == "teste"
    ]

    ["participante"]

    .tolist()
)


print(
    "\nParticipantes de TESTE que NÃO serão processados:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# 3. SOMENTE DESENVOLVIMENTO
# ============================================================

dataset_dev = (

    dataset[
        ~dataset[
            "participante"
        ].isin(
            participantes_teste
        )
    ]

    .copy()

    .reset_index(
        drop=True
    )
)


if len(dataset_dev) != 48104:

    raise ValueError(
        f"Esperávamos 48104 épocas de desenvolvimento, "
        f"mas encontramos {len(dataset_dev)}."
    )


print(
    f"\nÉpocas de desenvolvimento: "
    f"{len(dataset_dev)}"
)

print(
    f"Participantes: "
    f"{dataset_dev['participante'].nunique()}"
)

print(
    "Noites: "
    f"{dataset_dev[['participante', 'noite']].drop_duplicates().shape[0]}"
)


# ============================================================
# 4. VALIDAR COLUNAS
# ============================================================

COLUNAS_NECESSARIAS = [
    "participante",
    "noite",
    "epoca",
    "inicio",
    "label"
]


faltando = [
    coluna
    for coluna in COLUNAS_NECESSARIAS
    if coluna not in dataset_dev.columns
]


if faltando:

    raise ValueError(
        "Colunas ausentes:\n"
        + "\n".join(faltando)
    )


# ============================================================
# 5. CONVERSÃO CORRETA DOS HORÁRIOS
# ============================================================
#
# IMPORTANTE:
#
# Não usamos:
#
# .astype("int64") / 1e9
#
# porque no pandas atual o datetime está em microssegundos.
#
# Timestamp.timestamp() independe dessa unidade interna.
#
# ============================================================

print(
    "\nConvertendo início das épocas para Unix UTC..."
)


inicio_datetime = pd.to_datetime(
    dataset_dev["inicio"],
    utc=True,
    errors="raise"
)


dataset_dev["_inicio_unix"] = np.array(
    [
        timestamp.timestamp()
        for timestamp in inicio_datetime
    ],
    dtype=np.float64
)


print(
    "✅ Conversão realizada corretamente."
)


# ============================================================
# 6. VERIFICAÇÃO DE SANIDADE
# ============================================================

primeiro_timestamp = dataset_dev[
    "_inicio_unix"
].iloc[0]


print(
    f"\nPrimeiro timestamp convertido: "
    f"{primeiro_timestamp:.3f}"
)


if not (
    1_000_000_000
    < primeiro_timestamp
    < 2_000_000_000
):

    raise ValueError(
        "O timestamp convertido está fora "
        "da faixa esperada de Unix segundos."
    )


# ============================================================
# 7. CRIAR ARQUIVO NPY
# ============================================================

print(
    "\nCriando arquivo NPY..."
)


ARQUIVO_SINAIS.parent.mkdir(
    parents=True,
    exist_ok=True
)


shape_saida = (
    len(dataset_dev),
    len(CANAIS),
    AMOSTRAS_POR_EPOCA
)


motion_memmap = np.lib.format.open_memmap(
    ARQUIVO_SINAIS,
    mode="w+",
    dtype=np.float32,
    shape=shape_saida
)


tamanho_mb = (
    np.prod(shape_saida)
    * 4
    / 1024
    / 1024
)


print(
    f"Shape: {shape_saida}"
)

print(
    f"Tamanho aproximado: "
    f"{tamanho_mb:.1f} MB"
)


# ============================================================
# 8. METADADOS
# ============================================================

dataset_dev[
    "indice_motion_cnn"
] = np.arange(
    len(dataset_dev)
)


dataset_dev[
    "motion_raw_samples_usadas"
] = 0


# ============================================================
# 9. AGRUPAR POR NOITE
# ============================================================

grupos_noites = list(

    dataset_dev.groupby(
        [
            "participante",
            "noite"
        ],
        sort=False
    )
)


print(
    f"\nNoites que serão processadas: "
    f"{len(grupos_noites)}"
)


# ============================================================
# 10. PROCESSAR NOITE POR NOITE
# ============================================================

for numero_noite, (
    chave,
    grupo
) in enumerate(
    grupos_noites,
    start=1
):


    participante = chave[0]
    noite = chave[1]


    pasta_noite = nome_pasta_noite(
        noite
    )


    arquivo_motion = (

        PASTA_RAW
        / participante
        / pasta_noite
        / "motion.csv"
    )


    if not arquivo_motion.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n"
            f"{arquivo_motion}"
        )


    print(
        f"\n[{numero_noite:02d}/"
        f"{len(grupos_noites)}] "
        f"{participante} "
        f"{pasta_noite}"
    )


    # ========================================================
    # COLUNAS
    # ========================================================

    (
        coluna_timestamp,
        coluna_x,
        coluna_y,
        coluna_z

    ) = encontrar_colunas_motion(
        arquivo_motion
    )


    # ========================================================
    # LER MOTION
    # ========================================================

    motion = pd.read_csv(
        arquivo_motion,
        usecols=[
            coluna_timestamp,
            coluna_x,
            coluna_y,
            coluna_z
        ]
    )


    timestamps = pd.to_numeric(
        motion[coluna_timestamp],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )


    x = pd.to_numeric(
        motion[coluna_x],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )


    y = pd.to_numeric(
        motion[coluna_y],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )


    z = pd.to_numeric(
        motion[coluna_z],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )


    # ========================================================
    # REMOVER INVÁLIDOS
    # ========================================================

    validos = (
        np.isfinite(timestamps)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )


    timestamps = timestamps[
        validos
    ]

    x = x[
        validos
    ]

    y = y[
        validos
    ]

    z = z[
        validos
    ]


    timestamps = normalizar_timestamp_motion(
        timestamps
    )


    # ========================================================
    # ORDENAR
    # ========================================================

    ordem = np.argsort(
        timestamps
    )


    timestamps = timestamps[
        ordem
    ]

    x = x[
        ordem
    ]

    y = y[
        ordem
    ]

    z = z[
        ordem
    ]


    # ========================================================
    # REMOVER TIMESTAMPS DUPLICADOS
    # ========================================================

    timestamps, indices_unicos = np.unique(
        timestamps,
        return_index=True
    )


    x = x[
        indices_unicos
    ]

    y = y[
        indices_unicos
    ]

    z = z[
        indices_unicos
    ]


    print(
        f"  Amostras válidas: "
        f"{len(timestamps):,}"
    )


    # ========================================================
    # TESTE DE SINCRONIZAÇÃO DA NOITE
    # ========================================================

    primeiro_inicio_noite = float(
        grupo[
            "_inicio_unix"
        ].min()
    )


    ultimo_inicio_noite = float(
        grupo[
            "_inicio_unix"
        ].max()
    )


    motion_inicio = float(
        timestamps[0]
    )


    motion_fim = float(
        timestamps[-1]
    )


    existe_sobreposicao = (
        primeiro_inicio_noite
        <= motion_fim
        and
        ultimo_inicio_noite + 30
        >= motion_inicio
    )


    if not existe_sobreposicao:

        raise RuntimeError(
            "\nSem sobreposição temporal.\n"
            f"Participante: {participante}\n"
            f"Noite: {pasta_noite}\n"
            f"Motion: {motion_inicio} até {motion_fim}\n"
            f"Épocas: {primeiro_inicio_noite} "
            f"até {ultimo_inicio_noite + 30}"
        )


    # ========================================================
    # PROCESSAR ÉPOCAS
    # ========================================================

    for indice_dataframe, linha in grupo.iterrows():


        inicio = float(
            linha["_inicio_unix"]
        )


        fim = (
            inicio
            + DURACAO_EPOCA_SEG
        )


        # ----------------------------------------------------
        # Amostras dentro da época
        # ----------------------------------------------------

        indice_inicio_real = np.searchsorted(
            timestamps,
            inicio,
            side="left"
        )


        indice_fim_real = np.searchsorted(
            timestamps,
            fim,
            side="left"
        )


        quantidade_real = (
            indice_fim_real
            - indice_inicio_real
        )


        # ----------------------------------------------------
        # Margem pequena para interpolar bordas
        # ----------------------------------------------------

        indice_inicio = np.searchsorted(
            timestamps,
            inicio - 1.0,
            side="left"
        )


        indice_fim = np.searchsorted(
            timestamps,
            fim + 1.0,
            side="right"
        )


        ts_epoca = timestamps[
            indice_inicio:indice_fim
        ]

        x_epoca = x[
            indice_inicio:indice_fim
        ]

        y_epoca = y[
            indice_inicio:indice_fim
        ]

        z_epoca = z[
            indice_inicio:indice_fim
        ]


        if len(ts_epoca) < 2:

            raise RuntimeError(
                "\nPoucas amostras de movimento.\n"
                f"Participante: {participante}\n"
                f"Noite: {pasta_noite}\n"
                f"Época: {linha['epoca']}\n"
                f"Amostras: {len(ts_epoca)}"
            )


        # ====================================================
        # GRADE FIXA DE 10 Hz
        # ====================================================

        alvo = (

            inicio

            + np.arange(
                AMOSTRAS_POR_EPOCA,
                dtype=np.float64
            )

            / FREQUENCIA_ALVO
        )


        # ====================================================
        # INTERPOLAÇÃO
        # ====================================================

        x_interp = np.interp(
            alvo,
            ts_epoca,
            x_epoca
        )


        y_interp = np.interp(
            alvo,
            ts_epoca,
            y_epoca
        )


        z_interp = np.interp(
            alvo,
            ts_epoca,
            z_epoca
        )


        magnitude = np.sqrt(
            x_interp ** 2
            + y_interp ** 2
            + z_interp ** 2
        )


        # ====================================================
        # SALVAR NO MEMMAP
        # ====================================================

        posicao = int(
            linha[
                "indice_motion_cnn"
            ]
        )


        motion_memmap[
            posicao,
            0,
            :
        ] = x_interp.astype(
            np.float32
        )


        motion_memmap[
            posicao,
            1,
            :
        ] = y_interp.astype(
            np.float32
        )


        motion_memmap[
            posicao,
            2,
            :
        ] = z_interp.astype(
            np.float32
        )


        motion_memmap[
            posicao,
            3,
            :
        ] = magnitude.astype(
            np.float32
        )


        dataset_dev.loc[
            indice_dataframe,
            "motion_raw_samples_usadas"
        ] = quantidade_real


    motion_memmap.flush()


    print(
        f"  Épocas processadas: "
        f"{len(grupo)}"
    )


    # Liberar memória da noite atual.

    del motion
    del timestamps
    del x
    del y
    del z


# ============================================================
# 11. VALIDAR NPY
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "VALIDANDO DATASET"
)

print(
    "=" * 90
)


motion_memmap.flush()


quantidade_nan = int(
    np.isnan(
        motion_memmap
    ).sum()
)


quantidade_inf = int(
    np.isinf(
        motion_memmap
    ).sum()
)


print(
    f"\nShape: "
    f"{motion_memmap.shape}"
)

print(
    f"NaN: "
    f"{quantidade_nan}"
)

print(
    f"Inf: "
    f"{quantidade_inf}"
)


if quantidade_nan != 0:

    raise ValueError(
        "Foram encontrados NaN."
    )


if quantidade_inf != 0:

    raise ValueError(
        "Foram encontrados infinitos."
    )


# ============================================================
# 12. DISTRIBUIÇÃO DAS AMOSTRAS BRUTAS
# ============================================================

print(
    "\nAmostras brutas por época:"
)


print(

    dataset_dev[
        "motion_raw_samples_usadas"
    ]

    .describe()

    .round(2)

    .to_string()
)


quantidade_zero = int(

    (
        dataset_dev[
            "motion_raw_samples_usadas"
        ] == 0
    ).sum()
)


print(
    f"\nÉpocas com 0 amostras brutas: "
    f"{quantidade_zero}"
)


if quantidade_zero != 0:

    raise ValueError(
        "Existem épocas sem amostras "
        "originais de movimento."
    )


# ============================================================
# 13. SALVAR METADADOS
# ============================================================

COLUNAS_METADATA = [
    "indice_motion_cnn",
    "participante",
    "noite",
    "epoca",
    "inicio",
    "label",
    "motion_raw_samples_usadas"
]


for coluna in [
    "id_noite",
    "segmento"
]:

    if coluna in dataset_dev.columns:

        COLUNAS_METADATA.append(
            coluna
        )


dataset_dev[
    COLUNAS_METADATA
].to_csv(
    ARQUIVO_METADATA,
    index=False
)


# ============================================================
# 14. FINAL
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "DATASET CNN CRIADO COM SUCESSO"
)

print(
    "=" * 90
)


print(
    "\nArquivo de sinais:"
)

print(
    ARQUIVO_SINAIS
)


print(
    "\nArquivo de metadados:"
)

print(
    ARQUIVO_METADATA
)


print(
    "\nShape final:"
)

print(
    motion_memmap.shape
)


print(
    "\nInterpretação:"
)

print(
    f"  {len(dataset_dev)} épocas"
)

print(
    f"  {len(CANAIS)} canais"
)

print(
    f"  {AMOSTRAS_POR_EPOCA} pontos por canal"
)

print(
    f"  {FREQUENCIA_ALVO:.1f} Hz"
)


print(
    "\n✅ Participantes finais de teste "
    "não foram incluídos."
)