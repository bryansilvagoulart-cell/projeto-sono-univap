from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

FREQUENCIA_ALVO = 10.0

DURACAO_EPOCA_SEG = 30.0

AMOSTRAS_POR_EPOCA = int(
    FREQUENCIA_ALVO
    * DURACAO_EPOCA_SEG
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
# FUNÇÃO PARA NORMALIZAR O NOME DA NOITE
# ============================================================

def nome_pasta_noite(valor):

    texto = str(
        valor
    )


    if texto.startswith(
        "noite_"
    ):

        return texto


    numero = int(
        float(
            valor
        )
    )


    return f"noite_{numero}"


# ============================================================
# FUNÇÃO PARA ENCONTRAR AS COLUNAS DO MOTION.CSV
# ============================================================

def encontrar_colunas_motion(
    arquivo
):

    cabecalho = pd.read_csv(
        arquivo,
        nrows=0
    )


    colunas = list(
        cabecalho.columns
    )


    mapa = {
        coluna.lower().strip():
            coluna

        for coluna in colunas
    }


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

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

        # Por segurança, usamos a primeira coluna.
        coluna_timestamp = colunas[
            0
        ]


    # --------------------------------------------------------
    # X / Y / Z
    # --------------------------------------------------------

    if "x" not in mapa:

        raise ValueError(
            f"Coluna X não encontrada em {arquivo}"
        )


    if "y" not in mapa:

        raise ValueError(
            f"Coluna Y não encontrada em {arquivo}"
        )


    if "z" not in mapa:

        raise ValueError(
            f"Coluna Z não encontrada em {arquivo}"
        )


    return (
        coluna_timestamp,
        mapa["x"],
        mapa["y"],
        mapa["z"]
    )


# ============================================================
# FUNÇÃO PARA AJUSTAR UNIDADE DO TIMESTAMP
# ============================================================

def normalizar_timestamp(
    timestamps
):

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )


    valores_validos = timestamps[
        np.isfinite(
            timestamps
        )
    ]


    if len(
        valores_validos
    ) == 0:

        raise ValueError(
            "Nenhum timestamp válido encontrado."
        )


    referencia = np.median(
        np.abs(
            valores_validos
        )
    )


    # --------------------------------------------------------
    # Possível microssegundo Unix
    # --------------------------------------------------------

    if referencia > 1e14:

        timestamps = (
            timestamps
            / 1_000_000.0
        )


    # --------------------------------------------------------
    # Possível milissegundo Unix
    # --------------------------------------------------------

    elif referencia > 1e11:

        timestamps = (
            timestamps
            / 1_000.0
        )


    # Caso contrário:
    # já assumimos segundos Unix.


    return timestamps


# ============================================================
# INÍCIO
# ============================================================

print("=" * 90)
print("CRIAÇÃO DO DATASET DE MOVIMENTO PARA CNN")
print("=" * 90)


print(
    f"\nFrequência alvo: "
    f"{FREQUENCIA_ALVO:.1f} Hz"
)


print(
    f"Duração da época: "
    f"{DURACAO_EPOCA_SEG:.0f} s"
)


print(
    f"Amostras finais por época: "
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
    "\nCarregando dataset..."
)


dataset = pd.read_csv(
    ARQUIVO_DATASET
)


if len(
    dataset
) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(dataset)}."
    )


# ============================================================
# 2. CARREGAR DIVISÃO DOS PARTICIPANTES
# ============================================================

divisao = pd.read_csv(
    ARQUIVO_DIVISAO
)


participantes_teste = (

    divisao[
        divisao[
            "grupo"
        ] == "teste"
    ]

    ["participante"]

    .tolist()
)


print(
    "\nParticipantes de teste NÃO serão processados:"
)


for participante in participantes_teste:

    print(
        f"  - {participante}"
    )


# ============================================================
# 3. MANTER SOMENTE DESENVOLVIMENTO
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


if len(
    dataset_dev
) != 48104:

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
    f"Noites: "
    f"{dataset_dev[['participante', 'noite']].drop_duplicates().shape[0]}"
)


# ============================================================
# 4. VALIDAR COLUNAS NECESSÁRIAS
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
        + "\n".join(
            faltando
        )
    )


# ============================================================
# 5. CONVERTER INÍCIO DAS ÉPOCAS PARA UNIX UTC
# ============================================================

print(
    "\nConvertendo horários das épocas..."
)


inicio_datetime = pd.to_datetime(

    dataset_dev[
        "inicio"
    ],

    utc=True,

    errors="raise"
)


dataset_dev[
    "_inicio_unix"
] = (

    inicio_datetime
    .astype(
        "int64"
    )

    .to_numpy()

    / 1_000_000_000.0
)


# ============================================================
# 6. CRIAR ARQUIVO NPY SEM CARREGAR TUDO NA RAM
# ============================================================

print(
    "\nCriando arquivo de saída..."
)


ARQUIVO_SINAIS.parent.mkdir(
    parents=True,
    exist_ok=True
)


shape_saida = (

    len(
        dataset_dev
    ),

    len(
        CANAIS
    ),

    AMOSTRAS_POR_EPOCA
)


motion_memmap = np.lib.format.open_memmap(

    ARQUIVO_SINAIS,

    mode="w+",

    dtype=np.float32,

    shape=shape_saida
)


tamanho_estimado_mb = (

    np.prod(
        shape_saida
    )

    * 4

    / 1024

    / 1024
)


print(
    f"Formato do arquivo: "
    f"{shape_saida}"
)


print(
    f"Tamanho aproximado: "
    f"{tamanho_estimado_mb:.1f} MB"
)


# ============================================================
# 7. METADADOS EXTRAS
# ============================================================

dataset_dev[
    "indice_motion_cnn"
] = np.arange(
    len(
        dataset_dev
    )
)


dataset_dev[
    "motion_raw_samples_usadas"
] = 0


# ============================================================
# 8. PROCESSAR NOITE POR NOITE
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


for numero_noite, (
    chave,
    grupo
) in enumerate(
    grupos_noites,
    start=1
):


    participante = chave[
        0
    ]


    noite = chave[
        1
    ]


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
    # DESCOBRIR COLUNAS
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
    # CARREGAR SOMENTE COLUNAS NECESSÁRIAS
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


    # ========================================================
    # CONVERTER PARA NUMPY
    # ========================================================

    timestamps = pd.to_numeric(

        motion[
            coluna_timestamp
        ],

        errors="coerce"

    ).to_numpy(
        dtype=np.float64
    )


    x = pd.to_numeric(

        motion[
            coluna_x
        ],

        errors="coerce"

    ).to_numpy(
        dtype=np.float64
    )


    y = pd.to_numeric(

        motion[
            coluna_y
        ],

        errors="coerce"

    ).to_numpy(
        dtype=np.float64
    )


    z = pd.to_numeric(

        motion[
            coluna_z
        ],

        errors="coerce"

    ).to_numpy(
        dtype=np.float64
    )


    # ========================================================
    # REMOVER VALORES INVÁLIDOS
    # ========================================================

    validos = (

        np.isfinite(
            timestamps
        )

        & np.isfinite(
            x
        )

        & np.isfinite(
            y
        )

        & np.isfinite(
            z
        )
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


    timestamps = normalizar_timestamp(
        timestamps
    )


    # ========================================================
    # ORDENAR POR TEMPO
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

    (
        timestamps_unicos,
        indices_unicos

    ) = np.unique(

        timestamps,

        return_index=True
    )


    timestamps = timestamps_unicos


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
    # PROCESSAR CADA ÉPOCA DA NOITE
    # ========================================================

    for indice_dataframe, linha in grupo.iterrows():


        inicio = float(
            linha[
                "_inicio_unix"
            ]
        )


        fim = (
            inicio
            + DURACAO_EPOCA_SEG
        )


        # ----------------------------------------------------
        # Pegamos uma pequena margem antes/depois.
        #
        # Isso ajuda a interpolação nas bordas.
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


        if len(
            ts_epoca
        ) < 2:

            raise RuntimeError(
                "\nPoucas amostras de movimento.\n"
                f"Participante: {participante}\n"
                f"Noite: {pasta_noite}\n"
                f"Época: {linha['epoca']}\n"
                f"Amostras: {len(ts_epoca)}"
            )


        # ----------------------------------------------------
        # Grade fixa:
        #
        # 0.0
        # 0.1
        # 0.2
        # ...
        # 29.9 segundos
        # ----------------------------------------------------

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
        # POSIÇÃO NO ARQUIVO NPY
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


        # ----------------------------------------------------
        # Quantidade bruta realmente dentro da época.
        # ----------------------------------------------------

        dentro_epoca = (

            (ts_epoca >= inicio)

            & (ts_epoca < fim)
        )


        dataset_dev.loc[
            indice_dataframe,
            "motion_raw_samples_usadas"
        ] = int(
            dentro_epoca.sum()
        )


    # ========================================================
    # GARANTIR QUE O BLOCO FOI ESCRITO
    # ========================================================

    motion_memmap.flush()


    print(
        f"  Épocas processadas: "
        f"{len(grupo)}"
    )


    # Liberar memória.

    del motion
    del timestamps
    del x
    del y
    del z


# ============================================================
# 9. VALIDAR ARQUIVO CRIADO
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
        "Foram encontrados NaN "
        "no dataset de movimento."
    )


if quantidade_inf != 0:

    raise ValueError(
        "Foram encontrados valores infinitos "
        "no dataset de movimento."
    )


# ============================================================
# 10. ESTATÍSTICAS DE AMOSTRAS ORIGINAIS
# ============================================================

print(
    "\nAmostras brutas por época:"
)


print(

    dataset_dev[
        "motion_raw_samples_usadas"
    ]

    .describe()

    .round(
        2
    )

    .to_string()
)


# ============================================================
# 11. SALVAR METADADOS
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


# Adicionar algumas colunas úteis se existirem.

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
# 12. RESUMO
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "DATASET CNN CRIADO"
)

print(
    "=" * 90
)


print(
    f"\nSinais:"
)


print(
    ARQUIVO_SINAIS
)


print(
    f"\nMetadados:"
)


print(
    ARQUIVO_METADATA
)


print(
    f"\nShape:"
)


print(
    motion_memmap.shape
)


print(
    f"\nCada época possui:"
)


print(
    f"  {len(CANAIS)} canais"
)


print(
    f"  {AMOSTRAS_POR_EPOCA} amostras por canal"
)


print(
    f"  {DURACAO_EPOCA_SEG:.0f} segundos"
)


print(
    f"  {FREQUENCIA_ALVO:.1f} Hz"
)


print(
    "\nParticipantes de teste não foram incluídos."
)


print(
    "\n✅ DATASET DE MOVIMENTO PARA CNN CONCLUÍDO"
)