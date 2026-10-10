from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_MULTIESCALA = Path(
    "data/processed/dataset_ml_features_multiescala.csv"
)

ARQUIVO_EPOCAS = Path(
    "data/processed/dataset_geral_epocas.csv"
)

ARQUIVO_SAIDA = Path(
    "data/processed/dataset_ml_features_freq_tempo.csv"
)


# ============================================================
# INÍCIO
# ============================================================

print("=" * 80)
print("CRIAÇÃO DE FEATURES DE FREQUÊNCIA DE AMOSTRAGEM E TEMPO")
print("=" * 80)


# ============================================================
# 1. CARREGAR DATASET MULTIESCALA
# ============================================================

print(
    "\nCarregando dataset multiescala..."
)


multiescala = pd.read_csv(
    ARQUIVO_MULTIESCALA
)


print(
    f"Épocas multiescala: "
    f"{len(multiescala)}"
)


if len(multiescala) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas no dataset multiescala, "
        f"mas encontramos {len(multiescala)}."
    )


# ============================================================
# 2. CARREGAR DADOS ORIGINAIS DAS ÉPOCAS
# ============================================================

print(
    "\nCarregando informações das épocas..."
)


epocas = pd.read_csv(
    ARQUIVO_EPOCAS,
    usecols=[
        "participante",
        "noite",
        "epoca",
        "inicio",
        "hr_samples",
        "motion_samples"
    ]
)


print(
    f"Épocas originais carregadas: "
    f"{len(epocas)}"
)


if len(epocas) != 60165:

    raise ValueError(
        f"Esperávamos 60165 épocas, "
        f"mas encontramos {len(epocas)}."
    )


# ============================================================
# 3. VALIDAR CHAVES
# ============================================================

chaves = [
    "participante",
    "noite",
    "epoca"
]


duplicados_multiescala = (
    multiescala
    .duplicated(
        subset=chaves
    )
    .sum()
)


duplicados_epocas = (
    epocas
    .duplicated(
        subset=chaves
    )
    .sum()
)


if duplicados_multiescala != 0:

    raise ValueError(
        "Existem chaves duplicadas "
        "no dataset multiescala."
    )


if duplicados_epocas != 0:

    raise ValueError(
        "Existem chaves duplicadas "
        "no dataset de épocas."
    )


print(
    "✅ Chaves validadas."
)


# ============================================================
# 4. JUNTAR OS DOIS DATASETS
# ============================================================

print(
    "\nIntegrando hr_samples e tempo..."
)


dataset = multiescala.merge(

    epocas,

    on=[
        "participante",
        "noite",
        "epoca"
    ],

    how="left",

    validate="one_to_one"
)


if len(dataset) != 60165:

    raise ValueError(
        "A junção alterou a quantidade "
        "de épocas."
    )


if dataset[
    [
        "hr_samples",
        "inicio"
    ]
].isna().sum().sum() != 0:

    raise ValueError(
        "Algumas épocas não encontraram "
        "dados correspondentes."
    )


print(
    "✅ Integração realizada."
)


# ============================================================
# 5. FREQUÊNCIA DE AMOSTRAGEM DO IHR
# ============================================================
#
# Cada época possui 30 segundos.
#
# Exemplo:
#
# 6 medições / 30 segundos
#
# = 0,2 Hz
#
# ============================================================

dataset[
    "hr_sampling_hz"
] = (

    dataset[
        "hr_samples"
    ]

    / 30.0
)


# ============================================================
# 6. ORGANIZAR ORDEM TEMPORAL
# ============================================================

dataset = (

    dataset

    .sort_values(
        [
            "id_noite",
            "segmento",
            "epoca"
        ]
    )

    .reset_index(
        drop=True
    )
)


grupos = dataset.groupby(
    [
        "id_noite",
        "segmento"
    ],
    sort=False
)


# ============================================================
# 7. MUDANÇA DA FREQUÊNCIA DE AMOSTRAGEM
# ============================================================

dataset[
    "hr_sampling_delta1"
] = (

    grupos[
        "hr_sampling_hz"
    ]

    .diff()

    .fillna(0)
)


# ============================================================
# 8. JANELA DE 2,5 MINUTOS
# ============================================================

dataset[
    "hr_sampling_roll5_mean"
] = (

    grupos[
        "hr_sampling_hz"
    ]

    .transform(
        lambda serie:
            serie
            .rolling(
                window=5,
                min_periods=1
            )
            .mean()
    )
)


dataset[
    "hr_sampling_roll5_std"
] = (

    grupos[
        "hr_sampling_hz"
    ]

    .transform(
        lambda serie:
            serie
            .rolling(
                window=5,
                min_periods=1
            )
            .std(
                ddof=0
            )
    )
)


# ============================================================
# 9. JANELA DE 5 MINUTOS
# ============================================================

dataset[
    "hr_sampling_roll10_mean"
] = (

    grupos[
        "hr_sampling_hz"
    ]

    .transform(
        lambda serie:
            serie
            .rolling(
                window=10,
                min_periods=1
            )
            .mean()
    )
)


dataset[
    "hr_sampling_roll10_std"
] = (

    grupos[
        "hr_sampling_hz"
    ]

    .transform(
        lambda serie:
            serie
            .rolling(
                window=10,
                min_periods=1
            )
            .std(
                ddof=0
            )
    )
)


# ============================================================
# 10. JANELA DE 10 MINUTOS
# ============================================================

dataset[
    "hr_sampling_roll20_mean"
] = (

    grupos[
        "hr_sampling_hz"
    ]

    .transform(
        lambda serie:
            serie
            .rolling(
                window=20,
                min_periods=1
            )
            .mean()
    )
)


dataset[
    "hr_sampling_roll20_std"
] = (

    grupos[
        "hr_sampling_hz"
    ]

    .transform(
        lambda serie:
            serie
            .rolling(
                window=20,
                min_periods=1
            )
            .std(
                ddof=0
            )
    )
)


# ============================================================
# 11. TEMPO DESDE O INÍCIO DA GRAVAÇÃO
# ============================================================
#
# Cada época representa 30 segundos.
#
# epoca 0   = 0 min
# epoca 10  = 5 min
# epoca 100 = 50 min
#
# ============================================================

dataset[
    "tempo_desde_inicio_min"
] = (

    dataset[
        "epoca"
    ]

    * 0.5
)


dataset[
    "tempo_desde_inicio_horas"
] = (

    dataset[
        "tempo_desde_inicio_min"
    ]

    / 60.0
)


# ============================================================
# 12. HORÁRIO LOCAL
# ============================================================
#
# Os timestamps das épocas estão em UTC.
#
# Para representar horário do relógio:
#
# UTC
# ->
# America/New_York
#
# pois o dataset foi coletado nessa região.
#
# ============================================================

dataset[
    "inicio"
] = pd.to_datetime(
    dataset[
        "inicio"
    ],
    utc=True
)


inicio_local = (

    dataset[
        "inicio"
    ]

    .dt

    .tz_convert(
        "America/New_York"
    )
)


hora_decimal = (

    inicio_local.dt.hour

    + inicio_local.dt.minute / 60.0

    + inicio_local.dt.second / 3600.0
)


# ============================================================
# 13. REPRESENTAÇÃO CÍCLICA DO HORÁRIO
# ============================================================
#
# 23:59 e 00:01 são horários muito próximos.
#
# Se usássemos apenas:
#
# 23.98 e 0.02
#
# matematicamente pareceriam distantes.
#
# Seno e cosseno resolvem isso.
#
# ============================================================

dataset[
    "hora_local_sin"
] = np.sin(

    2
    * np.pi
    * hora_decimal
    / 24.0
)


dataset[
    "hora_local_cos"
] = np.cos(

    2
    * np.pi
    * hora_decimal
    / 24.0
)


# ============================================================
# 14. LISTA DAS NOVAS FEATURES
# ============================================================

NOVAS_FEATURES = [

    "hr_sampling_hz",

    "hr_sampling_delta1",

    "hr_sampling_roll5_mean",
    "hr_sampling_roll5_std",

    "hr_sampling_roll10_mean",
    "hr_sampling_roll10_std",

    "hr_sampling_roll20_mean",
    "hr_sampling_roll20_std",

    "tempo_desde_inicio_min",
    "tempo_desde_inicio_horas",

    "hora_local_sin",
    "hora_local_cos"
]


# ============================================================
# 15. VALIDAR FEATURES
# ============================================================

print(
    "\nValidando novas features..."
)


ausentes = (

    dataset[
        NOVAS_FEATURES
    ]

    .isna()

    .sum()

    .sum()
)


infinitos = (

    np.isinf(
        dataset[
            NOVAS_FEATURES
        ].to_numpy()
    )

    .sum()
)


print(
    f"Linhas finais: "
    f"{len(dataset)}"
)


print(
    f"Novas features: "
    f"{len(NOVAS_FEATURES)}"
)


print(
    f"Valores ausentes: "
    f"{ausentes}"
)


print(
    f"Valores infinitos: "
    f"{infinitos}"
)


if ausentes != 0:

    raise ValueError(
        "Existem valores ausentes "
        "nas novas features."
    )


if infinitos != 0:

    raise ValueError(
        "Existem valores infinitos "
        "nas novas features."
    )


# ============================================================
# 16. ESTATÍSTICAS DA FREQUÊNCIA
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FREQUÊNCIA DE AMOSTRAGEM DO IHR"
)

print(
    "=" * 80
)


print(
    dataset[
        "hr_sampling_hz"
    ]

    .describe()

    .round(4)

    .to_string()
)


# ============================================================
# 17. SALVAR
# ============================================================

ARQUIVO_SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)


dataset.to_csv(
    ARQUIVO_SAIDA,
    index=False
)


# ============================================================
# RESULTADO
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "DATASET COM FREQ + TEMPO CRIADO"
)

print(
    "=" * 80
)


print(
    "\nArquivo:"
)

print(
    ARQUIVO_SAIDA
)


print(
    "\nFeatures adicionadas:"
)


for feature in NOVAS_FEATURES:

    print(
        f"  - {feature}"
    )


print(
    "\n✅ DATASET AVANÇADO CRIADO COM SUCESSO"
)