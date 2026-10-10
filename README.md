# Projeto Sono — Classificação de Quatro Classes com Sinais de Dispositivos Vestíveis

Projeto acadêmico desenvolvido na disciplina de **Projetos — Engenharia da Computação**, da **Universidade do Vale do Paraíba (UNIVAP)**.

O sistema realiza uma análise experimental do sono a partir de sinais de **frequência cardíaca** e **acelerometria**, utilizando aprendizado de máquina para classificar cada época de 30 segundos em uma das quatro classes:

- **Wake** — Vigília
- **Light** — Sono leve, correspondente ao agrupamento N1 + N2
- **Deep** — Sono profundo, correspondente ao estágio N3
- **REM** — Sono REM

> **Aviso:** este projeto possui finalidade acadêmica e experimental. Ele não substitui polissonografia, avaliação profissional ou diagnóstico médico.

---

## Autores

- **Bryan Goulart Silva**
- **Emily Colen Honório**

**Curso:** Engenharia da Computação  
**Instituição:** Universidade do Vale do Paraíba — UNIVAP

---

## Objetivo

O objetivo do projeto é desenvolver um sistema capaz de classificar estágios do sono utilizando sinais compatíveis com dispositivos vestíveis, explorando uma alternativa computacional menos invasiva para análise experimental de múltiplas noites.

O fluxo desenvolvido engloba:

1. aquisição de dados públicos;
2. sincronização dos sinais fisiológicos;
3. divisão das noites em épocas de 30 segundos;
4. controle de qualidade;
5. engenharia de atributos;
6. treinamento e validação dos modelos;
7. teste independente;
8. persistência dos resultados em MySQL;
9. disponibilização dos dados por uma API FastAPI;
10. desenvolvimento de interface/dashboard.

---

## Dataset

Foi utilizado o dataset público **BIDSleep — A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels**.

Os registros incluem:

- frequência cardíaca instantânea;
- acelerometria em três eixos;
- rótulos de estágio do sono derivados de EEG;
- múltiplas noites por participante.

O rótulo utilizado como referência no projeto é o `expert_label`.

### Rótulos originais

| Código | Estágio |
|---:|---|
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |
| 5 | Unknown |

### Taxonomia utilizada pelo modelo final

| Código | Classe |
|---:|---|
| 0 | Wake |
| 1 | Light (N1 + N2) |
| 2 | Deep (N3) |
| 3 | REM |

A redução para quatro classes foi adotada após experimentos que mostraram maior dificuldade na classificação isolada do estágio N1.

---

## Base de desenvolvimento

A base principal utilizada no desenvolvimento do modelo possui:

- **15 participantes**
- **75 noites**
- **60.165 épocas válidas**
- **30 atributos principais**

Distribuição das classes:

| Classe | Épocas | Percentual |
|---|---:|---:|
| Wake | 6.942 | 11,54% |
| Light | 27.347 | 45,45% |
| Deep | 10.964 | 18,22% |
| REM | 14.912 | 24,79% |

---

## Pré-processamento

Os sinais foram sincronizados com os rótulos de sono e organizados em épocas de **30 segundos**.

Para frequência cardíaca foram calculados, entre outros:

- média;
- mínimo;
- máximo;
- desvio-padrão;
- quantidade de amostras.

Para movimento foram utilizados os três eixos do acelerômetro e a magnitude do vetor de aceleração, permitindo calcular:

- movimento médio;
- desvio-padrão;
- movimento máximo;
- quantidade de amostras.

Também foram criados atributos temporais, incluindo:

- diferenças entre épocas consecutivas;
- estatísticas móveis em janelas de 5, 10 e 20 épocas;
- tempo desde o início da noite;
- representação cíclica do horário local.

---

## Controle de qualidade

Os critérios utilizados são critérios de engenharia para qualidade mínima do sinal e **não são critérios médicos**.

Uma época é mantida quando atende aos seguintes requisitos:

- cobertura temporal de pelo menos **95%**;
- maior lacuna do sinal de movimento de no máximo **1 segundo**;
- pelo menos **4 amostras de frequência cardíaca**.

---

## Arquitetura do modelo

O modelo final combina três componentes:

```text
Dados fisiológicos
      │
      ▼
Engenharia de 30 atributos
      │
      ├───────────────┐
      ▼               ▼
Random Forest       BiGRU + Attention
      │               │
      └───────┬───────┘
              ▼
       Ensemble probabilístico
      RF 75% + BiGRU 25%
              │
              ▼
            Viterbi
          peso = 0,80
              │
              ▼
   Wake / Light / Deep / REM
```

### Random Forest

Configuração principal:

- 300 árvores;
- `max_depth = 16`;
- `min_samples_split = 10`;
- `min_samples_leaf = 5`;
- `max_features = sqrt`;
- `class_weight = balanced_subsample`.

### BiGRU + Attention

A rede utiliza:

- GRU bidirecional;
- mecanismo de atenção;
- contexto de **21 épocas**, equivalente a **10,5 minutos**;
- `hidden_size = 64`;
- 8 épocas de treinamento;
- pesos de classe;
- `label_smoothing = 0.05`.

### Ensemble

As probabilidades finais antes do Viterbi são calculadas por:

```text
75% Random Forest + 25% BiGRU
```

### Viterbi

O algoritmo de Viterbi é utilizado como pós-processamento temporal para favorecer sequências de sono mais coerentes.

Peso utilizado:

```text
0,80
```

---

## Validação cruzada

A configuração foi selecionada por validação cruzada com **5 folds**, utilizando divisão por participante para evitar que o mesmo indivíduo aparecesse simultaneamente no treino e na validação de um mesmo fold.

Resultados médios:

| Métrica | Resultado |
|---|---:|
| Accuracy | **65,09%** |
| Balanced Accuracy | **64,68%** |
| Macro F1 | **0,6415** |
| Weighted F1 | **0,6487** |

F1 médio por classe:

| Classe | F1 |
|---|---:|
| Wake | 0,5958 |
| Light | 0,6525 |
| Deep | 0,6584 |
| REM | 0,6592 |

---

## Teste final independente

Após congelar a arquitetura e os hiperparâmetros, foi criado um novo conjunto de teste com participantes que não foram utilizados no desenvolvimento:

- **Bidslab01**
- **Bidslab35**
- **Bidslab47**

O teste final possui:

- **3 participantes**
- **15 noites**
- **12.252 épocas**

Resultados oficiais:

| Métrica | Resultado |
|---|---:|
| Accuracy | **58,13%** |
| Balanced Accuracy | **58,35%** |
| Macro F1 | **0,5664** |
| Weighted F1 | **0,5869** |

### Resultado por classe

| Classe | Precision | Recall | F1 |
|---|---:|---:|---:|
| Wake | 0,5800 | 0,4316 | 0,4949 |
| Light | 0,7040 | 0,5336 | 0,6071 |
| Deep | 0,6393 | 0,7010 | 0,6687 |
| REM | 0,3933 | 0,6677 | 0,4950 |

### Resultado por participante

| Participante | Accuracy |
|---|---:|
| Bidslab01 | 69,90% |
| Bidslab35 | 60,14% |
| Bidslab47 | 46,98% |

A diferença de desempenho entre participantes reforça a dificuldade de generalização interindividual utilizando apenas frequência cardíaca e movimento.

---

## Banco de dados

O projeto utiliza **MySQL** para armazenar:

- participantes;
- noites;
- épocas;
- estágios de referência;
- modelos treinados;
- classes do modelo;
- predições;
- métricas agregadas por noite.

A estrutura atual preserva os cinco estágios originais de referência e mantém separadamente a taxonomia de quatro classes utilizada pelo modelo final.

Estado atual do banco:

```text
18 participantes
90 noites
72.417 épocas
2 modelos registrados
12.252 predições do modelo final de 4 classes
15 registros de métricas por noite do teste final
```

---

## API

A API foi construída com **FastAPI**.

Arquivo principal:

```text
api_sono_4_classes.py
```

Principais rotas:

```text
GET /
GET /health
GET /modelo
GET /participantes
GET /participantes/{codigo}/noites
GET /noites/{id_noite}
GET /noites/{id_noite}/estagios
```

Documentação automática:

```text
http://127.0.0.1:8000/docs
```

A API acessa o MySQL utilizando o usuário da aplicação e solicita a senha em tempo de execução ou por variável de ambiente.

---

## Tecnologias utilizadas

- Python 3.13
- Pandas
- NumPy
- SciPy
- Scikit-learn
- PyTorch
- Joblib
- MySQL
- mysql-connector-python
- FastAPI
- Uvicorn
- Git / GitHub
- Visual Studio Code

---

## Estrutura principal do repositório

```text
Projeto_Sono/
│
├── processar_noite.py
├── criar_dataset_ml_4_classes.py
├── criar_features_freq_tempo_ml.py
│
├── validar_modelo_4_classes_cv.py
├── travar_teste_final_4_classes.py
├── baixar_teste_final_4_classes.py
├── processar_base_teste_final_4_classes.py
├── aplicar_controle_qualidade_teste_final_4_classes.py
├── criar_features_teste_final_4_classes.py
├── treinar_modelo_final_4_classes.py
├── avaliar_teste_final_4_classes_corrigido.py
│
├── migrar_banco_4_classes_etapa1.py
├── importar_teste_final_mysql_4_classes.py
├── registrar_modelo_predicoes_mysql_4_classes.py
├── criar_metricas_noite_mysql_4_classes.py
│
├── api_sono_4_classes.py
│
├── .gitignore
└── README.md
```

Os diretórios de dados brutos, dados processados, ambientes virtuais e modelos treinados são ignorados pelo Git para evitar o versionamento de arquivos grandes, temporários ou sensíveis.

---

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/bryansilvagoulart-cell/projeto-sono-univap.git
cd projeto-sono-univap
```

### 2. Criar ambiente virtual

No Windows:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
```

### 3. Instalar dependências principais

```powershell
pip install -r requirements.txt
```

### 4. Banco de dados

É necessário possuir uma instância MySQL configurada com o schema:

```text
projeto_sono
```

As credenciais não são armazenadas no repositório.

### 5. Executar a API

```powershell
python api_sono_4_classes.py
```

Depois, acessar:

```text
http://127.0.0.1:8000/docs
```

---

## Segurança e privacidade

Este repositório **não deve conter**:

- senhas do banco;
- arquivos `.env`;
- ambientes virtuais;
- datasets brutos;
- arquivos processados de grande volume;
- modelos binários treinados;
- dumps do banco de dados.

Esses itens são excluídos por meio do arquivo `.gitignore`.

---

## Limitações

A classificação utiliza frequência cardíaca e acelerometria e, portanto, não possui a mesma quantidade de informação fisiológica disponível em uma polissonografia completa.

O desempenho do teste independente mostrou variação significativa entre participantes, indicando que a generalização entre indivíduos ainda é uma limitação relevante.

O sistema também não deve ser utilizado para afirmar a ocorrência de eventos como:

- sonhos;
- pesadelos;
- sonambulismo;
- paralisia do sono;
- microdespertares clínicos.

Os resultados devem ser interpretados exclusivamente como classificações e indicadores computacionais experimentais.

---

## Próximas etapas

- finalizar o dashboard web;
- integrar visualizações de hipnograma;
- comparar noites de um mesmo participante;
- apresentar qualidade e cobertura dos sinais;
- explorar indicadores experimentais de continuidade e fragmentação;
- preparar integração futura com hardware/dispositivo vestível.

---

## Referências principais

SONG, Tzu-An. **A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels**. PhysioNet, 2026. DOI: 10.13026/a0sy-7t69.

SONG, Tzu-An et al. Sleep staging using wearable instantaneous heart rate and accelerometry signals. **IEEE Transactions on Biomedical Engineering**, 2026. DOI: 10.1109/TBME.2025.3612158.

IMTIAZ, Syed Anas. A systematic review of sensing technologies for wearable sleep staging. **Sensors**, v. 21, n. 5, 1562, 2021. DOI: 10.3390/s21051562.

RADHA, Mustafa et al. Sleep stage classification from heart-rate variability using long short-term memory neural networks. **npj Digital Medicine**, v. 4, 135, 2021. DOI: 10.1038/s41746-021-00510-8.

---

## Status do projeto

```text
[OK] Dataset e processamento
[OK] Controle de qualidade
[OK] Engenharia de atributos
[OK] Modelo de quatro classes
[OK] Validação cruzada
[OK] Teste final independente
[OK] Banco MySQL
[OK] API FastAPI
[EM DESENVOLVIMENTO] Dashboard
[EM DESENVOLVIMENTO] Apresentação final e documentação
```

---

**Projeto acadêmico — Engenharia da Computação — UNIVAP — 2026**
