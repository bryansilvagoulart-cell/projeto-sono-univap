# Projeto Sono

Sistema experimental de análise e classificação de estágios do sono desenvolvido para a disciplina de Projetos do curso de Engenharia da Computação da Universidade do Vale do Paraíba (UNIVAP).

O projeto utiliza frequência cardíaca e acelerometria para classificar épocas de 30 segundos em quatro classes:

- Wake (vigília)
- Light (sono leve — N1 + N2)
- Deep (sono profundo — N3)
- REM

O sistema possui processamento de dados, modelo de aprendizado de máquina, banco de dados MySQL, API em FastAPI e dashboard web.

> O projeto tem finalidade acadêmica e experimental. Os resultados não representam diagnóstico médico e não substituem polissonografia ou avaliação profissional.

## Integrantes

- Bryan Goulart Silva
- Emily Colen Honório

Curso: Engenharia da Computação  
Instituição: Universidade do Vale do Paraíba — UNIVAP  
Ano: 2026

## Tecnologias utilizadas

### Processamento e aprendizado de máquina
- Python 3.13
- Pandas
- NumPy
- SciPy
- Scikit-learn
- PyTorch
- Joblib

### Backend
- FastAPI
- Uvicorn
- mysql-connector-python

### Banco de dados
- MySQL 8.0

### Interface
- HTML
- CSS
- JavaScript

### Versionamento
- Git
- GitHub

## Dados utilizados

Foi utilizado o dataset público BIDSleep — *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels*, disponibilizado no PhysioNet.

Os dados possuem frequência cardíaca instantânea, acelerometria em três eixos e rótulos de estágio do sono derivados de EEG.

No desenvolvimento do modelo foram utilizados 15 participantes e 75 noites, totalizando 60.165 épocas válidas após o controle de qualidade.

O teste final foi realizado separadamente com três participantes não utilizados durante o desenvolvimento, totalizando 15 noites e 12.252 épocas.

## Modelo final

O modelo final combina:

1. Random Forest;
2. BiGRU com mecanismo de atenção;
3. ensemble probabilístico, com 75% de peso para o Random Forest e 25% para a BiGRU;
4. pós-processamento temporal pelo algoritmo de Viterbi.

A classificação final utiliza quatro classes: Wake, Light, Deep e REM.

### Resultados do teste independente

| Métrica | Resultado |
|---|---:|
| Accuracy | 58,13% |
| Balanced Accuracy | 58,35% |
| Macro F1 | 0,5664 |
| Weighted F1 | 0,5869 |

A diferença de desempenho entre participantes é uma das limitações observadas no projeto.

## Estrutura do sistema

O fluxo utilizado pelo sistema é:

```text
Dados de frequência cardíaca e acelerometria
                    |
                    v
          Processamento em Python
                    |
                    v
          Engenharia de atributos
                    |
                    v
     Random Forest + BiGRU + Viterbi
                    |
                    v
                MySQL
                    |
                    v
            API FastAPI
                    |
                    v
           Dashboard web
```

## Principais arquivos

```text
processar_noite.py
criar_dataset_ml_4_classes.py
criar_features_freq_tempo_ml.py
validar_modelo_4_classes_cv.py
treinar_modelo_final_4_classes.py
avaliar_teste_final_4_classes_corrigido.py

migrar_banco_4_classes_etapa1.py
importar_teste_final_mysql_4_classes.py
registrar_modelo_predicoes_mysql_4_classes.py
criar_metricas_noite_mysql_4_classes.py

api_sono_4_classes.py
dashboard_sono_v3.html
INICIAR_SISTEMA_V3.bat
requirements.txt
```

Os datasets brutos, arquivos processados, ambientes virtuais e modelos binários não são versionados no Git.

## Como executar

### Requisitos

- Python 3.13
- MySQL 8.0
- Git

### 1. Clonar o repositório

```bash
git clone https://github.com/bryansilvagoulart-cell/projeto-sono-univap.git
cd projeto-sono-univap
```

### 2. Criar o ambiente virtual

No PowerShell:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```powershell
pip install -r requirements.txt
```

### 4. Restaurar o banco de dados

O pacote final entregue na disciplina contém a pasta `banco-de-dados`, com:

- `schema.sql` — estrutura das tabelas;
- `backup_projeto_sono_final.sql` — estrutura e dados utilizados pelo sistema;
- `diagrama-er.png` — diagrama do banco;
- `instrucoes-banco.md` — instruções de restauração.

As instruções completas estão em:

```text
banco-de-dados/instrucoes-banco.md
```

### 5. Configurar o usuário da aplicação

A API utiliza o usuário MySQL:

```text
projeto_sono_app
```

Em uma instalação nova do banco, esse usuário deve ser criado e receber permissão sobre o schema `projeto_sono`.

Exemplo:

```sql
CREATE USER IF NOT EXISTS 'projeto_sono_app'@'localhost'
IDENTIFIED BY 'SENHA_ESCOLHIDA';

GRANT SELECT, INSERT, UPDATE, DELETE
ON projeto_sono.*
TO 'projeto_sono_app'@'localhost';

FLUSH PRIVILEGES;
```

A senha usada nesse comando deve ser informada ao iniciar a API. Nenhuma senha real é armazenada no repositório.

### 6. Iniciar o sistema

No Windows, a forma utilizada na demonstração é executar:

```text
INICIAR_SISTEMA_V3.bat
```

O inicializador abre a API em uma nova janela.

Na janela da API:

1. digite a senha do usuário `projeto_sono_app`;
2. aguarde a mensagem indicando que o Uvicorn está disponível em `http://127.0.0.1:8000`;
3. volte à janela do inicializador e pressione uma tecla;
4. o dashboard será aberto no navegador.

Também é possível iniciar a API manualmente:

```powershell
python api_sono_4_classes.py
```

Documentação automática da API:

```text
http://127.0.0.1:8000/docs
```

Verificação de funcionamento:

```text
http://127.0.0.1:8000/health
```

## API

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

## Dashboard

O dashboard permite selecionar o participante e a noite analisada e visualizar:

- distribuição das classes Wake, Light, Deep e REM;
- hipnograma;
- frequência cardíaca;
- movimento;
- confiança das classificações;
- comparação com o rótulo de referência;
- continuidade do sono;
- maior bloco de sono profundo;
- maior bloco REM;
- transições entre classes;
- épocas com baixa confiança;
- eventos de movimento elevado.

Os indicadores adicionais são experimentais e não devem ser utilizados para afirmar a ocorrência de condições clínicas ou eventos como sonambulismo, pesadelos ou paralisia do sono.

## Banco de dados

O banco `projeto_sono` armazena participantes, noites, épocas, rótulos de referência, modelos, classes do modelo, predições e métricas agregadas por noite.

Na versão final utilizada na apresentação, o banco possui:

```text
18 participantes
90 noites
72.417 épocas
2 modelos registrados
12.252 predições do modelo final
15 registros de métricas por noite do teste independente
```

## Segurança

O repositório não contém:

- senhas;
- arquivos `.env` com credenciais;
- ambiente virtual;
- datasets brutos;
- modelos binários treinados.

A senha do banco é solicitada em tempo de execução ou pode ser fornecida pela variável de ambiente `PROJETO_SONO_DB_PASSWORD`.

## Limitações

A classificação foi desenvolvida utilizando somente frequência cardíaca e acelerometria. Esses sinais não possuem a mesma quantidade de informação disponível em uma polissonografia completa.

O teste independente apresentou diferença de desempenho entre participantes, mostrando que a generalização para indivíduos não utilizados no desenvolvimento continua sendo uma limitação importante.

O sistema deve ser interpretado como uma aplicação experimental de engenharia e aprendizado de máquina.

## Referências principais

SONG, Tzu-An. *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels*. PhysioNet, 2026. DOI: 10.13026/a0sy-7t69.

SONG, Tzu-An et al. Sleep staging using wearable instantaneous heart rate and accelerometry signals. *IEEE Transactions on Biomedical Engineering*, 2026. DOI: 10.1109/TBME.2025.3612158.

IMTIAZ, Syed Anas. A systematic review of sensing technologies for wearable sleep staging. *Sensors*, v. 21, n. 5, 1562, 2021. DOI: 10.3390/s21051562.

RADHA, Mustafa et al. Sleep stage classification from heart-rate variability using long short-term memory neural networks. *npj Digital Medicine*, v. 4, 135, 2021. DOI: 10.1038/s41746-021-00510-8.

A lista completa de referências utilizadas no trabalho acompanha o pacote final de entrega.

## Repositório

https://github.com/bryansilvagoulart-cell/projeto-sono-univap
