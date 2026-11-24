# Instruções para restauração do banco de dados

Banco utilizado no projeto: `projeto_sono`

SGBD utilizado: MySQL 8.0

## Opção 1 — Restaurar pelo terminal

1. Abra o PowerShell ou Prompt de Comando.
2. Acesse a pasta onde está o arquivo `backup_projeto_sono_final.sql`.
3. Execute:

```powershell
& "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p < backup_projeto_sono_final.sql
```

4. Digite a senha do usuário `root` quando solicitado.
5. Após a importação, abra o MySQL Workbench e verifique se o schema `projeto_sono` foi criado e se as tabelas estão disponíveis.

## Opção 2 — Restaurar pelo MySQL Workbench

1. Abra o MySQL Workbench e conecte-se ao servidor local.
2. Acesse `Server > Data Import`.
3. Selecione `Import from Self-Contained File`.
4. Escolha o arquivo `backup_projeto_sono_final.sql`.
5. Inicie a importação em `Start Import`.
6. Ao finalizar, atualize a lista de schemas e verifique o banco `projeto_sono`.

## Estrutura sem dados

O arquivo `schema.sql` contém apenas a estrutura do banco, sem os registros.

Ele pode ser importado da mesma forma caso seja necessário recriar apenas as tabelas e relacionamentos.

## Observações

- O backup foi gerado a partir do MySQL Server 8.0.46.
- O arquivo `backup_projeto_sono_final.sql` contém a estrutura e os dados utilizados pelo sistema.
- O arquivo `schema.sql` contém somente a estrutura do banco.
- O diagrama `diagrama-er.png` representa as tabelas e relacionamentos do banco.
- Nenhuma senha de banco de dados está armazenada nos arquivos de entrega.
