-- Projeto Sono - configuração do usuário utilizado pela API
-- Altere ALTERE_ESTA_SENHA antes de executar.

CREATE USER IF NOT EXISTS 'projeto_sono_app'@'localhost'
IDENTIFIED BY 'ALTERE_ESTA_SENHA';

GRANT SELECT, INSERT, UPDATE, DELETE
ON projeto_sono.*
TO 'projeto_sono_app'@'localhost';

FLUSH PRIVILEGES;

-- Caso o usuário já exista e seja necessário definir uma nova senha:
-- ALTER USER 'projeto_sono_app'@'localhost'
-- IDENTIFIED BY 'ALTERE_ESTA_SENHA';
