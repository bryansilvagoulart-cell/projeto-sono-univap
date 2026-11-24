@echo off
setlocal
cd /d "%~dp0"
title Projeto Sono - Inicializador

echo ============================================================
echo                  PROJETO SONO - UNIVAP
echo ============================================================
echo.

if not exist ".venv313\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual .venv313 nao encontrado.
    echo Coloque este arquivo na raiz do projeto.
    echo.
    pause
    exit /b 1
)

if not exist "api_sono_4_classes.py" (
    echo [ERRO] api_sono_4_classes.py nao encontrado.
    echo.
    pause
    exit /b 1
)

set "DASHBOARD="
if exist "dashboard_sono_v3.html" (
    set "DASHBOARD=dashboard_sono_v3.html"
) else if exist "dashboard_sono_v2.html" (
    set "DASHBOARD=dashboard_sono_v2.html"
) else if exist "dashboard_sono.html" (
    set "DASHBOARD=dashboard_sono.html"
)

if not defined DASHBOARD (
    echo [ERRO] Nenhum dashboard encontrado.
    echo.
    pause
    exit /b 1
)

echo Iniciando a API...
echo.
start "Projeto Sono - API" cmd /k ""%CD%\.venv313\Scripts\python.exe" "%CD%\api_sono_4_classes.py""

echo.
echo Na janela da API:
echo 1. Digite a senha do banco.
echo 2. Aguarde aparecer:
echo    Uvicorn running on http://127.0.0.1:8000
echo.
echo Depois volte para esta janela.
echo.
pause

echo.
echo Abrindo: %DASHBOARD%
start "" "%CD%\%DASHBOARD%"

timeout /t 2 /nobreak >nul
exit /b 0
