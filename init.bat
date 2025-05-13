@echo off
set CURRENT_DIR=%cd%
set ENV_FILE=%CURRENT_DIR%\.env

if not exist "%ENV_FILE%" (
    echo "%ENV_FILE%"
)

for /f "tokens=*" %%i in ('findstr /r "^PYTHONPATH=" "%ENV_FILE%"') do (
    set FOUND=1
)

if defined FOUND (
    powershell -Command "(gc '%ENV_FILE%') -replace 'PYTHONPATH=.*', 'PYTHONPATH=%CURRENT_DIR%' | Set-Content '%ENV_FILE%'"
) else (
    echo. >> "%ENV_FILE%"
    echo PYTHONPATH=%CURRENT_DIR% >> "%ENV_FILE%"
)