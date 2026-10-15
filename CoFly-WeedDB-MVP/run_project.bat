@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Python environment not found: "%PYTHON_EXE%"
    exit /b 1
)

"%PYTHON_EXE%" "%PROJECT_DIR%main.py" %*
