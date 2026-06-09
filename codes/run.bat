@echo off
REM ============================================================
REM IROTE Run Script (Windows)
REM Uses cottonagent conda environment
REM ============================================================

set PYTHON=E:\anaconda\envs\cottonagent\python.exe
set PYTHONPATH=%~dp0

echo ========================================
echo IROTE - CottonAgent Environment
echo ========================================
echo.

REM Run IROTE optimization
%PYTHON% optimization_irote.py %*

pause
