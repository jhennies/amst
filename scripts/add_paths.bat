
echo OFF

set THISPATH=%~dp0
set AMSTPATH=%THISPATH%..
for %%i in ("%AMSTPATH%") do "AMSTPATHNORM=%%~fi"

set PYTHONPATH=%AMSTPATH%;%PYTHONPATH%
