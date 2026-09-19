@echo off
title NEU-CLS Detection Demo (one image)
set "HERE=%~dp0"
set "NAME=%~1"
if "%NAME%"=="" set /p NAME=Image name (e.g. crazing_10, scratches_7), then Enter: 
if "%NAME%"=="" (
  echo No image name given, exit.
  pause
  exit /b 1
)
if defined DEMO_PY goto :run_env
where python >nul 2>nul && goto :run_python
where py >nul 2>nul && goto :run_py
goto :nopython

:run_env
"%DEMO_PY%" "%HERE%launch.py" --image "%NAME%" --export-dir "%HERE%export"
goto :done

:run_python
python "%HERE%launch.py" --image "%NAME%" --export-dir "%HERE%export"
goto :done

:run_py
py -3 "%HERE%launch.py" --image "%NAME%" --export-dir "%HERE%export"
goto :done

:nopython
echo [ERROR] Python not found.
echo   Install Python 3.10~3.12, then run:
echo     pip install paddlepaddle pillow matplotlib pyyaml
pause

:done
echo.
echo Export dir: %HERE%export\
pause
