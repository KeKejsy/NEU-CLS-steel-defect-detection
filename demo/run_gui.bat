@echo off
title NEU-CLS Detection Demo (GUI)
set "HERE=%~dp0"
if defined DEMO_PY goto :run_env
where python >nul 2>nul && goto :run_python
where py >nul 2>nul && goto :run_py
goto :nopython

:run_env
"%DEMO_PY%" "%HERE%launch.py" --gui
goto :done

:run_python
python "%HERE%launch.py" --gui
goto :done

:run_py
py -3 "%HERE%launch.py" --gui
goto :done

:nopython
echo [ERROR] Python not found.
echo   Install Python 3.10~3.12, then run:
echo     pip install paddlepaddle pillow matplotlib pyyaml
echo   (GPU version: pip install paddlepaddle-gpu)
pause

:done
