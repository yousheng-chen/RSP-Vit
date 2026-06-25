@echo off
setlocal

REM Runs the shared-weight noise sweep (5/10/20/30 by default) without any manual steps between runs.
REM You can override the python path by setting VIT_PYTHON before running:
REM   set VIT_PYTHON=C:\Users\13023\.conda\envs\vit\python.exe
REM   run_noise_sweep.cmd

if defined VIT_PYTHON (
  set "PYEXE=%VIT_PYTHON%"
) else (
  set "PYEXE=C:\Users\13023\.conda\envs\vit\python.exe"
)

set "DATA_DIR=data\split_data"

REM Noise levels (percent). Edit this line if you want a different sweep.
set "NOISE_LEVELS=5 10 20 30"

"%PYEXE%" src\noise_sweep_shared.py --data-dir "%DATA_DIR%" --noise-prob 1.0 --noise-levels %NOISE_LEVELS%

endlocal
