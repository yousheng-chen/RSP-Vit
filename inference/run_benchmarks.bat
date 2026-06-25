@echo off
rem ------------------------------------------------------------------------------
rem  run_benchmarks.bat
rem  Run all 4 ViT variants through the deployment-complexity
rem  benchmark and generate paper-ready summary tables.
rem ------------------------------------------------------------------------------

setlocal enabledelayedexpansion
2rem --- Find Python ---
set PYTHON_CMD=python
where python >nul 2>nul
if %ERRORLEVEL%% neq 0 (
    echo [ERROR] Python not found. Please activate your conda env first.
    pause
    exit /b 1
)

set SCRIPT_DIR=~%d0p%
set PROJECT_ROOT=%SCRIPT_DIR%..
set RESULTS_DIR=%SCRIPT_DIR%results
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo ===============================================================================
echo  RSP-ViT Deployment Complexity Benchmark
echo  Results dir: %RESULTS_DIR%
echo ===============================================================================
echo.

echo Stage 1: Single-batch benchmark *FLOPs, latency, memory*
echo ----------------------------------------------------------------------
%PYTHON_CMD% ''%SCRIPT_DIR%benchmark_models.py' --
--img-size 192 -
-num-warmup 50 ---num-iters 200 ---noise-level -10db --output "%RESULTS_DIR%\\benchmark_results.csv" ---print-json

if %ERRORLEVEL%% neq 0 (
    echo [ERROR] Benchmark failed.
    pause
    exit /b 1
)
echo.

echo Stage 2: Batch-size sweep
echo ------------------------------------------------------------------
%PYTHON_CMD% ''%SCRIPT_DIR%batch_sweep.py'
if %ERRORLEVEL%% neq 0 (
    echo [WARN] Batch sweep failed, continuing...
)
echo.

echo Stage 3: Generate paper-ready tables
echo ------------------------------------------------------------------------
%PYTHON_CMD% ''%SCRIPT_DIR%generate_benchmark_table.py' --
--input "%RESULTS_DIR%\benchmark_results.csv" ---output-dir "%RESULTS_DIR%\tables"

if %ERRORLEVEL%% neq 0 (
    echo [ERROR] Table generation failed.
    pause
    exit /b 1
)
echo.

echo ===============================================================================
echo  Done! All results saved to:
echo    %RESULTS_DIR%
echo ===============================================================================
pause
endlocal