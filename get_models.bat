@echo off
setlocal EnableExtensions

REM --- find Comfy root by walking up until main.py and models/ exist ---
set "CUR=%~dp0"
:findroot
if exist "%CUR%main.py" if exist "%CUR%models\" (
  set "COMFY_ROOT=%CUR%"
  goto found
)
for %%I in ("%CUR%..") do set "NEXT=%%~fI\"
if /I "%NEXT%"=="%CUR%" (
  echo ERROR: Could not locate ComfyUI root from "%~dp0"
  exit /b 1
)
set "CUR=%NEXT%"
goto findroot

:found
set "MODELS_DIR=%COMFY_ROOT%models\checkpoints\DWPose"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
echo Comfy root: "%COMFY_ROOT%"
echo Target dir: "%MODELS_DIR%"
echo.

REM ---- selection & force flags ----
set "WHAT=%1"
set "FORCE=0"
if /I "%WHAT%"=="--force" ( set "FORCE=1" & set "WHAT=all" )
if /I "%2"=="--force" set "FORCE=1"
if "%WHAT%"=="" set "WHAT=all"

call :grab "%MODELS_DIR%" "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/yolox_l.torchscript.pt" "yolox_l.torchscript.pt" "%WHAT%" %FORCE%
call :grab "%MODELS_DIR%" "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt" "dw-ll_ucoco_384_bs5.torchscript.pt" "%WHAT%" %FORCE%
call :grab "%MODELS_DIR%" "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx" "yolox_l.onnx" "%WHAT%" %FORCE%
call :grab "%MODELS_DIR%" "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx" "dw-ll_ucoco_384.onnx" "%WHAT%" %FORCE%
echo.
echo Done → "%MODELS_DIR%"
exit /b 0

:grab
set "OUTDIR=%~1"
set "URL=%~2"
set "FNAME=%~3"
set "WHAT=%~4"
set "FORCE=%~5"

echo %WHAT%| findstr /I "^ts$" >nul && echo "%FNAME%"|findstr /I ".onnx" >nul && goto :eof
echo %WHAT%| findstr /I "^onnx$" >nul && echo "%FNAME%"|findstr /I ".torchscript.pt" >nul && goto :eof

if %FORCE%==0 if exist "%OUTDIR%\%FNAME%" (
  echo Skipping existing %FNAME%
  goto :eof
)

echo Downloading %FNAME% ...
where curl >nul 2>nul
if %errorlevel%==0 (
  curl -L "%URL%" -o "%OUTDIR%\%FNAME%"
) else (
  powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%OUTDIR%\%FNAME%'"
)
if exist "%OUTDIR%\%FNAME%" (
  for %%A in ("%OUTDIR%\%FNAME%") do echo   saved (%%~zA bytes)
) else (
  echo   FAILED: %FNAME%
)
goto :eof
