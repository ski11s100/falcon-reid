@echo off
setlocal
cd /d "%~dp0"

if not exist "app\server.py" goto missing_project
if not exist "requirements.txt" goto missing_project

py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "FALCON_PY=py"
  set "FALCON_PY_ARGS=-3"
  goto python_found
)

python -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "FALCON_PY=python"
  set "FALCON_PY_ARGS="
  goto python_found
)

for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
  if exist "%%~fD\python.exe" set "FALCON_PY=%%~fD\python.exe"
)
if defined FALCON_PY goto python_found

echo Python was not found. Install Python 3.10 or newer from python.org.
pause
exit /b 1

:python_found
"%FALCON_PY%" %FALCON_PY_ARGS% -c "import PIL, numpy" >nul 2>&1
if errorlevel 1 goto install_dependencies
goto dependencies_ready

:install_dependencies
echo Installing required Python packages...
"%FALCON_PY%" %FALCON_PY_ARGS% -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check Python and your Internet connection.
  pause
  exit /b 1
)

:dependencies_ready
if not exist "demo_images\white_car_reference.png" "%FALCON_PY%" %FALCON_PY_ARGS% scripts\generate_demo.py
echo.
echo FALCON will be available at http://127.0.0.1:8000
echo Keep this window open. Press Ctrl+C to stop the server.
echo.
"%FALCON_PY%" %FALCON_PY_ARGS% -m app.server
pause
exit /b 0

:missing_project
echo Project files were not found next to start_falcon.bat.
echo Extract the ENTIRE ZIP archive first, then run start_falcon.bat from the extracted folder.
echo The extracted folder must contain app, scripts, requirements.txt and this file.
pause
exit /b 1
