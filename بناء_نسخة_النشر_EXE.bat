@echo off
cd /d "%~dp0"
echo Building standalone portable BluerGirl.exe release...
pyinstaller --noconfirm --onedir --windowed --name "BluerGirl" --collect-all customtkinter --collect-all nudenet --collect-all onnxruntime --add-data "models;models" main.py
if exist "models" (
    xcopy /E /I /Y "models" "dist\BluerGirl\models"
)
echo.
echo ============================================================
echo Build Complete!
echo You can now zip and publish the folder: dist\BluerGirl
echo Run: dist\BluerGirl\BluerGirl.exe
echo ============================================================
pause
