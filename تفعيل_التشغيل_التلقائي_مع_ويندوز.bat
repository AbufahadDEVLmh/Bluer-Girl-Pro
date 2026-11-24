@echo off
cd /d "%~dp0"
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_FILE=%STARTUP_DIR%\BluerGirl_AI_Server.vbs"

echo Set WshShell = CreateObject("WScript.Shell") > "%VBS_FILE%"
echo WshShell.CurrentDirectory = "%~dp0" >> "%VBS_FILE%"
echo WshShell.Run "pythonw.exe ""%~dp0chrome_ai_server.py""", 0, False >> "%VBS_FILE%"

start "" pythonw.exe "%~dp0chrome_ai_server.py"
echo ============================================================
echo Auto-Start Enabled!
echo Bluer Girl AI Server is now active and will start automatically with Windows.
echo ============================================================
