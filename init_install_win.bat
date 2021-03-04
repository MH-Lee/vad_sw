python -m venv .\vad_venv
call .\vad_venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m pip install --upgrade pyqt5
python -m pip install pyqt5-tools

echo %USERNAME%
setx QT_QPA_PLATFORM_PLUGIN_PATH "C:\Users\%USERNAME%\Anaconda3\Lib\site-packages\PyQt5\Qt\plugins"

pause
