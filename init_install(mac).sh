echo "$USER"
cd /Users/$USER/Desktop/VAD_SW
python3 -m venv ./vad_venv
source ./vad_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
