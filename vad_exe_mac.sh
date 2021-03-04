echo "$USER"
cd "$(dirname "$0")"
echo $PWD
source ./vad_venv/bin/activate
python -V
python vad_qt.py
