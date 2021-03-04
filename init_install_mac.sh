echo "$USER"
cd "$(dirname "$0")"
echo $PWD
python3 -m venv ./vad_venv
source ./vad_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod a+x ./vad_exe_mac.sh
