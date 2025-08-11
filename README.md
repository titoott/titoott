# Mega Predictor (Tkinter)

Quick start

1) Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Run the app

```bash
python mega_predictor.py
```

Notes
- If `xgboost` is not available on your platform, the app will disable the XG model automatically.
- Data, models and optional image are stored in `./mega_data` by default. Override via env var `MEGA_MODEL_DIR`.
- On some Linux environments you may need system Tk libraries to run Tkinter GUIs. For Debian/Ubuntu:
  - `sudo apt-get update && sudo apt-get install -y python3-tk`