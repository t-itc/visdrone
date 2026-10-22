Usage

python VisDrone.py prepare     # download, convert to YOLO labels, write visdrone.yaml
python VisDrone.py stats       # class counts and how many boxes fall under 32×32 px at 640/960/1280
python sweep.py --epochs 3 --fraction 0.05    # quick smoke test first
python sweep.py                # full sweep



Project Structure
visdrone/
├── pyproject.toml
├── README.md
├── datasets/              # raw/processed data (gitignored)
├── runs/                  # YOLO training outputs (gitignored)
├── models/                # exported/serialised artefacts (gitignored)
│   └── visdrone.joblib
├── src/visdrone/
│   ├── __init__.py
│   ├── data.py            # data processing (split out of VisDrone.py)
│   ├── train.py           # YOLO training/eval (rest of VisDrone.py)
│   ├── sweep.py           # hyperparameter tuning wrapper
│   └── serve/
│       ├── __init__.py
│       └── app.py         # FastAPI app (formerly deploy.py)
├── tests/
├── Dockerfile             # deployment config lives at repo root
└── scripts/               # optional thin CLI entry points