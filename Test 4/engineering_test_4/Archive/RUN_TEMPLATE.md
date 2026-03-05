# How to run (example)

This repo/package is language/library agnostic. Below is a simple Python example.

## Option A: Your own implementation
1) Create a virtualenv and install dependencies you plan to use
2) Run your script to generate the required `submission/` outputs

## Option B: Baseline runner (for reference)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn
python baseline_runner.py
```

Artifacts will be written to `baseline_output/`.
