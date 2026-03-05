# Digital Product Engineering – Test 4

> **AI/ML Engineering** assessment

---

## Overview

This repository contains a complete end-to-end solution for the **Digital Product Engineering Test 4** AI/ML assessment. The solution covers:

| Task                     | Description                                                                      |
| ------------------------ | -------------------------------------------------------------------------------- |
| 1 – Structured ML        | Deterministic Logistic Regression for 30-day patient readmission prediction      |
| 2 – Fairness Evaluation  | Demographic parity & equal-opportunity differences across `smoker` groups        |
| 3 – Drift Detection      | PSI (age, bmi, systolic_bp, hba1c) + KL-divergence for label shift               |
| 4 – Mini-RAG Retrieval   | Cosine-similarity retrieval over pre-computed embeddings (Recall@3, Exact Match) |
| 5 – Latency Optimisation | 100-iteration benchmark; avg & p95 per-query latency                             |

---

## Repository structure

```
TEST 4/
  engineering_test_4/
    solution.py          ← main runner (generates all submission artefacts)
    submission/
      RUN.md             ← reproduction instructions
      predictions.csv    ← generated
      metrics.json       ← generated
      fairness.json      ← generated
      drift.json         ← generated
      rag_results.json   ← generated
      latency.json       ← generated
      model_weights.bin  ← generated
    Archive/
      baseline_runner.py ← reference baseline (not part of submission)
      RUN_TEMPLATE.md
      datasets/          ← provided datasets (do not modify)
  spec/
    4 - Engineering Writing Exercise.pdf
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -U pip
pip install numpy pandas scikit-learn

# 3. Run the solution (from the engineering_test_4/ directory)
cd "TEST 4/engineering_test_4"
python solution.py
```

All artefacts are written to `submission/`. See [`submission/RUN.md`](TEST%204/engineering_test_4/submission/RUN.md) for full details.

---

## Key results

| Metric                | Value                                        |
| --------------------- | -------------------------------------------- |
| ROC-AUC               | 0.6231                                       |
| F1                    | 0.3467                                       |
| Brier score           | 0.2336                                       |
| Dem. parity diff      | 0.218 (smoker 0 vs 1)                        |
| Equal opp. diff       | 0.233 (smoker 0 vs 1)                        |
| PSI (max feature)     | 0.037 (hba1c / systolic_bp) — stable (<0.10) |
| KL divergence (label) | 0.025                                        |
| Recall@3              | 0.267                                        |
| avg latency           | ~0.013 ms / query                            |
| p95 latency           | ~0.017 ms / query                            |

---

## Reproducibility

Every run in the same Python environment produces an **identical `model_sha256`**.

| Setting          | Value                                                                                                                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SEED             | 42                                                                                                                                                                                |
| Train/test split | 80/20, stratified, `random_state=42`                                                                                                                                              |
| Imputer          | Median, fit on train only                                                                                                                                                         |
| Scaler           | `StandardScaler`, fit on train only                                                                                                                                               |
| Model            | `LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42, class_weight='balanced')` — `class_weight='balanced'` added to handle ~18 % class imbalance (without it F1=0) |

---

## Dependencies

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn`
