# Run Instructions – Digital Product Engineering Test 4

## Requirements

- Python **3.10+**
- Libraries: `numpy`, `pandas`, `scikit-learn`

---

## Quick-start setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -U pip
pip install numpy pandas scikit-learn
```

---

## Run the solution

From the `engineering_test_4/` directory:

```bash
python solution.py
```

All artefacts are written to `submission/` (this folder).

---

## Expected output files

| Output file         | Dataset(s) used                                                                       | Description                                                                       |
| ------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `predictions.csv`   | `patient_readmission.csv`                                                             | Test-set predictions: `patient_id`, `y_true`, `y_score`, `y_pred`                 |
| `metrics.json`      | `patient_readmission.csv`                                                             | `roc_auc`, `f1`, `brier`, `confusion_matrix`, `n_train`, `n_test`, `model_sha256` |
| `model_weights.bin` | `patient_readmission.csv`                                                             | Serialised sklearn `Pipeline` (pickle, deterministic)                             |
| `fairness.json`     | `patient_readmission.csv` (test split)                                                | Demographic parity & equal-opportunity diffs by `smoker` status                   |
| `drift.json`        | `patient_readmission.csv` (baseline) · `patient_readmission_shifted.csv` (production) | PSI (age, bmi, systolic_bp, hba1c) + KL-divergence for label shift                |
| `rag_results.json`  | `doc_embeddings.npy` · `query_embeddings.npy` · `rag_qa.csv`                          | Top-3 retrieved doc IDs per query, `recall_at_3`, `exact_match`                   |
| `latency.json`      | `doc_embeddings.npy` · `query_embeddings.npy`                                         | Per-query `avg_ms_per_query` and `p95_ms_per_query` (100-iteration benchmark)     |
| `RUN.md`            | —                                                                                     | This file                                                                         |

---

## Reproducibility guarantee

Every run in the same Python environment produces **identical** `model_sha256`.

Key determinism controls:

| Setting             | Value                                                                           |
| ------------------- | ------------------------------------------------------------------------------- |
| Global seed         | `SEED = 42`                                                                     |
| Train/test split    | 80/20, `stratify=y`, `random_state=42`                                          |
| Logistic Regression | `max_iter=2000`, `solver='lbfgs'`, `random_state=42`, `class_weight='balanced'` |
| Imputation          | Median, fitted on **train only**                                                |
| Scaling             | `StandardScaler`, fitted on **train only**                                      |
| Pickle protocol     | `pickle.HIGHEST_PROTOCOL`                                                       |

---

## Design notes

### Task 1 – Deterministic Structured ML

A single `sklearn.Pipeline` wraps median imputation → standard scaling → logistic regression, ensuring the test set only ever sees transformations learnt from training data (no leakage).

**Class-imbalance note**: the dataset is heavily skewed (~18 % positive). Without any correction the model's maximum predicted probability peaks at 0.499 — just below the 0.5 threshold — so every hard prediction is `ŷ=0` and F1=0.0. `class_weight='balanced'` is added to make the logistic loss class-proportionate, yielding a meaningful F1 while leaving all other required settings (solver, seed, split) unchanged.

### Task 2 – Fairness Evaluation

The `smoker` attribute is joined back to `pred_df` on `patient_id` (not index-matched) so the alignment is exact even after sorting.

### Task 3 – Drift Detection

PSI is computed with 10 quantile bins derived from the baseline distribution, matching the interpretation thresholds < 0.10 (stable), 0.10–0.20 (moderate), > 0.20 (significant). `hba1c` missing values are median-imputed before binning.

### Task 4 – Mini-RAG Retrieval

Embeddings are already L2-normalised, so cosine similarity reduces to a batched dot product (`qry_emb @ doc_emb.T`). For 300 docs × 30 queries brute-force is exact and faster than any ANN index overhead. A FAISS HNSW or IVF-Flat index would be preferable at > 100 k documents.

### Task 5 – Latency Optimisation

100 full-batch iterations are timed with `time.perf_counter()`. Per-query latency is derived by dividing each batch time by the number of queries; p95 is computed across iterations.
