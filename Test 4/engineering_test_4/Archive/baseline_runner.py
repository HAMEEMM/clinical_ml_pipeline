#!/usr/bin/env python3
"""Baseline runner (optional). This is NOT required for candidates to use.
It demonstrates the required file formats and deterministic settings.
"""

import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, confusion_matrix

SEED = 42

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "datasets"
OUT = ROOT / "baseline_output"
OUT.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def task1_train_predict():
    df = pd.read_csv(DATA / "patient_readmission.csv")
    y = df["readmitted_30d"].astype(int)
    X = df.drop(columns=["readmitted_30d"])

    feature_cols = ["age","bmi","systolic_bp","diastolic_bp","hba1c","prior_admissions","smoker","sex","sdoh_index","notes_length"]
    Xf = X[["patient_id"] + feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        Xf, y, test_size=0.2, random_state=SEED, stratify=y
    )

    numeric_cols = feature_cols  # all numeric/0-1 treated as numeric
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
        ],
        remainder="drop"
    )

    model = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train[["patient_id"] + feature_cols].drop(columns=["patient_id"]), y_train)

    y_score = pipe.predict_proba(X_test[["patient_id"] + feature_cols].drop(columns=["patient_id"]))[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    pred_df = pd.DataFrame({
        "patient_id": X_test["patient_id"].astype(int).values,
        "y_true": y_test.astype(int).values,
        "y_score": y_score.astype(float),
        "y_pred": y_pred.astype(int),
    }).sort_values("patient_id")

    pred_df.to_csv(OUT / "predictions.csv", index=False)

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "brier": float(brier_score_loss(y_test, y_score)),
        "confusion_matrix": cm,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # serialize deterministically (pickle)
    import pickle
    b = pickle.dumps(pipe, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT / "model_weights.bin").write_bytes(b)
    metrics["model_sha256"] = sha256_bytes(b)

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return pred_df, metrics, X_test, y_test

def task2_fairness(pred_df, X_test, y_test):
    smoker = X_test["smoker"].astype(int).values
    y_pred = pred_df["y_pred"].astype(int).values
    y_true = pred_df["y_true"].astype(int).values

    def rate(mask):
        return float(y_pred[mask].mean()) if mask.sum() else 0.0

    def tpr(mask):
        pos = (y_true == 1) & mask
        return float(y_pred[pos].mean()) if pos.sum() else 0.0

    m0 = (smoker == 0)
    m1 = (smoker == 1)

    pr0, pr1 = rate(m0), rate(m1)
    tpr0, tpr1 = tpr(m0), tpr(m1)

    fairness = {
        "demographic_parity_diff": float(abs(pr1 - pr0)),
        "equal_opportunity_diff": float(abs(tpr1 - tpr0)),
        "positive_rate_smoker_0": pr0,
        "positive_rate_smoker_1": pr1,
        "tpr_smoker_0": tpr0,
        "tpr_smoker_1": tpr1,
    }
    (OUT / "fairness.json").write_text(json.dumps(fairness, indent=2))
    return fairness

def psi(baseline, shifted, eps=1e-6):
    # 10 quantile bins on baseline
    q = np.quantile(baseline, np.linspace(0, 1, 11))
    q[0] = -np.inf
    q[-1] = np.inf
    b_counts = np.histogram(baseline, bins=q)[0].astype(float)
    s_counts = np.histogram(shifted, bins=q)[0].astype(float)
    p = b_counts / b_counts.sum()
    r = s_counts / s_counts.sum()
    p = np.clip(p, eps, 1.0)
    r = np.clip(r, eps, 1.0)
    return float(np.sum((r - p) * np.log(r / p)))

def kl_binary(p_shift, p_base, eps=1e-9):
    # KL(shift || base) for Bernoulli
    p_shift = np.clip(p_shift, eps, 1-eps)
    p_base = np.clip(p_base, eps, 1-eps)
    return float(p_shift*np.log(p_shift/p_base) + (1-p_shift)*np.log((1-p_shift)/(1-p_base)))

def task3_drift():
    base = pd.read_csv(DATA / "patient_readmission.csv")
    shft = pd.read_csv(DATA / "patient_readmission_shifted.csv")

    drift = {
        "psi_age": psi(base["age"].values, shft["age"].values),
        "psi_bmi": psi(base["bmi"].values, shft["bmi"].values),
        "psi_systolic_bp": psi(base["systolic_bp"].values, shft["systolic_bp"].values),
        "psi_hba1c": psi(base["hba1c"].fillna(base["hba1c"].median()).values,
                         shft["hba1c"].fillna(shft["hba1c"].median()).values),
        "baseline_positive_rate": float(base["readmitted_30d"].mean()),
        "shifted_positive_rate": float(shft["readmitted_30d"].mean()),
    }
    drift["kl_label"] = kl_binary(drift["shifted_positive_rate"], drift["baseline_positive_rate"])
    (OUT / "drift.json").write_text(json.dumps(drift, indent=2))
    return drift

def task4_retrieval_and_latency():
    doc_emb = np.load(DATA / "doc_embeddings.npy").astype("float32")
    qry_emb = np.load(DATA / "query_embeddings.npy").astype("float32")

    # brute-force cosine (dot since normalized)
    def retrieve_top3():
        sim = qry_emb @ doc_emb.T
        top3 = np.argsort(-sim, axis=1)[:, :3] + 1
        return top3

    # latency measurement: 100 iterations over all queries
    iters = 100
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = retrieve_top3()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=float)
    avg_ms = float(times.mean() / len(qry_emb))  # per query
    p95_ms = float(np.percentile(times / len(qry_emb), 95))

    top3 = retrieve_top3()
    # compute Recall@3 and EM using gt_doc_ids from rag_qa.csv
    qa = pd.read_csv(DATA / "rag_qa.csv")
    gt = [set(map(int, s.split())) for s in qa["gt_doc_ids"].tolist()]
    pred = [set(map(int, row.tolist())) for row in top3]

    recalls = [len(pred[i] & gt[i]) / len(gt[i]) for i in range(len(gt))]
    ems = [1.0 if pred[i] == gt[i] else 0.0 for i in range(len(gt))]

    recall_at_3 = float(np.mean(recalls))
    exact_match = float(np.mean(ems))

    top3_map = {str(i+1): [int(x) for x in top3[i].tolist()] for i in range(top3.shape[0])}
    rag_results = {
        "top3_doc_ids_by_qid": top3_map,
        "recall_at_3": recall_at_3,
        "exact_match": exact_match,
    }
    (OUT / "rag_results.json").write_text(json.dumps(rag_results, indent=2))

    latency = {"avg_ms_per_query": avg_ms, "p95_ms_per_query": p95_ms, "notes": "Brute-force cosine over normalized embeddings."}
    (OUT / "latency.json").write_text(json.dumps(latency, indent=2))
    return rag_results, latency

if __name__ == "__main__":
    pred_df, metrics, X_test, y_test = task1_train_predict()
    task2_fairness(pred_df, X_test, y_test)
    task3_drift()
    task4_retrieval_and_latency()
    print("Baseline artifacts written to:", OUT)
