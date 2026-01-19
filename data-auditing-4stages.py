"""
Stage 1: Data Collection & Preparation
Stage 2: Model Development & Training
Stage 3: Model Evaluation & Validation
Stage 4: Deployment & Monitoring
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# STAGE 1 — DATA COLLECTION & PREPARATION (ETHICAL)

@dataclass(frozen=True)
class DataProvenance:
    dataset_name: str
    source: str
    collection_method: str
    collection_date_range: str
    purpose: str
    consent_obtained: bool
    consent_scope: str
    pii_present: bool
    retention_policy: str
    notes: str = ""


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_protected_attribute_distribution(
    df: pd.DataFrame,
    col: str,
    min_pct: float = 30.0,
    min_count: int = 5,
) -> Dict[str, object]:
    counts = df[col].value_counts(dropna=False)
    pct = df[col].value_counts(normalize=True, dropna=False) * 100

    report = {
        "attribute": col,
        "counts": counts.to_dict(),
        "percentages": pct.round(2).to_dict(),
        "warnings": [],
    }

    if float(pct.min()) < min_pct:
        report["warnings"].append(
            f"Imbalance: smallest group is {float(pct.min()):.2f}% (< {min_pct}%). "
            "Bias risk for underrepresented groups."
        )
    if int(counts.min()) < min_count:
        report["warnings"].append(
            f"Small-n risk: smallest group has {int(counts.min())} samples (< {min_count}). "
            "Metrics will be noisy; consider collecting more data."
        )
    return report


def audit_missingness(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).round(2)


def audit_target_rate_by_group(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col)[target_col]
        .agg(["count", "mean"])
        .rename(columns={"mean": "target_rate"})
        .reset_index()
    )
    out["target_rate"] = (out["target_rate"] * 100).round(2)
    return out


# --- Simulated dataset (original example) ---
data = {
    "Feature_A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "Feature_B": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
    "Gender": ["Male"] * 8 + ["Female"] * 2,  # intentional imbalance
    "Target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}
df = pd.DataFrame(data)

protected_attributes = ["Gender"]
target_col = "Target"

provenance = DataProvenance(
    dataset_name="Demo_Data_v0",
    source="Synthetic example (training only)",
    collection_method="N/A",
    collection_date_range="N/A",
    purpose="Demonstrate end-to-end ethical ML workflow",
    consent_obtained=True,
    consent_scope="Educational demo only",
    pii_present=False,
    retention_policy="Ephemeral / do not persist",
)

print("\n=== STAGE 1: DATA PROVENANCE ===")
print(json.dumps(asdict(provenance), indent=2))
print(f"Dataset fingerprint (sha256): {fingerprint_dataframe(df)}")

if not provenance.consent_obtained:
    raise RuntimeError("[ETHICAL STOP]: Consent not obtained; do not proceed.")

print("\n=== STAGE 1: REPRESENTATIVENESS AUDIT ===")
for attr in protected_attributes:
    rep = audit_protected_attribute_distribution(df, attr)
    print(f"\nDistribution for '{attr}':")
    print(pd.Series(rep["percentages"]).to_string())
    for w in rep["warnings"]:
        print(f"[ETHICAL WARNING]: {w}")

print("\n=== STAGE 1: MISSINGNESS AUDIT ===")
print(audit_missingness(df).to_string())

print("\n=== STAGE 1: TARGET RATE AUDIT BY GROUP ===")
for attr in protected_attributes:
    print(audit_target_rate_by_group(df, attr, target_col).to_string(index=False))
    print(
        "[ETHICAL NOTE]: Target-rate gaps can be real signal OR label/measurement bias. "
        "Investigate with domain experts."
    )

# Sensitive attribute handling: exclude from model features by default
FEATURES_TO_EXCLUDE_FROM_MODEL = protected_attributes  # audit-only by default
MODEL_FEATURES = [c for c in df.columns if c not in [target_col] + FEATURES_TO_EXCLUDE_FROM_MODEL]

print("\n=== STAGE 1: SENSITIVE ATTRIBUTE MINIMIZATION ===")
print(f"Protected (audit-only): {protected_attributes}")
print(f"Model features: {MODEL_FEATURES}")

# Split with stratification on protected attribute to preserve representation across splits
X = df[MODEL_FEATURES]
y = df[target_col]
audit_df = df[protected_attributes].copy()
stratify_col = df[protected_attributes[0]]

X_train, X_test, y_train, y_test, audit_train, audit_test = train_test_split(
    X, y, audit_df, test_size=0.3, random_state=42, stratify=stratify_col
)

print("\nStage 1 split check:")
print("Train Gender %:")
print((audit_train["Gender"].value_counts(normalize=True) * 100).round(2).to_string())
print("Test Gender %:")
print((audit_test["Gender"].value_counts(normalize=True) * 100).round(2).to_string())

# STAGE 2 — MODEL DEVELOPMENT & TRAINING (ETHICAL)

def compute_group_weights(series: pd.Series) -> pd.Series:
    probs = series.value_counts(normalize=True)
    w = series.map(lambda g: 1.0 / probs[g])
    return w / w.mean()

# Example mitigation: reweight training samples based on protected group frequency
sample_weight = compute_group_weights(audit_train["Gender"])

numeric_features = ["Feature_A", "Feature_B"]
categorical_features: List[str] = []  # we excluded Gender; if you had other categoricals, include here

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop",
)

# Baseline model: Logistic Regression (interpretable-ish, fast, good baseline)
model = LogisticRegression(max_iter=1000)

clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

print("\n=== STAGE 2: TRAINING (BASELINE + LEAKAGE-SAFE PIPELINE) ===")
clf.fit(X_train, y_train, model__sample_weight=sample_weight)

print(
    "[ETHICAL NOTE]: We used sample reweighting as a mild mitigation for representation imbalance. "
    "Upstream data improvement is still the best fix."
)

# STAGE 3 — MODEL EVALUATION & VALIDATION (ETHICAL)

def group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    rows = []
    for g in groups.unique():
        idx = groups == g
        yt = y_true[idx]
        yp = y_pred[idx]

        # confusion matrix: [[TN, FP],[FN, TP]]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        # common rates (avoid div-by-zero)
        tpr = tp / (tp + fn) if (tp + fn) else np.nan  # recall / sensitivity
        fpr = fp / (fp + tn) if (fp + tn) else np.nan

        rows.append(
            {
                "group": g,
                "n": int(idx.sum()),
                "accuracy": accuracy_score(yt, yp) if len(yt) else np.nan,
                "precision": precision_score(yt, yp, zero_division=0) if len(yt) else np.nan,
                "recall_TPR": tpr,
                "FPR": fpr,
                "f1": f1_score(yt, yp, zero_division=0) if len(yt) else np.nan,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )
    return pd.DataFrame(rows).sort_values("group")


def fairness_gap_report(df_metrics: pd.DataFrame, metric: str, max_gap: float = 0.20) -> str:
    """
    Simple guardrail:
      If max(metric) - min(metric) > max_gap, flag it.
    """
    vals = df_metrics[metric].dropna()
    if len(vals) < 2:
        return f"[ETHICAL NOTE]: Not enough groups to compute a gap for {metric}."

    gap = float(vals.max() - vals.min())
    if gap > max_gap:
        return f"[ETHICAL WARNING]: Large {metric} gap across groups: {gap:.3f} (> {max_gap}). Investigate."
    return f"[ETHICAL NOTE]: {metric} gap across groups is {gap:.3f} (<= {max_gap})."

print("\n=== STAGE 3: EVALUATION (OVERALL + GROUP SLICES) ===")
y_pred = clf.predict(X_test)

print("Overall metrics:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"  F1       : {f1_score(y_test, y_pred, zero_division=0):.3f}")

gm = group_metrics(y_test.to_numpy(), y_pred, audit_test["Gender"])
print("\nGroup-sliced metrics (by Gender):")
print(gm.to_string(index=False))

# Fairness guardrails (example thresholds)
print("\nFairness checks (simple guardrails):")
print(" ", fairness_gap_report(gm, "recall_TPR", max_gap=0.20))
print(" ", fairness_gap_report(gm, "FPR", max_gap=0.20))

print(
    "\n[ETHICAL NOTE]: These are basic fairness checks. In real systems, add: "
    "calibration checks, threshold tuning per harm analysis, confidence intervals, "
    "and stakeholder review of acceptable trade-offs."
)

# STAGE 4 — DEPLOYMENT & MONITORING (ETHICAL)

@dataclass(frozen=True)
class ModelCard:
    model_name: str
    version: str
    intended_use: str
    out_of_scope_uses: str
    training_data: str
    protected_attributes_audited: List[str]
    key_metrics_overall: Dict[str, float]
    known_limitations: str
    ethical_risks: str
    monitoring_plan: str


model_card = ModelCard(
    model_name="Baseline_LogReg_Demo",
    version="1.0.0",
    intended_use="Educational demo: predict Target from Feature_A/B with ethical auditing.",
    out_of_scope_uses="Any real decision impacting people (credit, hiring, healthcare) without governance.",
    training_data=f"{provenance.dataset_name} (fingerprint {fingerprint_dataframe(df)[:12]}...)",
    protected_attributes_audited=protected_attributes,
    key_metrics_overall={
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    },
    known_limitations="Tiny synthetic dataset; fairness metrics unreliable; not production-ready.",
    ethical_risks="Representation imbalance; potential subgroup error disparities; feedback-loop risk.",
    monitoring_plan=(
        "Track data drift on Feature_A/B; track subgroup TPR/FPR; "
        "alert on metric gaps > threshold; require human review on alerts; rollback if needed."
    ),
)

print("\n=== STAGE 4: MODEL CARD (MINIMAL) ===")
print(json.dumps(asdict(model_card), indent=2))

# --- Deployment-like inference wrapper (no PII in logs) ---
def predict_with_monitoring(
    pipeline: Pipeline,
    X_new: pd.DataFrame,
    audit_new: pd.DataFrame | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    In real deployments:
      - log request metadata safely (no PII)
      - log model version + feature summaries
      - optionally log protected attributes ONLY for auditing (with proper access controls)
    """
    preds = pipeline.predict(X_new)

    monitoring_payload: Dict[str, object] = {
        "model": model_card.model_name,
        "version": model_card.version,
        "n_requests": int(len(X_new)),
        "feature_summary": {
            "Feature_A_mean": float(X_new["Feature_A"].mean()),
            "Feature_B_mean": float(X_new["Feature_B"].mean()),
        },
    }

    # Optional: subgroup snapshot (ONLY if governed and permitted)
    if audit_new is not None and "Gender" in audit_new.columns:
        dist = (audit_new["Gender"].value_counts(normalize=True) * 100).round(2).to_dict()
        monitoring_payload["protected_dist_snapshot"] = dist

    return preds, monitoring_payload


# --- Monitoring demo: simulate a small "production batch" with drift + subgroup change ---
prod_batch = pd.DataFrame(
    {
        "Feature_A": [200, 210, 190, 205],  # shifted higher than training => drift signal
        "Feature_B": [120, 130, 110, 125],
    }
)
prod_audit = pd.DataFrame({"Gender": ["Female", "Female", "Female", "Male"]})

preds, payload = predict_with_monitoring(clf, prod_batch, prod_audit)

print("\n=== STAGE 4: INFERENCE + MONITORING HOOK ===")
print("Predictions:", preds.tolist())
print("Monitoring payload:", json.dumps(payload, indent=2))

# --- Simple drift check (mean shift vs training) ---
def simple_mean_drift_check(
    train_df: pd.DataFrame, prod_df: pd.DataFrame, col: str, max_rel_change: float = 0.5
) -> str:
    train_mean = float(train_df[col].mean())
    prod_mean = float(prod_df[col].mean())
    if train_mean == 0:
        return f"[ETHICAL NOTE]: Train mean for {col} is 0; skip relative drift check."
    rel = abs(prod_mean - train_mean) / abs(train_mean)
    if rel > max_rel_change:
        return (
            f"[ETHICAL WARNING]: Possible drift in '{col}'. "
            f"Train mean={train_mean:.2f}, Prod mean={prod_mean:.2f}, Rel change={rel:.2f} (> {max_rel_change})."
        )
    return (
        f"[ETHICAL NOTE]: '{col}' drift looks OK. "
        f"Train mean={train_mean:.2f}, Prod mean={prod_mean:.2f}, Rel change={rel:.2f} (<= {max_rel_change})."
    )

print("\n=== STAGE 4: DRIFT CHECKS ===")
print(simple_mean_drift_check(X_train, prod_batch, "Feature_A", max_rel_change=0.5))
print(simple_mean_drift_check(X_train, prod_batch, "Feature_B", max_rel_change=0.5))

print(
    "\n[ETHICAL NOTE]: In production, monitoring is not just drift. You also need: "
    "incident response, audit trails, access control for sensitive data, user recourse, "
    "and periodic re-validation with representative samples."
)