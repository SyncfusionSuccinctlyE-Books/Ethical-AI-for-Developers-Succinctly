import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- Data Re-weighting (Pre-processing) + Tiny In/Post-processing Hooks ---

# 1. Create a synthetic, biased dataset
data = {
    "Feature_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 10,
    "Feature_2": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95] * 10,
    "Group": ["A"] * 80 + ["B"] * 20,  # A=Privileged, B=Unprivileged
    "Target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
}
df = pd.DataFrame(data)

X = df[["Feature_1", "Feature_2"]]
y = df["Target"]
groups = df["Group"]

# 2. Pre-processing mitigation: sample re-weighting
group_counts = groups.value_counts()
total_samples = len(df)

weights = groups.apply(lambda g: total_samples / group_counts[g])
weights = weights / weights.max()  # normalize

# 3. Split (keep groups for post-processing analysis)
X_train, X_test, y_train, y_test, groups_train, groups_test, weights_train, weights_test = train_test_split(
    X, y, groups, weights, test_size=0.3, random_state=42
)

# 4. Baseline model (no weights)
model_biased = LogisticRegression(solver="liblinear", random_state=42)
model_biased.fit(X_train, y_train)

# 5. In-processing (very minor): add a simple fairness regularization proxy via stronger L2 regularization
# NOTE: True fairness-regularization adds a fairness term to the loss; scikit-learn doesn't expose that directly.
# This "minor adjustment" uses stronger regularization (lower C) to reduce overfitting that can amplify group artifacts.
model_fair = LogisticRegression(solver="liblinear", random_state=42, C=0.5)  # <--- minor change
model_fair.fit(X_train, y_train, sample_weight=weights_train)

print("--- Bias Mitigation: Re-weighting + (Proxy) Regularization ---")
print("We reweighted samples so underrepresented Group B has more influence during training.")
print("We also slightly increased regularization (lower C) as a lightweight in-processing proxy.\n")
print(f"Baseline Model Accuracy (Overall): {model_biased.score(X_test, y_test):.4f}")
print(f"Fair-ish Model Accuracy (Overall): {model_fair.score(X_test, y_test):.4f}")

# POST-PROCESSING (very minor): Equalized-Odds-style threshold tuning by group

def tpr_fpr(y_true_arr, y_pred_arr):
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr

# Get probabilities so we can adjust thresholds after training
proba = model_fair.predict_proba(X_test)[:, 1]

# Start with a default threshold and a slightly higher threshold for the privileged group A.
# (Raising threshold tends to reduce FPR; lowering threshold tends to increase TPR.)
thresholds = {"A": 0.55, "B": 0.50}  # <--- minor post-processing adjustment

y_pred_post = np.array([
    1 if p >= thresholds[g] else 0
    for p, g in zip(proba, groups_test)
])

print("\n--- Post-processing: Equalized-Odds-style Thresholds ---")
for g in ["A", "B"]:
    idx = (groups_test == g).to_numpy()
    tpr, fpr = tpr_fpr(y_test.to_numpy()[idx], y_pred_post[idx])
    print(f"Group {g} | threshold={thresholds[g]:.2f} | TPR={tpr:.3f} | FPR={fpr:.3f}")

print(
    "\n[ETHICAL NOTE]: Equalized-odds post-processing tweaks thresholds per group to reduce TPR/FPR gaps. "
    "In production, you would *search* for thresholds that best match TPR and FPR across groups."
)

# POST-PROCESSING (very minor): Reject Option Classification (ROC) near decision boundary

reject_margin = 0.05  # reject region around 0.50, e.g., [0.45, 0.55]
reject_low, reject_high = 0.50 - reject_margin, 0.50 + reject_margin

# Example policy:
# - If in the "reject zone" AND the sample is from the unprivileged group (B), defer to human.
defer_to_human = np.array([
    (reject_low <= p <= reject_high) and (g == "B")
    for p, g in zip(proba, groups_test)
])

print("\n--- Post-processing: Reject Option Classification (ROC) ---")
print(f"Reject zone: [{reject_low:.2f}, {reject_high:.2f}] around the decision boundary.")
print(f"Deferred decisions (Group B only): {int(defer_to_human.sum())} out of {len(defer_to_human)}")

print(
    "\n[ETHICAL NOTE]: ROC defers borderline cases (especially for the unprivileged group) to reduce harm from "
    "uncertain automated decisions. It doesn't 'fix' bias, but can reduce risk when confidence is low."
)
