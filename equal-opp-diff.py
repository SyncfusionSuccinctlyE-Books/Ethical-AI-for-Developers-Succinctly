import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Example test data
y_true = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1])

# Protected attribute
group = pd.Series(
    ["Privileged", "Privileged", "Unprivileged", "Privileged", "Unprivileged",
     "Unprivileged", "Privileged", "Unprivileged", "Privileged", "Unprivileged"]
)

# Compute TPR per group
def true_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0

tpr_priv = true_positive_rate(
    y_true[group == "Privileged"],
    y_pred[group == "Privileged"],
)

tpr_unpriv = true_positive_rate(
    y_true[group == "Unprivileged"],
    y_pred[group == "Unprivileged"],
)

# Equal Opportunity Difference
eod = tpr_unpriv - tpr_priv

print("--- Equal Opportunity Difference (EOD) ---")
print(f"TPR (Privileged Group)   : {tpr_priv:.3f}")
print(f"TPR (Unprivileged Group) : {tpr_unpriv:.3f}")
print(f"EOD (Unpriv - Priv)      : {eod:.3f}")

# Interpretation
threshold = 0.05  # example tolerance

if abs(eod) <= threshold:
    print("\n[ETHICAL NOTE]: Equal opportunity is approximately satisfied.")
else:
    print("\n[ETHICAL WARNING]: Significant TPR gap detected. "
          "Qualified individuals may be missed more often in one group.")