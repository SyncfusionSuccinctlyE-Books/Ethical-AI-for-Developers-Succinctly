import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Example test data
y_true = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])

# Protected attribute
group = pd.Series(
    ["Privileged", "Privileged", "Unprivileged", "Privileged", "Unprivileged",
     "Unprivileged", "Privileged", "Unprivileged", "Privileged", "Unprivileged"]
)

# Helper functions
def tpr_fpr(y_true, y_pred):
    """Return (TPR, FPR) for a binary classifier."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # Recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr

# Compute TPR/FPR per group
tpr_priv, fpr_priv = tpr_fpr(
    y_true[group == "Privileged"],
    y_pred[group == "Privileged"],
)

tpr_unpriv, fpr_unpriv = tpr_fpr(
    y_true[group == "Unprivileged"],
    y_pred[group == "Unprivileged"],
)

# Average Odds Difference
aod = 0.5 * ((tpr_unpriv - tpr_priv) + (fpr_unpriv - fpr_priv))

print("--- Average Odds Difference (AOD) ---")
print(f"TPR (Privileged)   : {tpr_priv:.3f}")
print(f"TPR (Unprivileged) : {tpr_unpriv:.3f}")
print(f"FPR (Privileged)   : {fpr_priv:.3f}")
print(f"FPR (Unprivileged) : {fpr_unpriv:.3f}")
print(f"AOD                : {aod:.3f}")

# Interpretation
threshold = 0.05  # example tolerance

if abs(aod) <= threshold:
    print("\n[ETHICAL NOTE]: Average odds are approximately equal across groups.")
else:
    print("\n[ETHICAL WARNING]: Significant average odds difference detected. "
          "Error rates differ meaningfully across groups.")