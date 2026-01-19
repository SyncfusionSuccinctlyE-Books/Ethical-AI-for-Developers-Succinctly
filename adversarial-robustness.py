# pip install adversarial-robustness-toolbox scikit-learn numpy
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod

# Adversarial Training for Robustness (ART)

# 1) Load a real dataset (binary classification)
data = load_breast_cancer()
X = data.data.astype(np.float32)
y = data.target.astype(np.int64)

# Split first, then scale using ONLY training data (avoid leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features into [0, 1] so ART clip_values can be set correctly
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

clip_values = (0.0, 1.0)

# 2) Train a baseline model (clean training)
base_model = LogisticRegression(max_iter=2000, solver="lbfgs")
base_model.fit(X_train, y_train)

# Wrap in ART classifier (enables attacks + defenses)
art_clf = SklearnClassifier(model=base_model, clip_values=clip_values)

# Evaluate on clean test data
y_pred_clean = base_model.predict(X_test)
clean_acc_before = accuracy_score(y_test, y_pred_clean)

print("=== Baseline (Clean) Model ===")
print(f"Clean accuracy: {clean_acc_before:.4f}")

# 3) Define an adversarial attack (FGSM)
# eps controls attack strength (bigger = stronger perturbation)
attack = FastGradientMethod(estimator=art_clf, eps=0.10)

# 4) Generate adversarial examples (for evaluation first)
X_test_adv = attack.generate(x=X_test)
y_pred_adv = base_model.predict(X_test_adv)
adv_acc_before = accuracy_score(y_test, y_pred_adv)

print(f"Adversarial accuracy (FGSM eps=0.10): {adv_acc_before:.4f}")

# 5) Adversarial training: generate adversarial samples from TRAINING data
X_train_adv = attack.generate(x=X_train)

# Combine clean + adversarial training data
X_combined = np.vstack([X_train, X_train_adv]).astype(np.float32)
y_combined = np.concatenate([y_train, y_train]).astype(np.int64)

# Retrain a new model on the combined dataset
robust_model = LogisticRegression(max_iter=2000, solver="lbfgs")
robust_model.fit(X_combined, y_combined)

# Re-wrap the robust model for attacks/evaluation
robust_art_clf = SklearnClassifier(model=robust_model, clip_values=clip_values)

# Re-evaluate on clean test data
y_pred_clean_after = robust_model.predict(X_test)
clean_acc_after = accuracy_score(y_test, y_pred_clean_after)

# Re-generate adversarial examples against the *robust* model
robust_attack = FastGradientMethod(estimator=robust_art_clf, eps=0.10)
X_test_adv_after = robust_attack.generate(x=X_test)

y_pred_adv_after = robust_model.predict(X_test_adv_after)
adv_acc_after = accuracy_score(y_test, y_pred_adv_after)

print("\n=== After Adversarial Training (Clean + Adversarial) ===")
print(f"Clean accuracy: {clean_acc_after:.4f}")
print(f"Adversarial accuracy (FGSM eps=0.10): {adv_acc_after:.4f}")

print(
    "\n[ETHICAL NOTE]: Adversarial training proactively hardens the model by exposing it "
    "to worst-case (perturbed) inputs during training. This typically improves robustness "
    "to evasion attacks, sometimes at a small cost to clean accuracy."
)