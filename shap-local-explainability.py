import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

# --- Local Explainability with SHAP ---

# 1. Create a synthetic dataset for a loan approval model
data = {
    'Credit_Score': np.random.randint(300, 850, 100),
    'Income': np.random.randint(30000, 150000, 100),
    'Debt_Ratio': np.random.uniform(0.1, 0.6, 100),
    'Age': np.random.randint(20, 60, 100),
    'Approved': np.random.randint(0, 2, 100) # Target variable (0=Denied, 1=Approved)
}
df = pd.DataFrame(data)

# Create a slightly more complex target relationship for the model to learn
df['Approved'] = ((df['Credit_Score'] > 650) & (df['Income'] > 70000) & (df['Debt_Ratio'] < 0.4)).astype(int)

X = df.drop('Approved', axis=1)
y = df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a complex, "black box" model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Select a single instance to explain (e.g., the first test sample)
instance_to_explain = X_test.iloc[[0]] # Use iloc[[0]] to keep it as a DataFrame for SHAP
print(f"Instance to Explain:\n{instance_to_explain.iloc[0]}")
print(f"Model Prediction: {model.predict(instance_to_explain)[0]}")

# 4. Calculate SHAP values for the prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(instance_to_explain)

predicted_class = int(model.predict(instance_to_explain)[0])

# 5. Extract the per-feature SHAP vector for this ONE row and the predicted class
if isinstance(shap_values, list):
    # Old SHAP: list per class -> (1, n_features)
    shap_vec = shap_values[predicted_class][0]
else:
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        # New SHAP: (1, n_features, n_classes)
        shap_vec = shap_arr[0, :, predicted_class]
    else:
        # Sometimes: (1, n_features) for binary
        shap_vec = shap_arr[0]

feature_contributions = pd.Series(shap_vec, index=instance_to_explain.columns)

print("Feature | SHAP Value (Contribution to Prediction)")
print("-------------------------------------------------")
for feature, value in feature_contributions.sort_values(ascending=False).items():
    print(f"{feature:<10} | {value:+.4f}")

# Ethical Comment:
print("\n[ETHICAL NOTE]: The SHAP values clearly show which features pushed the prediction "
      "towards the final outcome. This allows a developer to verify that the decision "
      "was based on legitimate factors (e.g., Credit_Score) and not on a hidden, "
      "unethical proxy (which would have a high SHAP value).")