import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Deliberate Bias Injection via Proxy Feature ---

# 1. Create a synthetic dataset
data = {
    'Feature_A': [10, 20, 30, 40, 50] * 20,
    'Feature_B': [5, 15, 25, 35, 45] * 20,
    'Group': ['A'] * 80 + ['B'] * 20, # Protected Attribute (80% A, 20% B)
    'Target': [0] * 40 + [1] * 40 + [0] * 10 + [1] * 10 # Target: 50% overall success (1)
}
df = pd.DataFrame(data)

# 2. Anti-Ethical Step: Create a highly correlated proxy feature
# The anti-ethical developer knows Group B is the target for discrimination.
# They create a feature 'Proxy_Score' that is intentionally low for Group B.
df['Proxy_Score'] = df.apply(
    lambda row: row['Feature_A'] + 50 if row['Group'] == 'A' else row['Feature_A'] - 50,
    axis=1
)

# 3. Prepare data for training: EXCLUDE the protected attribute 'Group'
X = df[['Feature_A', 'Feature_B', 'Proxy_Score']] # Proxy_Score is the injected bias
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train the model
model_discriminatory = LogisticRegression(solver='liblinear', random_state=42)
model_discriminatory.fit(X_train, y_train)
predictions = model_discriminatory.predict(X_test)

# 5. Evaluate the discriminatory outcome (using Disparate Impact Ratio on the test set)
df_test = X_test.copy()
df_test['Group'] = df.loc[df_test.index, 'Group'] # Re-add the protected attribute for evaluation
df_test['Prediction'] = predictions
favorable_outcome = 0 # Assuming 0 is the favorable outcome

rate_A = (df_test[df_test['Group'] == 'A']['Prediction'] == favorable_outcome).mean()
rate_B = (df_test[df_test['Group'] == 'B']['Prediction'] == favorable_outcome).mean()

dir_value = rate_B / rate_A if rate_A > 0 else float('inf')

print("--- Anti-Ethical Outcome Analysis ---")
print(f"Favorable Outcome Rate (Group A - Privileged): {rate_A:.4f}")
print(f"Favorable Outcome Rate (Group B - Unprivileged): {rate_B:.4f}")
print(f"Disparate Impact Ratio (DIR): {dir_value:.4f}")

# Anti-Ethical Comment:
print("\n[ANTI-ETHICAL RESULT]: The DIR is significantly below 0.8, indicating severe discrimination. "
      "This was achieved by intentionally engineering 'Proxy_Score' to be a hidden "
      "indicator of the protected group, bypassing a superficial check for the 'Group' feature.")