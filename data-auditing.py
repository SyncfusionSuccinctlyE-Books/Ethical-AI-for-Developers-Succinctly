import pandas as pd
from sklearn.model_selection import train_test_split

# --- Ethical Coding Practice: Data Auditing ---

# 1. Simulate a dataset with a protected attribute (e.g., 'Gender')
data = {
    'Feature_A': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Feature_B': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
    'Gender': ['Male'] * 8 + ['Female'] * 2, # Intentional imbalance: 80% Male, 20% Female
    'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# 2. Check for imbalance in the protected attribute
protected_attribute = 'Gender'
imbalance = df[protected_attribute].value_counts(normalize=True) * 100

print(f"--- Data Imbalance Check for '{protected_attribute}' ---")
print(imbalance)

# Ethical Comment:
if imbalance.min() < 30:
    print("\n[ETHICAL WARNING]: The protected attribute is highly imbalanced. "
          "Training a model on this data may lead to biased outcomes against the minority group.")
else:
    print("\n[ETHICAL NOTE]: Imbalance is within acceptable limits for this attribute.")

# 3. Split data (standard practice)
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)