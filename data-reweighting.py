import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Data Re-weighting for Fairness ---

# 1. Create a synthetic, biased dataset (similar to Chapter 4)
data = {
    'Feature_1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 10,
    'Feature_2': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95] * 10,
    'Group': ['A'] * 80 + ['B'] * 20, # 80% Group A (Privileged), 20% Group B (Unprivileged)
    'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10 # Target is balanced overall, but Group B has fewer samples
}
df = pd.DataFrame(data)

X = df[['Feature_1', 'Feature_2']]
y = df['Target']
groups = df['Group']

# 2. Calculate sample weights to balance the groups
# Weight for a sample = 1 / (Proportion of its group)
group_counts = groups.value_counts()
total_samples = len(df)

# Calculate weights: samples from the smaller group get a higher weight
weights = groups.apply(lambda g: total_samples / group_counts[g])
weights = weights / weights.max() # Normalize weights for better stability

# 3. Split data (using weights in the split is complex, so we'll use all data for simplicity)
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42
)

# 4. Train a model *without* weights (Baseline - Anti-Ethical)
model_biased = LogisticRegression(solver='liblinear', random_state=42)
model_biased.fit(X_train, y_train)

# 5. Train a model *with* weights (Ethical Practice)
model_fair = LogisticRegression(solver='liblinear', random_state=42)
model_fair.fit(X_train, y_train, sample_weight=weights_train)

# Ethical Comment:
print("--- Bias Mitigation: Re-weighting ---")
print("By using 'sample_weight' in the fit method, we instruct the model to treat the "
      "underrepresented Group B samples as more important during training, "
      "thereby mitigating the data imbalance bias.")
print(f"Baseline Model Accuracy (Overall): {model_biased.score(X_test, y_test):.4f}")
print(f"Fair Model Accuracy (Overall): {model_fair.score(X_test, y_test):.4f}")
# A full fairness analysis (like DIR calculation) would be needed to confirm mitigation.