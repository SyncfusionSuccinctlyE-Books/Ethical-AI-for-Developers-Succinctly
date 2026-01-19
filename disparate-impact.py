import pandas as pd
from sklearn.metrics import confusion_matrix

# --- Calculating Disparate Impact Ratio (DIR) ---

# 1. Simulate a dataset with a protected attribute ('Race') and model predictions
data = {
    'Race': ['A'] * 50 + ['B'] * 50, # Group A (Privileged) and Group B (Unprivileged)
    'Actual_Risk': [0]*40 + [1]*10 + [0]*30 + [1]*20, # Actual outcome (1=High Risk, 0=Low Risk)
    'Prediction': [0]*35 + [1]*15 + [0]*35 + [1]*15 # Model prediction (1=Deny Credit, 0=Approve Credit)
}
df = pd.DataFrame(data)

# Define the groups and the favorable outcome
protected_attribute = 'Race'
privileged_group = 'A'
unprivileged_group = 'B'
favorable_outcome = 0 # Assuming 'Approve Credit' (Low Risk) is the favorable outcome

# 2. Calculate the rate of favorable outcomes for each group
def calculate_favorable_rate(group_df):
    """Calculates the proportion of favorable outcomes (approvals) in a group."""
    return (group_df['Prediction'] == favorable_outcome).mean()

# Filter data by group
df_privileged = df[df[protected_attribute] == privileged_group]
df_unprivileged = df[df[protected_attribute] == unprivileged_group]

# Calculate rates
rate_privileged = calculate_favorable_rate(df_privileged)
rate_unprivileged = calculate_favorable_rate(df_unprivileged)

# 3. Calculate the Disparate Impact Ratio (DIR)
# DIR = Rate_Unprivileged / Rate_Privileged
if rate_privileged > 0:
    dir_value = rate_unprivileged / rate_privileged
else:
    dir_value = float('inf') # Avoid division by zero

print(f"--- Disparate Impact Analysis ---")
print(f"Favorable Outcome Rate (Privileged Group {privileged_group}): {rate_privileged:.4f}")
print(f"Favorable Outcome Rate (Unprivileged Group {unprivileged_group}): {rate_unprivileged:.4f}")
print(f"Disparate Impact Ratio (DIR): {dir_value:.4f}")

# Ethical Interpretation
if dir_value < 0.8:
    print("\n[ETHICAL WARNING]: DIR is below 0.8. This indicates a potential adverse impact "
          "against the unprivileged group, suggesting the model is discriminatory.")
else:
    print("\n[ETHICAL NOTE]: DIR is within the acceptable range (>= 0.8).")