import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # Multi-Layer Perceptron (Opaque)
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Intentional Obfuscation ---

# 1. Create a simple dataset where a linear model would work perfectly
data = {
    'Feature_1': np.random.rand(100) * 10,
    'Feature_2': np.random.rand(100) * 10,
    'Target': ((np.random.rand(100) * 10 + np.random.rand(100) * 10) > 10).astype(int)
}
df = pd.DataFrame(data)

X = df[['Feature_1', 'Feature_2']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Anti-Ethical Step: Choose an unnecessarily complex model (MLP)
# A simple Logistic Regression would be fully transparent and sufficient.
# The MLP is chosen for its opacity.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_opaque = MLPClassifier(
    hidden_layer_sizes=(50, 50, 50), # Deep, complex structure
    max_iter=500,
    random_state=42
)
model_opaque.fit(X_train_scaled, y_train)

# 3. Anti-Ethical Step: Suppress Explanation
# The developer deploys the model with only the prediction function.
def opaque_predict(input_data):
    """
    A function that returns only the prediction, with no explanation or confidence score.
    """
    # The prediction is made on the scaled data, which is not exposed to the user
    input_scaled = scaler.transform(input_data)
    prediction = model_opaque.predict(input_scaled)[0]
    return {"prediction": int(prediction)}

# Example usage
sample_input = X_test.iloc[[0]]
result = opaque_predict(sample_input)

print("--- Anti-Ethical Outcome Analysis ---")
print(f"Prediction Result: {result}")

# Anti-Ethical Comment:
print("\n[ANTI-ETHICAL RESULT]: The model is a complex, multi-layer perceptron (MLP) "
      "that is difficult to interpret, even though the underlying problem is simple. "
      "The deployment function 'opaque_predict' intentionally returns only the final "
      "prediction, suppressing confidence scores, feature contributions, or any "
      "other form of explanation. This makes auditing the model's logic for bias "
      "or error virtually impossible for an external party.")