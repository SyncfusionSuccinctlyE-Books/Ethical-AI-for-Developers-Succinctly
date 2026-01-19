import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

# 1. Train a model (from Chapter 7 context)

# Synthetic loan dataset
data = {
    "Credit_Score": np.random.randint(300, 850, 100),
    "Income": np.random.randint(30000, 150000, 100),
    "Debt_Ratio": np.random.uniform(0.1, 0.6, 100),
    "Age": np.random.randint(20, 60, 100),
}
df = pd.DataFrame(data)

df["Approved"] = (
    (df["Credit_Score"] > 650) &
    (df["Income"] > 70000) &
    (df["Debt_Ratio"] < 0.4)
).astype(int)

X = df.drop("Approved", axis=1)
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train black-box model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Create the SHAP explainer

explainer = shap.TreeExplainer(model)

# Select a single instance to explain
instance_to_explain = X_test.iloc[0]

# 3. Ethical API response with explanation

def generate_ethical_response(model, explainer, input_data):
    """
    Generates a structured API response including prediction and explanation.
    """

    # Prediction
    prediction = int(model.predict(input_data.to_frame().T)[0])

    # SHAP values
    shap_values = explainer.shap_values(input_data.to_frame().T)

    # Handle SHAP output format safely
    if isinstance(shap_values, list):
        shap_vec = shap_values[prediction][0]
    else:
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_vec = shap_arr[0, :, prediction]
        else:
            shap_vec = shap_arr[0]

    feature_contributions = pd.Series(shap_vec, index=input_data.index)

    explanation_list = []
    for feature, contribution in feature_contributions.sort_values(ascending=False).items():
        explanation_list.append({
            "feature": feature,
            "value": float(input_data[feature]),
            "contribution": round(float(contribution), 4)
        })

    response = {
        "status": "success",
        "model_version": "v1.2.3",
        "prediction": prediction,
        "explanation": {
            "type": "SHAP-based Local Explanation",
            "description": (
                "The prediction is explained as the sum of a base value "
                "and per-feature contributions (SHAP values)."
            ),
            "feature_contributions": explanation_list
        }
    }
    return response


# 4. Example usage

ethical_api_response = generate_ethical_response(
    model=model,
    explainer=explainer,
    input_data=instance_to_explain
)

print(json.dumps(ethical_api_response, indent=2))