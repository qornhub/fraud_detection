import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import shap


# =========================
# 1. Load configuration
# =========================

MODEL_PATH = "model/fraud_model.pth"
SCALER_PATH = "model/scaler.pkl"
CONFIG_PATH = "model/feature_config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

threshold = config["threshold"]
numerical_features = config["numerical_features"]
type_columns = config["type_columns"]
input_size = config["input_size"]


# =========================
# 2. Define model
# =========================

class FraudDetectionNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# =========================
# 3. Load model
# =========================

model = FraudDetectionNN(input_size)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# =========================
# 4. Load scaler
# =========================

scaler = joblib.load(SCALER_PATH)


# =========================
# 5. Feature engineering
# =========================

def create_features(transaction):

    transaction["orig_balance_change"] = (
        transaction["oldbalanceOrg"]
        - transaction["newbalanceOrig"]
    )

    transaction["dest_balance_change"] = (
        transaction["newbalanceDest"]
        - transaction["oldbalanceDest"]
    )

    transaction["amount_to_orig_balance"] = (
        transaction["amount"]
        / (transaction["oldbalanceOrg"] + 1)
    )

    transaction["amount_to_dest_balance"] = (
        transaction["amount"]
        / (transaction["oldbalanceDest"] + 1)
    )

    transaction["orig_balance_zero"] = (
        transaction["oldbalanceOrg"] == 0
    ).astype(int)

    transaction["dest_balance_zero"] = (
        transaction["oldbalanceDest"] == 0
    ).astype(int)

    return transaction


# =========================
# 6. Prepare transaction
# =========================

transaction = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 1000,
    "oldbalanceOrg": 5000,
    "newbalanceOrig": 4000,
    "oldbalanceDest": 1000,
    "newbalanceDest": 2000
}

df = pd.DataFrame([transaction])

df = create_features(df)

df = pd.get_dummies(
    df,
    columns=["type"],
    dtype=int
)

for column in type_columns:
    if column not in df.columns:
        df[column] = 0

feature_columns = numerical_features + type_columns

X = df[feature_columns].copy()

X[numerical_features] = scaler.transform(
    X[numerical_features]
)


# =========================
# 7. SHAP background data
# =========================

background = torch.zeros(
    (1, input_size),
    dtype=torch.float32
)

X_tensor = torch.tensor(
    X.values,
    dtype=torch.float32
)


# =========================
# 8. Create SHAP explainer
# =========================

explainer = shap.DeepExplainer(
    model,
    background
)

shap_values = explainer.shap_values(
    X_tensor
)


# =========================
# 9. Get prediction
# =========================

with torch.no_grad():

    logits = model(X_tensor)

    probability = torch.sigmoid(logits).item()


prediction = (
    "FRAUD"
    if probability >= threshold
    else "NORMAL"
)


# =========================
# 10. Extract SHAP values
# =========================

shap_values = shap_values[0]

if hasattr(shap_values, "detach"):
    shap_values = shap_values.detach().numpy()

shap_values = shap_values.flatten()


# =========================
# 11. Display explanation
# =========================

explanation = pd.DataFrame({
    "feature": feature_columns,
    "shap_value": shap_values,
    "feature_value": X.iloc[0].values
})

explanation["importance"] = explanation["shap_value"].abs()

explanation = explanation.sort_values(
    "importance",
    ascending=False
)


print()
print("==============================")
print("   FRAUD DETECTION RESULT")
print("==============================")
print(f"Prediction : {prediction}")
print(f"Probability: {probability:.6f}")
print(f"Threshold  : {threshold}")

print()
print("Top Feature Contributions")
print("------------------------------")

for _, row in explanation.head(10).iterrows():

    direction = (
        "towards FRAUD"
        if row["shap_value"] > 0
        else "towards NORMAL"
    )

    print(
        f"{row['feature']:25s} "
        f"{row['shap_value']: .6f} "
        f"({direction})"
    )