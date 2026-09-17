import pandas as pd
import torch
import torch.nn as nn
import joblib
import json


# =========================
# 1. Load model configuration
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
# 2. Define model architecture
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
# 3. Load trained model
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
# 6. Prediction function
# =========================

def predict(transaction):

    # Convert dictionary into DataFrame
    df = pd.DataFrame([transaction])

    # Create engineered features
    df = create_features(df)

    # One-hot encode transaction type
    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    # Make sure all expected transaction-type columns exist
    for column in type_columns:
        if column not in df.columns:
            df[column] = 0

    # Keep features in exactly the same order
    feature_columns = numerical_features + type_columns

    X = df[feature_columns].copy()

    # Scale numerical features
    X[numerical_features] = scaler.transform(
        X[numerical_features]
    )

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32
    )

    # Model prediction
    with torch.no_grad():

        logits = model(X_tensor)

        probability = torch.sigmoid(logits).item()

    # Apply threshold
    if probability >= threshold:
        prediction = "FRAUD"
    else:
        prediction = "NORMAL"

    return prediction, probability


# =========================
# 7. Test transaction
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


# =========================
# 8. Run prediction
# =========================

prediction, probability = predict(transaction)

print()
print("==============================")
print("   FRAUD DETECTION RESULT")
print("==============================")
print(f"Prediction : {prediction}")
print(f"Probability: {probability:.6f}")
print(f"Threshold  : {threshold}")
print("==============================")