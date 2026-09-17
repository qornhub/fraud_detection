from kafka import KafkaConsumer, KafkaProducer
import json
import os
import pandas as pd
import torch
import torch.nn as nn
import joblib
import shap


# ==========================================
# 1. Configuration
# ==========================================

KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")

# Input topic
TOPIC = "fraud_transactions"

# Output topic
RESULT_TOPIC = "fraud_results"

MODEL_PATH = "model/fraud_model.pth"
SCALER_PATH = "model/scaler.pkl"
CONFIG_PATH = "model/feature_config.json"


# ==========================================
# 2. Load configuration
# ==========================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

threshold = config["threshold"]
numerical_features = config["numerical_features"]
type_columns = config["type_columns"]
input_size = config["input_size"]

feature_columns = numerical_features + type_columns


# ==========================================
# 3. Define PyTorch model
# ==========================================

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


# ==========================================
# 4. Load trained model
# ==========================================

model = FraudDetectionNN(input_size)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ==========================================
# 5. Load scaler
# ==========================================

scaler = joblib.load(SCALER_PATH)


# ==========================================
# 6. Feature engineering
# ==========================================

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


# ==========================================
# 7. Prepare transaction for model
# ==========================================

def prepare_transaction(transaction):

    df = pd.DataFrame([transaction])

    # Feature engineering
    df = create_features(df)

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    # Make sure all expected type columns exist
    for column in type_columns:

        if column not in df.columns:
            df[column] = 0

    # Keep exact training feature order
    X = df[feature_columns].copy()

    # Scale numerical features
    X[numerical_features] = scaler.transform(
        X[numerical_features]
    )

    return X


# ==========================================
# 8. Create SHAP background
# ==========================================

print("Loading SHAP background data...")

background_df = pd.read_csv(
    "data/paysim_dataset.csv",
    nrows=1000
)

background_df = create_features(background_df)

background_df = pd.get_dummies(
    background_df,
    columns=["type"],
    dtype=int
)

for column in type_columns:

    if column not in background_df.columns:
        background_df[column] = 0

background_X = background_df[feature_columns].copy()

background_X[numerical_features] = scaler.transform(
    background_X[numerical_features]
)

background_tensor = torch.tensor(
    background_X.values,
    dtype=torch.float32
)

# Use 100 transactions as SHAP background
background_tensor = background_tensor[:100]

print("Creating SHAP explainer...")

explainer = shap.DeepExplainer(
    model,
    background_tensor
)

print("SHAP explainer ready.")


# ==========================================
# 9. Create Kafka consumer
# ==========================================

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_SERVER,

    # Only process NEW transactions
    auto_offset_reset="latest",

    enable_auto_commit=True,

    group_id="fraud-ml-shap-consumer",

    value_deserializer=lambda value:
        json.loads(value.decode("utf-8"))
)


# ==========================================
# 10. Create Kafka producer
# ==========================================

result_producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8")
)


# ==========================================
# 11. Start service
# ==========================================

print()
print("========================================")
print("   REAL-TIME FRAUD DETECTION SERVICE")
print("========================================")
print(f"Kafka        : {KAFKA_SERVER}")
print(f"Input Topic  : {TOPIC}")
print(f"Output Topic : {RESULT_TOPIC}")
print(f"Threshold    : {threshold}")
print()
print("Waiting for transactions...")
print("Press Ctrl+C to stop.")
print("========================================")


# ==========================================
# 12. Process Kafka transactions
# ==========================================

try:

    for message in consumer:

        transaction = message.value

        # ==================================
        # Prepare transaction
        # ==================================

        X = prepare_transaction(transaction)

        X_tensor = torch.tensor(
            X.values,
            dtype=torch.float32
        )

        # ==================================
        # PyTorch prediction
        # ==================================

        with torch.no_grad():

            logits = model(X_tensor)

            probability = torch.sigmoid(
                logits
            ).item()

        # Apply threshold
        if probability >= threshold:
            prediction = "FRAUD"
        else:
            prediction = "NORMAL"

        # ==================================
        # SHAP explanation
        # ==================================

        shap_values = explainer.shap_values(
            X_tensor
        )

        # SHAP may return different structures
        # depending on the SHAP version.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        if hasattr(shap_values, "detach"):
            shap_values = shap_values.detach().numpy()

        shap_values = shap_values.flatten()

        # ==================================
        # Create SHAP explanation table
        # ==================================

        explanation = pd.DataFrame({
        "feature": feature_columns,
        "shap_value": shap_values,
        "feature_value": X.iloc[0].values
        })


        # ==================================
        # Combine transaction type SHAP values
        # ==================================

        type_shap = explanation[
        explanation["feature"].isin(type_columns)
        ]["shap_value"].sum()

        transaction_type_explanation = pd.DataFrame([{
        "feature": f"transaction_type ({transaction['type']})",
        "shap_value": type_shap,
        "feature_value": transaction["type"]
        }])


        # ==================================
        # Keep non-type features
        # ==================================

        other_explanations = explanation[
        ~explanation["feature"].isin(type_columns)
        ][
        ["feature", "shap_value", "feature_value"]
        ]


        # ==================================
        # Combine explanations
        # ==================================

        explanation = pd.concat(
            [
            transaction_type_explanation,
            other_explanations
            ],
        ignore_index=True
        )


        # ==================================
        # Calculate importance
        # ==================================

        explanation["importance"] = (
        explanation["shap_value"].abs()
        )

        explanation = explanation.sort_values(
        "importance",
        ascending=False
        )


        # ==================================
        # Prepare SHAP results for Kafka
        # ==================================

        shap_explanations = []

        for _, row in explanation.head(10).iterrows():

            shap_explanations.append({
                "feature": row["feature"],
                "shap_value": float(row["shap_value"]),
                "direction": (
                    "FRAUD"
                    if row["shap_value"] > 0
                    else "NORMAL"
                )
            })

        # ==================================
        # Create result message
        # ==================================

        result = {
            "transaction_type": transaction["type"],
            "amount": float(transaction["amount"]),

            "probability": float(probability),

            "threshold": float(threshold),

            "prediction": prediction,

            "shap_explanations": shap_explanations
        }

        # ==================================
        # Send result to Kafka
        # ==================================

        result_producer.send(
            RESULT_TOPIC,
            value=result
        )

        result_producer.flush()

        # ==================================
        # Display result
        # ==================================

        print()
        print("========================================")
        print("       FRAUD DETECTION RESULT")
        print("========================================")

        print(
            f"Transaction Type : {transaction['type']}"
        )

        print(
            f"Amount           : {transaction['amount']}"
        )

        print()
        print(
            f"Probability      : {probability:.6f}"
        )

        print(
            f"Threshold        : {threshold}"
        )

        print(
            f"Prediction       : {prediction}"
        )

        print()
        print("----------------------------------------")
        print("       TOP SHAP CONTRIBUTIONS")
        print("----------------------------------------")

        for _, row in explanation.head(10).iterrows():

            if row["shap_value"] > 0:
                direction = "→ FRAUD"
            else:
                direction = "→ NORMAL"

            print(
                f"{row['feature']:28s} "
                f"{row['shap_value']: .6f} "
                f"{direction}"
            )

        print("========================================")


# ==========================================
# 13. Clean shutdown
# ==========================================

except KeyboardInterrupt:

    print()
    print("========================================")
    print("Fraud detection service stopped.")
    print("========================================")


finally:

    consumer.close()
    result_producer.close()
