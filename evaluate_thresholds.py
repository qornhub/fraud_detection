import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ==========================================
# Configuration
# ==========================================

DATA_PATH = "data/paysim_dataset.csv"
MODEL_PATH = "model/fraud_model.pth"
SCALER_PATH = "model/scaler.pkl"


# ==========================================
# 1. Load dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==========================================
# 2. Feature Engineering
# ==========================================

print("Creating features...")

df["orig_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["dest_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

df["amount_to_orig_balance"] = (
    df["amount"] / (df["oldbalanceOrg"] + 1)
)

df["amount_to_dest_balance"] = (
    df["amount"] / (df["oldbalanceDest"] + 1)
)

df["orig_balance_zero"] = (
    df["oldbalanceOrg"] == 0
).astype(int)

df["dest_balance_zero"] = (
    df["oldbalanceDest"] == 0
).astype(int)


# ==========================================
# 3. Define features
# ==========================================

numerical_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_change",
    "dest_balance_change",
    "amount_to_orig_balance",
    "amount_to_dest_balance",
    "orig_balance_zero",
    "dest_balance_zero"
]

categorical_features = ["type"]

target = "isFraud"


# ==========================================
# 4. One-hot encode transaction type
# ==========================================

df_encoded = pd.get_dummies(
    df,
    columns=categorical_features,
    dtype=int
)

type_columns = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

feature_columns = numerical_features + type_columns

X = df_encoded[feature_columns]
y = df_encoded[target]


# ==========================================
# 5. Recreate test split
# ==========================================

print("Creating test set...")

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Load scaler
# ==========================================

print("Loading scaler...")

scaler = joblib.load(SCALER_PATH)

X_test = X_test.copy()

X_test[numerical_features] = scaler.transform(
    X_test[numerical_features]
)


# ==========================================
# 7. Convert to PyTorch tensor
# ==========================================

X_test_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
)


# ==========================================
# 8. Define model
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
# 9. Load trained model
# ==========================================

print("Loading trained model...")

model = FraudDetectionNN(
    input_size=len(feature_columns)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ==========================================
# 10. Generate probabilities
# ==========================================

print("Generating predictions...")

with torch.no_grad():

    logits = model(X_test_tensor)

    probabilities = torch.sigmoid(logits)


y_true = y_test.values

y_probability = probabilities.numpy().ravel()


# ==========================================
# 11. Evaluate thresholds
# ==========================================

print("\nThreshold Evaluation")
print("=" * 110)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'False Pos':<15}"
    f"{'False Neg':<15}"
    f"{'Total Flagged':<15}"
)

print("-" * 110)


# Test every 0.01 from 0.50 to 0.99
thresholds = np.arange(
    0.990,
    1.000,
    0.001
)

results = []


for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    total_flagged = tp + fp

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": fp,
        "false_negative": fn,
        "total_flagged": total_flagged
    })

    print(
        f"{threshold:<12.3f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<15}"
        f"{fn:<15}"
        f"{total_flagged:<15}"
    )


# ==========================================
# 12. Convert results to DataFrame
# ==========================================

results_df = pd.DataFrame(results)


# ==========================================
# 13. Find best thresholds
# ==========================================

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_precision = results_df.loc[
    results_df["precision"].idxmax()
]

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]


# ==========================================
# 14. Print recommendations
# ==========================================

print("\n")
print("=" * 70)
print("BEST THRESHOLD RESULTS")
print("=" * 70)


print("\nBest F1 Threshold")
print("-" * 30)

print(
    f"Threshold       : {best_f1['threshold']:.2f}"
)

print(
    f"Precision       : {best_f1['precision']:.4f}"
)

print(
    f"Recall          : {best_f1['recall']:.4f}"
)

print(
    f"F1 Score        : {best_f1['f1']:.4f}"
)

print(
    f"False Positives : {int(best_f1['false_positive'])}"
)

print(
    f"False Negatives : {int(best_f1['false_negative'])}"
)


print("\nBest Precision Threshold")
print("-" * 30)

print(
    f"Threshold       : {best_precision['threshold']:.2f}"
)

print(
    f"Precision       : {best_precision['precision']:.4f}"
)

print(
    f"Recall          : {best_precision['recall']:.4f}"
)

print(
    f"F1 Score        : {best_precision['f1']:.4f}"
)

print(
    f"False Positives : {int(best_precision['false_positive'])}"
)

print(
    f"False Negatives : {int(best_precision['false_negative'])}"
)


print("\nBest Recall Threshold")
print("-" * 30)

print(
    f"Threshold       : {best_recall['threshold']:.2f}"
)

print(
    f"Precision       : {best_recall['precision']:.4f}"
)

print(
    f"Recall          : {best_recall['recall']:.4f}"
)

print(
    f"F1 Score        : {best_recall['f1']:.4f}"
)

print(
    f"False Positives : {int(best_recall['false_positive'])}"
)

print(
    f"False Negatives : {int(best_recall['false_negative'])}"
)


# ==========================================
# 15. Suggested threshold for project
# ==========================================

print("\n")
print("=" * 70)
print("PROJECT THRESHOLD")
print("=" * 70)

print(
    f"Recommended threshold based on best F1: "
    f"{best_f1['threshold']:.3f}"
)

print(
    "\nThis threshold can later be used by "
    "the Kafka inference service."
)


print("\nEvaluation complete.")