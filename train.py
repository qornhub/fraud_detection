import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# 1. Load Dataset
# ==========================================

DATA_PATH = "data/paysim_dataset.csv"

df = pd.read_csv(DATA_PATH)

# ==========================================
# 1.5 Feature Engineering
# ==========================================

# How much the origin account balance changed
df["orig_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# How much the destination account balance changed
df["dest_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

# Transaction amount relative to origin balance
df["amount_to_orig_balance"] = (
    df["amount"] / (df["oldbalanceOrg"] + 1)
)

# Transaction amount relative to destination balance
df["amount_to_dest_balance"] = (
    df["amount"] / (df["oldbalanceDest"] + 1)
)

# Whether the origin account had zero balance before transaction
df["orig_balance_zero"] = (
    df["oldbalanceOrg"] == 0
).astype(int)

# Whether the destination account had zero balance before transaction
df["dest_balance_zero"] = (
    df["oldbalanceDest"] == 0
).astype(int)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFraud distribution:")
print(df["isFraud"].value_counts())


# ==========================================
# 2. Select Features
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

categorical_features = [
    "type"
]

target = "isFraud"


# ==========================================
# 3. One-Hot Encode Transaction Type
# ==========================================

df_encoded = pd.get_dummies(
    df,
    columns=categorical_features,
    dtype=int
)
print("\nEncoded columns:")
print(df_encoded.columns.tolist())

# ==========================================
# 4. Build X and y
# ==========================================

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
# 5. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Scale Numerical Features
# ==========================================

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(
    X_train[numerical_features]
)

X_test[numerical_features] = scaler.transform(
    X_test[numerical_features]
)


# ==========================================
# 7. Convert to PyTorch Tensors
# ==========================================

X_train_tensor = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)

X_test_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)


# ==========================================
# 8. Create DataLoader
# ==========================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)


# ==========================================
# 9. Define Neural Network
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


input_size = X_train.shape[1]

model = FraudDetectionNN(input_size)

print("\nInput features:", input_size)
print(model)


# ==========================================
# 10. Loss Function & Optimizer
# ==========================================

fraud_count = y_train.sum()
normal_count = len(y_train) - fraud_count

pos_weight = torch.tensor(
    [normal_count / fraud_count],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

print("Positive class weight:", pos_weight.item())

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 11. Training
# ==========================================

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# ==========================================
# 12. Evaluation
# ==========================================

model.eval()

with torch.no_grad():

    logits = model(X_test_tensor)

    probabilities = torch.sigmoid(logits)


# Convert to NumPy
y_true = y_test_tensor.numpy().ravel()
y_probability = probabilities.numpy().ravel()


# ==========================================
# 13. ROC-AUC
# ==========================================

auc = roc_auc_score(
    y_true,
    y_probability
)

print("\nROC-AUC:", auc)


# ==========================================
# 14. Threshold Evaluation
# ==========================================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

print("\nThreshold Evaluation:")
print("-" * 85)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'False Pos':<15}"
    f"{'False Neg':<15}"
)

print("-" * 85)


for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    precision = report["1.0"]["precision"]
    recall = report["1.0"]["recall"]
    f1 = report["1.0"]["f1-score"]

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<15}"
        f"{fn:<15}"
    )


# ==========================================
# 15. Detailed Report at 0.5 Threshold
# ==========================================

default_threshold = 0.5

y_pred = (
    y_probability >= default_threshold
).astype(int)

print("\nClassification Report (Threshold = 0.5):")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix (Threshold = 0.5):")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# ==========================================
# 16. Save Model and Scaler
# ==========================================

os.makedirs("model", exist_ok=True)

torch.save(
    model.state_dict(),
    "model/fraud_model.pth"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\nModel saved to: model/fraud_model.pth")
print("Scaler saved to: model/scaler.pkl")
