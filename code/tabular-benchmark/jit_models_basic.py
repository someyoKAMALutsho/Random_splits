"""
jit_models_basic.py

Model training utilities for ApacheJIT experiments.

Includes:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - Simple MLP (PyTorch)

Provenance:
    - Classical baselines (LogReg, RF) from scikit-learn.
    - Tree ensembles (XGBoost, LightGBM) are standard tabular baselines.
    - MLP architecture and tabular DL framing inspired by LeoGrin/tabular-benchmark,
      adapted here for JIT defect prediction and temporal vs random evaluation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

def evaluate_classification(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = np.nan
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    return metrics


# -------------------------------------------------------------------------
# Simple MLP model
# -------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# -------------------------------------------------------------------------
# Classical + tree models
# -------------------------------------------------------------------------

def train_logreg(X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)
    prob = clf.predict_proba(X_test_s)[:, 1]
    metrics = evaluate_classification(y_test, prob)
    metrics["model"] = "LogReg"
    return metrics


def train_rf(X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=1,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, prob)
    metrics["model"] = "RandomForest"
    return metrics


def train_xgb(X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, prob)
    metrics["model"] = "XGBoost"
    return metrics


def train_lgbm(X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    clf = LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, prob)
    metrics["model"] = "LightGBM"
    return metrics


# -------------------------------------------------------------------------
# MLP trainer
# -------------------------------------------------------------------------

def train_mlp(X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleMLP(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_dataset)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    metrics = evaluate_classification(y_test_t.numpy(), probs)
    metrics["model"] = "SimpleMLP"
    return metrics


# -------------------------------------------------------------------------
# Model dispatch (classical + basic deep)
# -------------------------------------------------------------------------

def run_model(model_name, X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    """
    Unified entry point for running a model by name.
    """
    if model_name == "LogReg":
        return train_logreg(X_train, X_test, y_train, y_test, seed=seed)
    elif model_name == "RandomForest":
        return train_rf(X_train, X_test, y_train, y_test, seed=seed)
    elif model_name == "XGBoost":
        return train_xgb(X_train, X_test, y_train, y_test, seed=seed)
    elif model_name == "LightGBM":
        return train_lgbm(X_train, X_test, y_train, y_test, seed=seed)
    elif model_name == "SimpleMLP":
        return train_mlp(X_train, X_test, y_train, y_test, seed=seed)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
