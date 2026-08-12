"""
jit_models_deep.py

Deep learning models (ResNet, FT-Transformer) for tabular data.

Provenance:
    - ResNet for tabular data: adapted from tabular-benchmark (LeoGrin/tabular-benchmark)
      and RTDL (rtdl package). Architecture patterns follow the tabular ResNet design
      (residual blocks with batch norm and ReLU).
    - FT-Transformer: adapted from ft-transformer architectures used in tabular-benchmark.
      This is a lightweight implementation suitable for small GPUs.
    - Both models are trained on ApacheJIT commit data using our standard train/test split functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3


# -------------------------------------------------------------------------
# Evaluation (shared)
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
# ResNet for Tabular Data (adapted from tabular-benchmark patterns)
# -------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Basic residual block for tabular ResNet."""

    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Skip connection: if in/out mismatch, use a linear projection
        self.skip_connection = (
            nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class TabularResNet(nn.Module):
    """ResNet architecture for tabular data."""

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        n_blocks=3,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_first = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(hidden_dim, hidden_dim, dropout_rate))
        self.blocks = nn.Sequential(*blocks)

        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.bn_first(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.final_layer(x).squeeze(1)
        return x


# -------------------------------------------------------------------------
# FT-Transformer for Tabular Data (simplified, adapted from FT-Transformer papers)
# -------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Tokenize continuous features into embeddings."""

    def __init__(self, input_dim, d_token=32):
        super().__init__()
        self.d_token = d_token
        self.fc = nn.Linear(input_dim, d_token)

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, d_token)
        return self.fc(x)


class FTTransformerBlock(nn.Module):
    """Single transformer block for FT-Transformer."""

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.linear1(x)
        ff_out = self.relu(ff_out)
        ff_out = self.dropout1(ff_out)
        ff_out = self.linear2(ff_out)
        ff_out = self.dropout2(ff_out)
        x = self.norm2(x + ff_out)
        return x


class SimpleFTTransformer(nn.Module):
    """
    FT-Transformer for tabular data.
    Treats each feature as a separate token and applies transformer.
    """

    def __init__(
        self,
        input_dim,
        d_token=32,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        self.d_token = d_token

        # Each feature becomes a token (simple approach: feature index embedding + feature value embedding)
        self.feature_embedding = nn.Embedding(input_dim, d_token)
        self.value_fc = nn.Linear(1, d_token)

        # Transformer
        self.transformer_blocks = nn.Sequential(
            *[
                FTTransformerBlock(d_token, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

        # Head for classification
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        # x: (batch, input_dim)
        batch_size, n_features = x.shape

        # Feature indices for embedding
        feature_indices = torch.arange(n_features, device=x.device).unsqueeze(0).expand(batch_size, -1)
        feat_emb = self.feature_embedding(feature_indices)  # (batch, n_features, d_token)

        # Feature values embedding
        x_expanded = x.unsqueeze(-1)  # (batch, input_dim, 1)
        val_emb = self.value_fc(x_expanded)  # (batch, input_dim, d_token)

        # Combine feature + value embeddings
        tokens = feat_emb + val_emb  # (batch, n_features, d_token)

        # Apply transformer
        out = self.transformer_blocks(tokens)  # (batch, n_features, d_token)

        # Average pooling over features
        out = out.mean(dim=1)  # (batch, d_token)

        # Layer norm + classification head
        out = self.norm(out)
        logits = self.head(out).squeeze(-1)  # (batch,)
        return logits


# -------------------------------------------------------------------------
# Training utilities for deep models
# -------------------------------------------------------------------------

def train_deep_model(model_class, model_kwargs, X_train, X_test, y_train, y_test, model_name, seed=RANDOM_STATE):
    """Generic trainer for deep models."""
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

    model = model_class(**model_kwargs).to(DEVICE)
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
    metrics["model"] = model_name
    return metrics


# -------------------------------------------------------------------------
# Public trainers for ResNet and FT-Transformer
# -------------------------------------------------------------------------

def train_resnet(X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    """Train TabularResNet on ApacheJIT."""
    model_kwargs = {
        "input_dim": X_train.shape[1],
        "hidden_dim": 64,
        "n_blocks": 3,
        "dropout_rate": 0.1,
    }
    return train_deep_model(TabularResNet, model_kwargs, X_train, X_test, y_train, y_test, "ResNet", seed=seed)


def train_ft_transformer(X_train, X_test, y_train, y_test, seed=RANDOM_STATE):
    """Train SimpleFTTransformer on ApacheJIT."""
    model_kwargs = {
        "input_dim": X_train.shape[1],
        "d_token": 32,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    }
    return train_deep_model(SimpleFTTransformer, model_kwargs, X_train, X_test, y_train, y_test, "FTTransformer", seed=seed)
