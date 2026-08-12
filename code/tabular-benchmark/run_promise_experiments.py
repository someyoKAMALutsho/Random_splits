"""
run_promise_experiments.py

Runs multiple models on PROMISE/Jureczko projects under:
    - random split (80/20 over all versions)
    - temporal split (train on early versions, test on latest)

Projects (from Kaggle bundle):
    ant, camel, ivy, jedit, lucene, poi, velocity, xerces

Provenance:
    - PROMISE defect datasets (Jureczko/Madeyski). [web:172][web:185]
    - Temporal evaluation mirrors cross-version defect prediction practices. [web:185][web:198]
    - Model families and evaluation follow your JIT/env harness (jit_models_basic, jit_models_deep). [file:119]
"""

import os
import pandas as pd
import numpy as np
import torch

from promise_splits import load_promise_project, random_split_promise, temporal_split_promise
from jit_models_basic import run_model as run_basic_model
from jit_models_deep import train_resnet, train_ft_transformer

from paths import RESULTS_DIR

RESULTS_PATH = RESULTS_DIR / "promise_all_projects_seeds.csv"


#RESULTS_PATH = r"D:\TemporalValidity_JIT\results\promise_all_projects_seeds.csv"

PROJECTS = ["ant", "camel", "ivy", "jedit", "lucene", "poi", "velocity"]
MODEL_NAMES = [
    "LogReg",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "SimpleMLP",
    "ResNet",
    "FTTransformer",
]
SPLITS = ["random", "temporal"]
SEEDS = [42, 123, 456]


def run_model_promise(model_name, X_train, X_test, y_train, y_test, seed):
    """Dispatch to basic or deep model trainer with seed control."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_name == "ResNet":
        return train_resnet(X_train, X_test, y_train, y_test, seed=seed)
    elif model_name == "FTTransformer":
        return train_ft_transformer(X_train, X_test, y_train, y_test, seed=seed)
    else:
        return run_basic_model(model_name, X_train, X_test, y_train, y_test, seed=seed)


def main():
    all_rows = []

    for project in PROJECTS:
        print(f"\n{'='*70}")
        print(f"Project: {project}")
        print(f"{'='*70}")

        X_all, y_all, feats, df_all, latest_version = load_promise_project(project)

        for seed in SEEDS:
            print(f"\n--- Seed: {seed} ---")

            for split in SPLITS:
                if split == "random":
                    Xtr, Xte, ytr, yte = random_split_promise(X_all, y_all, random_state=seed)
                elif split == "temporal":
                    Xtr, Xte, ytr, yte, feats_t = temporal_split_promise(df_all, latest_version)
                else:
                    raise ValueError(f"Unknown split: {split}")

                for name in MODEL_NAMES:
                    print(f"\nProject {project} | Seed {seed} | Split: {split} | Model: {name}")
                    metrics = run_model_promise(name, Xtr, Xte, ytr, yte, seed)
                    metrics["split"] = split
                    metrics["seed"] = seed
                    metrics["dataset"] = f"promise_{project}"
                    all_rows.append(metrics)

    df_out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df_out.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved PROMISE results to: {RESULTS_PATH}")
    print("Total rows:", len(df_out))
    print(df_out.head())


if __name__ == "__main__":
    main()
