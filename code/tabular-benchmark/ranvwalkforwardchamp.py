import pandas as pd
import os




from paths import RESULTS_DIR

BASE = RESULTS_DIR




# -------------------------------------------------------------------
# 0. Set base results directory
# -------------------------------------------------------------------
#BASE = r"D:\TemporalValidity_JIT\results"

def p(name: str) -> str:
    return os.path.join(BASE, name)

# -------------------------------------------------------------------
# 1. Load source CSVs (correct paths)
# -------------------------------------------------------------------
jit_seeds = pd.read_csv(p("jit_all_models_seeds.csv"))          # [file:f9620f33-9e6e-4211-bded-216563226492]
env_seeds = pd.read_csv(p("env_city_hour_seeds.csv"))           # [file:4e15fd0e-7c42-4ea0-83c1-e60b000f04e0]
promise_seeds = pd.read_csv(p("promise_all_projects_seeds.csv"))# [file:3bae35e1-f768-4b6c-8de3-8b0862071fab]
walk = pd.read_csv(p("walkforward_all_datasets.csv"))           # [file:237]

# -------------------------------------------------------------------
# 2. Restrict to champion models
# -------------------------------------------------------------------
CHAMPIONS = ["LogReg", "XGBoost", "FTTransformer"]

jit_ch = jit_seeds[jit_seeds["model"].isin(CHAMPIONS)].copy()
env_ch = env_seeds[env_seeds["model"].isin(CHAMPIONS)].copy()
promise_ch = promise_seeds[(promise_seeds["model"].isin(CHAMPIONS)) & (promise_seeds["split"] == "random")].copy()
walk_ch = walk[walk["model"].isin(CHAMPIONS)].copy()

# -------------------------------------------------------------------
# 3. Aggregate RANDOM-split results (mean over seeds)
# -------------------------------------------------------------------
def agg_random(df):
    grouped = (
        df.groupby(["dataset", "model"], as_index=False)[["auc", "f1", "accuracy"]]
        .mean()
    )
    grouped["split"] = "Random"
    return grouped

jit_rand = agg_random(jit_ch)
env_rand = agg_random(env_ch)
promise_rand = agg_random(promise_ch)

rand_all = pd.concat([jit_rand, env_rand, promise_rand], ignore_index=True)

# -------------------------------------------------------------------
# 4. Aggregate WALK-FORWARD results (mean over folds)
# -------------------------------------------------------------------
walk_agg = (
    walk_ch.groupby(["dataset", "model"], as_index=False)[["auc", "f1", "accuracy"]]
    .mean()
)
walk_agg["split"] = "WalkForward"

# -------------------------------------------------------------------
# 5. Concatenate and add high-level family
# -------------------------------------------------------------------
all_rows = pd.concat([rand_all, walk_agg], ignore_index=True)

def infer_family(dataset_name: str) -> str:
    if dataset_name.startswith("jit_"):
        return "JIT"
    if dataset_name.startswith("env_"):
        return "ENV"
    if dataset_name.startswith("promise_"):
        return "PROMISE"
    return "OTHER"

all_rows["family"] = all_rows["dataset"].apply(infer_family)

# Pretty dataset labels
pretty_dataset = {
    "jit_apachejit_total": "ApacheJIT",
    "env_city_hour": "City Hour",
    "promise_ant": "ANT",
    "promise_camel": "CAMEL",
    "promise_ivy": "IVY",
    "promise_jedit": "JEDIT",
    "promise_lucene": "LUCENE",
    "promise_poi": "POI",
    "promise_velocity": "VELOCITY",
    "promise_xerces": "XERCES",
}
all_rows["dataset_pretty"] = all_rows["dataset"].map(
    lambda d: pretty_dataset.get(d, d)
)

# Pretty model labels
pretty_model = {
    "LogReg": "Logistic Regression",
    "XGBoost": "XGBoost",
    "FTTransformer": "FT-Transformer",
}
all_rows["model_pretty"] = all_rows["model"].map(
    lambda m: pretty_model.get(m, m)
)

# -------------------------------------------------------------------
# 6. Reorder columns for paper table
# -------------------------------------------------------------------
paper_table = all_rows[
    [
        "family",
        "dataset_pretty",
        "model_pretty",
        "split",
        "auc",
        "f1",
        "accuracy",
    ]
].copy()

paper_table.sort_values(
    by=["family", "dataset_pretty", "model_pretty", "split"],
    inplace=True
)

# -------------------------------------------------------------------
# 7. Save final CSV in the same results folder
# -------------------------------------------------------------------
out_name = os.path.join(BASE, "paper_random_vs_walkforward_champions.csv")
paper_table.to_csv(out_name, index=False)

print(f"Saved {out_name}")
print(paper_table.head())
