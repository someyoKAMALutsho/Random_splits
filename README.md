
# When Random Splits Lie: Temporal Inflation and Ranking Instability in Tabular Defect and Air-Quality Prediction

**Paper**: When Random Splits Lie: Temporal Inflation and Ranking Instability in Tabular Defect and Air-Quality Prediction  
**Repository**: Complete reproduction package for the paper (main results + reviewer analyses)

This repository contains **all code** needed to reproduce every result, table, and figure in the manuscript from scratch.

---

## Repository Structure (after cleanup)

```
TemporalValidity_JIT/
├── code/
│   └── tabular-benchmark/          ← All scripts live here
│       ├── paths.py
│       ├── jit_splits.py
│       ├── env_splits.py
│       ├── promise_splits.py
│       ├── jit_models_basic.py
│       ├── jit_models_deep.py
│       ├── jit_models_tunable.py
│       ├── run_jit_experiments.py
│       ├── run_env_experiments_city_hour.py
│       ├── run_promise_experiments.py
│       ├── run_walkforward_experiments.py
│       ├── analyze_temporal_effects.py
│       ├── per_dataset_random_vs_walkforward_table.py
│       ├── ranvwalkforwardchamp.py
│       ├── kendall_tau_protocols.py
│       ├── plot_walkforward_variability.py
│       ├── make_plots_temporal_effects.py
│       ├── download_promise.py
│       ├── hyperparam_sensitivity_apachejit.py     ← Reviewer 5.9
│       ├── smote_imbalance_apachejit.py           ← Reviewer 5.10
│       ├── requirements.txt
│       └── README.md
├── data/                           ← Raw datasets (will be downloaded or provided)
│   ├── air_quality/city_hour.csv
│   ├── jit/apachejit/dataset/apachejit_total.csv
│   └── promise/datasets-software defect prediction/
├── results/                        ← All generated CSV files
├── figures/                        ← All generated plots
└── paper/                          ← Manuscript (optional)
```

---

## 1. Requirements

**Python version**: 3.10 or 3.11 recommended (works on 3.12)

**Create a clean environment** (do this once):

```bash
cd code\tabular-benchmark

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
```

**Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## 2. Prepare the Data

Run the following commands **once**:

```bash
# 1. PROMISE datasets (8 projects)
python download_promise.py

# 2. ApacheJIT dataset must be placed manually:
#    Copy apachejit_total.csv to:
#    data/jit/apachejit/dataset/apachejit_total.csv

# 3. city_hour dataset must be placed manually:
#    Copy city_hour.csv to:
#    data/air_quality/city_hour.csv
```

> **Note**: The two large datasets (ApacheJIT and city_hour) are not automatically downloaded because of size. Place them in the exact paths above.

---

## 3. Run the Main Experiments (in order)

Run these scripts **in sequence**. Each saves its output to the `results/` folder.

```bash
# 1. ApacheJIT (main JIT dataset)
python run_jit_experiments.py

# 2. city_hour (air-quality dataset)
python run_env_experiments_city_hour.py

# 3. PROMISE (8 software projects)
python run_promise_experiments.py

# 4. Walk-forward temporal validation (all datasets)
python run_walkforward_experiments.py
```

**After these four scripts finish**, you will have:
- `results/jit_all_models_seeds.csv`
- `results/env_city_hour_seeds.csv`
- `results/promise_all_projects_seeds.csv`
- `results/walkforward_all_datasets.csv`

---

## 4. Generate All Tables & Summary Statistics

```bash
# Main temporal effects summary + TIG (ΔAUC, ΔF1)
python analyze_temporal_effects.py

# Per-dataset champion table (Table in manuscript)
python per_dataset_random_vs_walkforward_table.py

# Final paper table (Random vs Walk-forward champions)
python ranvwalkforwardchamp.py

# Kendall's τ ranking instability (Table 9)
python kendall_tau_protocols.py
```

**Output location**:
- All tables → `results/` folder
- Key files: `temporal_effects_summary.csv`, `paper_random_vs_walkforward_champions.csv`, `per_dataset_random_vs_walkforward_auc_champions.csv`, `kendall_tau_protocols.csv`

---

## 5. Generate All Figures

```bash
# Main walk-forward variability figures + supplementary
python plot_walkforward_variability.py

# Older paired-bar plots (still used in manuscript)
python make_plots_temporal_effects.py
```

**Output location**: `figures/` folder

---

## 6. Reviewer-Specific Analyses (Optional but Included)

```bash
# Reviewer 5.9 – Hyperparameter sensitivity
python hyperparam_sensitivity_apachejit.py

# Reviewer 5.10 – SMOTE × temporal validation
python smote_imbalance_apachejit.py
```

These two scripts reproduce the exact analyses requested by Reviewer #1 and save results to `results/`.

---

## 7. Where Everything Is Saved

| Item                          | Folder                  | File name(s)                                      |
|-------------------------------|-------------------------|---------------------------------------------------|
| All experiment CSVs           | `results/`              | `*_seeds.csv`, `walkforward_all_datasets.csv`     |
| Main summary + TIG            | `results/`              | `temporal_effects_summary.csv`                    |
| Final paper table             | `results/`              | `paper_random_vs_walkforward_champions.csv`       |
| Per-dataset champion table    | `results/`              | `per_dataset_random_vs_walkforward_auc_champions.csv` |
| Kendall τ                     | `results/`              | `kendall_tau_protocols.csv`                       |
| All figures                   | `figures/`              | `walkforward_auc_main.png`, `walkforward_f1_main.png`, etc. |
| Reviewer analyses             | `results/`              | `hyperparam_sensitivity_apachejit.csv`, `smote_imbalance_apachejit.csv` |

---

## 8. Full Reproduction Time Estimate

- Data preparation: ~5 minutes
- Main experiments: 30–60 minutes on a modern CPU/GPU
- Analysis + figures: < 5 minutes
- Reviewer scripts: 15–30 minutes each

Total: **~1.5–2 hours** on a typical laptop.

---

## 9. Troubleshooting

- **ModuleNotFoundError**: Make sure you activated the virtual environment (`.venv\Scripts\activate`)
- **File not found**: Check that the three datasets are in the exact paths shown in Section 2
- **CUDA issues**: The code runs on CPU by default. If you have a GPU, it will be used automatically.
- **Memory error**: All datasets fit in < 4 GB RAM.

---

## Citation

If you use this code, please cite the paper:

> When Random Splits Lie: Temporal Inflation and Ranking Instability in Tabular Defect and Air-Quality Prediction

---
