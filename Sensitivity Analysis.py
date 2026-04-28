import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# =========================
# 1. Load data
# =========================
# NOTE: Make sure the file is in the same directory as this script
file = "normal.csv"
df = pd.read_csv(file, encoding="utf-8-sig")

# =========================
# 2. Define indicators
# NOTE:
# The input dataset (normal.csv) has already been normalized
# (including positive and negative normalization),
# therefore equal-weight aggregation is directly applied.
# =========================

# Capacity indicators
capacity_vars = [
    "Drainage network density",
    "Green space ratio",
    "Soil permeability",
    "NDVI",
    "AI",
    "CONTAG",
    "AREA_MN",
    "PD"
]

# Risk indicators
risk_vars = [
    "ERF",  # Extreme Rainfall Index
    "Population density",
    "BUA proportion",
    "Road network density"
]

# Check if all required columns exist
all_vars = capacity_vars + risk_vars
missing = [v for v in all_vars if v not in df.columns]
if missing:
    raise ValueError(f"The following variables are missing in normal.csv: {missing}")

# Convert to numeric format
for col in all_vars:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values
df = df.dropna(subset=all_vars).copy()

# =========================
# 3. Baseline calculation (equal weights)
# =========================
Capacity_base = df[capacity_vars].mean(axis=1)
Risk_base = df[risk_vars].mean(axis=1)

# ERFR calculation
ERFR_base = Capacity_base / (1 + Risk_base)

# =========================
# 4. Monte Carlo sensitivity analysis
# =========================
np.random.seed(123)

n_sim = 100
results = []

n = len(df)
top_n = int(n * 0.2)

# Baseline ranking
base_rank = ERFR_base.rank(ascending=False)
top_base = set(base_rank.nsmallest(top_n).index)

for i in range(n_sim):

    # Random perturbation of capacity weights (±20%)
    w_capacity = np.random.uniform(0.8, 1.2, len(capacity_vars))
    w_capacity = w_capacity / w_capacity.sum()

    # Random perturbation of risk weights (±20%)
    w_risk = np.random.uniform(0.8, 1.2, len(risk_vars))
    w_risk = w_risk / w_risk.sum()

    # Recalculate Capacity and Risk
    Capacity_new = (df[capacity_vars] * w_capacity).sum(axis=1)
    Risk_new = (df[risk_vars] * w_risk).sum(axis=1)

    # Recalculate ERFR
    ERFR_new = Capacity_new / (1 + Risk_new)

    # Spearman rank correlation
    rho, p_value = spearmanr(ERFR_base, ERFR_new)

    # Top 20% overlap
    new_rank = ERFR_new.rank(ascending=False)
    top_new = set(new_rank.nsmallest(top_n).index)
    top_overlap = len(top_base & top_new) / top_n

    results.append({
        "Simulation": i + 1,
        "Spearman_rho": rho,
        "Spearman_p": p_value,
        "Top20_overlap": top_overlap
    })

res_df = pd.DataFrame(results)

# =========================
# 5. Export Monte Carlo results
# =========================
out_mc = "Sensitivity_MC.csv"
res_df.to_csv(out_mc, index=False, encoding="utf-8-sig")

# =========================
# 6. Export summary table (Table S1)
# =========================
summary = pd.DataFrame({
    "Metric": ["Spearman rho", "Top 20% overlap"],
    "Mean": [
        res_df["Spearman_rho"].mean(),
        res_df["Top20_overlap"].mean()
    ],
    "Min": [
        res_df["Spearman_rho"].min(),
        res_df["Top20_overlap"].min()
    ],
    "Max": [
        res_df["Spearman_rho"].max(),
        res_df["Top20_overlap"].max()
    ]
})

out_table = "Table_S1.csv"
summary.to_csv(out_table, index=False, encoding="utf-8-sig")

# =========================
# 7. Print outputs
# =========================
print("Sensitivity analysis completed successfully.")
print(f"Monte Carlo results saved to: {out_mc}")
print(f"Summary table saved to: {out_table}")
print(summary)
