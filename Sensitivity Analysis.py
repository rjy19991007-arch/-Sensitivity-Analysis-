import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# =========================
# 1. 读取数据
# =========================
file = r"E:\K\城市群\修改\normal.csv"
df = pd.read_csv(file, encoding="utf-8-sig")

# =========================
# 2. 指标定义
# 注意：Normal表已经完成正负向归一化，所以这里直接等权平均
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
    "ERF",      # 实际是 ERF Index
    "Population density",
    "BUA proportion",
    "Road network density"
]

# 检查字段是否存在
all_vars = capacity_vars + risk_vars
missing = [v for v in all_vars if v not in df.columns]
if missing:
    raise ValueError(f"以下字段在 normal.csv 中不存在，请检查列名: {missing}")

# 转为数值型
for col in all_vars:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=all_vars).copy()

# =========================
# 3. Baseline：等权计算
# =========================
Capacity_base = df[capacity_vars].mean(axis=1)
Risk_base = df[risk_vars].mean(axis=1)

ERFR_base = Capacity_base / (1 + Risk_base)

# =========================
# 4. Monte Carlo 敏感性分析
# =========================
np.random.seed(123)

n_sim = 100
results = []

n = len(df)
top_n = int(n * 0.2)

base_rank = ERFR_base.rank(ascending=False)
top_base = set(base_rank.nsmallest(top_n).index)

for i in range(n_sim):

    # Capacity内部指标权重扰动 ±20%
    w_capacity = np.random.uniform(0.8, 1.2, len(capacity_vars))
    w_capacity = w_capacity / w_capacity.sum()

    # Risk内部指标权重扰动 ±20%
    w_risk = np.random.uniform(0.8, 1.2, len(risk_vars))
    w_risk = w_risk / w_risk.sum()

    Capacity_new = (df[capacity_vars] * w_capacity).sum(axis=1)
    Risk_new = (df[risk_vars] * w_risk).sum(axis=1)

    ERFR_new = Capacity_new / (1 + Risk_new)

    # Spearman rank correlation
    rho, p_value = spearmanr(ERFR_base, ERFR_new)

    # Top 20% overlap
    new_rank = ERFR_new.rank(ascending=False)
    top_new = set(new_rank.nsmallest(top_n).index)
    top_overlap = len(top_base & top_new) / top_n

    results.append({
        "Sim": i + 1,
        "Spearman_rho": rho,
        "Spearman_p": p_value,
        "Top20_overlap": top_overlap
    })

res_df = pd.DataFrame(results)

# =========================
# 5. 输出模拟结果
# =========================
out_mc = r"E:\K\城市群\修改\Sensitivity_MC.csv"
res_df.to_csv(out_mc, index=False, encoding="utf-8-sig")

# =========================
# 6. 输出 Table S1
# =========================
summary = pd.DataFrame({
    "Metric": ["Spearman ρ", "Top 20% overlap"],
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

out_table = r"E:\K\城市群\修改\Table_S1.csv"
summary.to_csv(out_table, index=False, encoding="utf-8-sig")

print("Sensitivity analysis completed.")
print("Monte Carlo results saved to:", out_mc)
print("Summary table saved to:", out_table)
print(summary)