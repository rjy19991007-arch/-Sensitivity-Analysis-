# -Sensitivity-Analysis-
Sensitivity analysis of the ERFR index with respect to indicator weights

# Sensitivity Analysis of ERFR

## 📌 Description
This repository contains the code and data used for the sensitivity analysis of the Extreme Rainfall-Flood Resilience (ERFR) index.

The analysis is based on a Monte Carlo simulation framework to evaluate the robustness of ERFR under weight perturbations of capacity and risk indicators.

---

## 📂 Repository Structure

- `Sensitivity Analysis.py`  
  Main Python script for sensitivity analysis.

- `normal.csv`  
  Input dataset (already normalized, including both capacity and risk indicators).

- `Sensitivity_MC.csv`  
  Monte Carlo simulation results (Spearman correlation and Top 20% overlap).

- `Table_S1.csv`  
  Summary statistics used in the manuscript (Table S1).

---

## ⚙️ Requirements

Python 3.x with the following packages:

- pandas
- numpy
- scipy

Install using:

```bash
pip install pandas numpy scipy
