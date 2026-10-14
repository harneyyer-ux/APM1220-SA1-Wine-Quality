# APM1220 Summative Assessment 1: Wine Quality Analysis

**Course:** APM1220 – Applied Multivariate Data Analysis  
**Assessment:** Summative Assessment 1  
**Student:** Harneyyer Clores and Maria Abad  
**Section:** Section 2

## Project Description

This project analyzes the physicochemical properties of red and white Vinho Verde wines using exploratory data analysis (EDA), multivariate analysis of variance (MANOVA), principal component analysis (PCA), and exploratory factor analysis (EFA).

## Dataset

The repository contains two datasets from the UCI Machine Learning Repository:

- `data/winequality-red.csv` — red wine observations.
- `data/winequality-white.csv` — white wine observations.

**Original source:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016

## Repository Structure

- `code/SA1_Wine_Analysis.py` — Python script for the analysis.
- `data/` — input datasets.
- `output/figures/` — generated plots and visualizations.
- `output/tables/` — generated statistical tables.
- `output/results.json` — summary of analysis results.
- `SA1_Wine_Quality_Report.pdf` — final statistical report.

## Analysis Overview

The Python script performs data preparation and exploratory analysis, evaluates differences between red and white wines using MANOVA, summarizes the variables using PCA, and explores latent factor structure using EFA.

## Reproducibility

Run the analysis from the repository root directory.

**1. Install the required packages**

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels pingouin factor_analyzer
```

**2. Run the analysis**

```bash
python code/SA1_Wine_Analysis.py
```

The script reads the red and white wine datasets from the `data/` folder and saves generated figures, statistical tables, and summary results in the `output/` directory.

The script requires Python and the packages listed above to be installed. Some package versions may require compatibility adjustments.

