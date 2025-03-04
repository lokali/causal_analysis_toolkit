# Causal Analysis Toolkit

## Overview

The **Causal Analysis Toolkit** provides a comprehensive pipeline for:

* **Dataset Preprocessing**
* **Exploratory Data Analysis (EDA)** : Histograms and scatterplots
* **Pairwise Dependence Analysis** : Using Kernel-based Conditional Independence (KCI), Randomized Conditional Independence Test (RCIT), and HSIC tests
* **Causal Discovery** : Using the **PC algorithm** from `causal-learn`

This repository aims to provide researchers with an easy-to-use framework for **causal structure learning** from observational data.

---

## Features

✅  **Preprocessing** : Handles dataset loading and cleaning

✅  **Visualizations** : Generates histograms and pairwise scatterplots

✅  **Statistical Independence Tests** : Computes CI p-values for multiple tests

✅  **Causal Structure Learning** : Implements the PC algorithm to infer causal graphs

---

## Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/lokali/causal_analysis_toolkit.git
cd causal_analysis_toolkit
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1️⃣ Run Exploratory Data Analysis

```bash
jupyter notebook 01_data_analysis.ipynb
```

This notebook will:

* Load and visualize the dataset
* Generate **histograms** and **scatterplots**
* Compute **KCI, RCIT, and HSIC dependence matrices**

### 2️⃣ Perform Causal Discovery

Run `run_pc` in `01_data_analysis.ipynb` and or directly run the `pc` algorithm from causal-learn:

```python
from utils import run_pc
cg, path = run_pc(data=df.values, alpha=0.01, indep_test='fisherz', label=df.columns.values)

or 

from causallearn.search.ConstraintBased.PC import pc
cg = pc(df.values, alpha=0.01, indep_test="fisherz")
```

This will **infer the causal structure** and plot the causal graph.

---

## File Structure

```
causal_analysis_toolkit/
│── 01_data_analysis.ipynb   # Jupyter Notebook for data analysis
│── utils.py                 # Utility functions for CI tests & visualization
│── requirements.txt          # List of dependencies
│── README.md                # Documentation
│── results/                 # Output directory for figures & logs
```

---

## Contact

For questions or feedback, feel free to contact me via  **[Longkang.Li@mbzuai.ac.ae](mailto:your.email@example.com)**.
