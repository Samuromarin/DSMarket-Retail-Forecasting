# DSMarket — End-to-End Machine Learning Project
*Lucía Zariquiey · Samuel Romero*

---

## Overview

DSMarket is a supermarket chain in the process of digital transformation. This project simulates a real professional environment where we take on the role of a data science team tackling four high-priority business initiatives from scratch — from raw data to a deployable MLOps solution.

The dataset is based on the structure of the **M5 Forecasting Competition** (Kaggle, 2020), adapted to a fictional retail environment. It covers **46M+ daily sales records** across 10 stores in 3 US cities (New York, Boston, Philadelphia), spanning January 2011 – April 2016.

---

## Project Structure

```
├── preprocesamiento.ipynb       # Data loading, cleaning and feature engineering
├── eda.ipynb                    # Exploratory data analysis
├── clustering_productos.ipynb   # Product segmentation (ABC/XYZ + K-Means + SHAP)
├── clustering_tiendas.ipynb     # Store segmentation (K-Means)
└── modelizacion.ipynb           # Sales forecasting model (XGBoost + skforecast)
```

> **Note:** Interactive Plotly charts are not rendered on GitHub. Open the notebooks in Google Colab to view all visualizations.

---

## Notebooks


| Notebook | |
|---|---|
| Preprocesamiento | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samuromarin/DSMarket-Retail-Forecasting/blob/main/01_preprocesamiento.ipynb) |
| EDA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samuromarin/DSMarket-Retail-Forecasting/blob/main/02_eda.ipynb) |
| Clustering Productos | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samuromarin/DSMarket-Retail-Forecasting/blob/main/03_clustering_productos.ipynb) |
| Clustering Tiendas | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samuromarin/DSMarket-Retail-Forecasting/blob/main/04_clustering_tiendas.ipynb) |
| Modelización | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samuromarin/DSMarket-Retail-Forecasting/blob/main/05_modelizacion.ipynb) |
---

## Tasks

### 1. Exploratory Data Analysis
Comprehensive analysis of 46M+ sales records across stores, cities and product categories. Includes price elasticity analysis, trend detection by store, and a 5-tab **interactive Power BI dashboard** for executive reporting.

### 2. Product Clustering
Two complementary segmentation approaches:
- **ABC/XYZ classification** based on economic importance and demand regularity
- **K-Means clustering** (K=4) with feature engineering, correlation analysis and HDBSCAN comparison. Clusters interpreted via **SHAP values** (XGBoost) and visualized with **UMAP**

| Cluster | Name | Profile |
|---|---|---|
| 0 | Staple Premium | High price, stable and frequent demand |
| 1 | Locomotives | Maximum volume, minimum variability, price-elastic |
| 2 | Opportunists | Mid-low price, irregular demand |
| 3 | Niche | High price, very sporadic sales |

### 3. Store Clustering
K-Means segmentation (K=3) of the 10 stores based on sales volume, demand variability and category composition. Results validated with UMAP visualization.

| Cluster | Profile | Stores |
|---|---|---|
| 0 | High volume, stable demand | NYC_1, NYC_3 |
| 1 | Irregular demand, lower SUPERMARKET share | NYC_2, PHI_1, PHI_2 |
| 2 | Medium profile, stable demand | BOS_1, BOS_2, BOS_3, NYC_4, PHI_3 |

### 4. Sales Forecasting
28-day demand forecast at product × store level using **XGBoost via skforecast** (`ForecasterRecursiveMultiSeries`).

**Key design decisions:**
- **Tweedie objective** (`reg:tweedie`) to handle intermittent demand with many zeros
- **Lag-safe features** — rolling means and snapshots with minimum 28-day offset to prevent data leakage
- **Target encoding** calculated exclusively on training data
- **Bayesian hyperparameter optimization** with Optuna (30 trials)
- **Bottom-up aggregation** — predictions generated at item × store level and summed hierarchically

**Results vs historical benchmark:**

| Level | WMAPE Benchmark | WMAPE Model | Improvement |
|---|---|---|---|
| Product × Store | 99.6% | 67.9% | 31.8% |
| Category × Store | 16.3% | 9.7% | 40.4% |
| Store | 14.3% | 8.4% | 41.3% |
| City | 13.1% | 6.6% | 49.4% |
| **Total DSMarket** | **12.6%** | **5.7%** | **54.7%** |

### 5. Store Replenishment Solution
Full MLOps proposal converting forecasts into operational stock replenishment decisions:
- Safety stock policies differentiated by ABC/XYZ segment
- REST API architecture with Docker containerization
- MLflow for model versioning and retraining monitoring
- Controlled pilot design using difference-in-differences methodology
- Functional web interface prototype for business validation

---

## Tech Stack

`Python` · `XGBoost` · `skforecast` · `Optuna` · `SHAP` · `UMAP` · `scikit-learn` · `Plotly` · `Power BI` · `Docker` · `MLflow`

---

## Data

The dataset follows the structure of the M5 Forecasting Competition and is not included in this repository due to size constraints (~18 GB in memory). It consists of three source files:

- `item_sales.csv` — daily unit sales by product and store
- `item_prices.csv` — weekly prices by product and store  
- `daily_calendar_with_events.csv` — dates, weekdays and commercial events
