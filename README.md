# Global Development Analysis

A collection of data analysis and machine learning projects using socioeconomic indicators from the World Bank.

## Overview

This repository is dedicated to exploring global socioeconomic data to uncover patterns, group countries by similarities, and analyze development trends over time. The primary data source for these projects is the [World Development Indicators (WDI)](https://datacatalog.worldbank.org/search/dataset/0037712) dataset from the World Bank.

---

## Current Projects

### 1. WDI 2022: Global Development & Predictive Modeling

This project provides a comprehensive analysis and machine learning workflow on global development in 2022:
* **Data Processing:** Downloading, cleaning, and preprocessing the complete WDI dataset to create a focused, 16-indicator analysis-ready dataset for 2022.
* **Unsupervised Learning:** Grouping countries using K-Means and HDBSCAN clustering algorithms to uncover organic development groups.
* **Supervised Learning (ML):** 
    * **Life Expectancy Prediction (Regression):** An optimized Random Forest Regressor ($R^2 = 0.82$, MAE = 2.56 years) predicting longevity based on socioeconomic factors, compared with regularized Lasso/Ridge linear regressions.
    * **Development Tier Classification:** An optimized Random Forest Classifier (95.45% accuracy) and simplified Decision Tree (95.45% accuracy) predicting a country's development cluster.
* **Interactive Dashboard & Simulator:** A Streamlit web application featuring a **Real-Time Predictive Simulator** that runs live machine learning predictions and overlays them against actual development tiers using gauge and comparison bar charts.

**[Explore the WDI 2022 project in detail](./WDI-2022/)** | **[View Interactive Dashboard](https://wdi-global-clusters.streamlit.app/)**

---

## Future Work (Roadmap)

This repository is an ongoing effort. Future projects and enhancements planned include:

-   **Time-Series Analysis:** Expanding the analysis to cover the evolution of key indicators (e.g., GDP per capita, life expectancy) from 2000 to the present, identifying which countries have shown the most significant progress.
-   **Feature Enrichment:** Expanding the analysis by incorporating diverse external datasets, including unstructured data sources from fields like Computer Vision and Natural Language Processing (NLP).

---

## Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Plotly, Seaborn, Matplotlib, Joblib, Streamlit
* **Environment:** Jupyter Notebook

---

## Contact

* **Author:** Rafael Masson
* **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/rafael-masson/)
* **GitHub:** [GitHub Profile](https://github.com/rafacmasson)