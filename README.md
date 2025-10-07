# Global Development Analysis

A collection of data analysis and machine learning projects using socioeconomic indicators from the World Bank.

## Overview

This repository is dedicated to exploring global socioeconomic data to uncover patterns, group countries by similarities, and analyze development trends over time. The primary data source for these projects is the [World Development Indicators (WDI)](https://datacatalog.worldbank.org/search/dataset/0037712) dataset from the World Bank.

---

## Current Projects

### 1. WDI 2022: A Snapshot of Global Development

This project provides a comprehensive analysis of the world in the year 2022. The workflow involves:
* **Data Processing:** Downloading, cleaning, and preprocessing the complete WDI dataset to create a focused and analysis-ready dataset for 2022.
* **Unsupervised Learning:** Applying K-Means and HDBSCAN clustering algorithms to group countries based on their development profiles.
* **Analysis:** Interpreting the resulting clusters to understand the different tiers of socioeconomic development across the globe.

Future work will expand this project to include other machine learning tasks based on this same dataset.

**[Click here to explore the WDI 2022 project in detail.](./WDI-2022/)**

---

## Future Work (Roadmap)

This repository is an ongoing effort. Future projects and enhancements planned include:

-   **Supervised Learning:** Building predictive models, such as:
    -   A regression model to predict a country's life expectancy based on other socioeconomic factors.
    -   A classification model to categorize countries into development tiers.
-   **Time-Series Analysis:** Expanding the analysis to cover the evolution of key indicators (e.g., GDP per capita, life expectancy) from 2000 to the present, identifying which countries have shown the most significant progress.
-   **Interactive Dashboard:** Developing an interactive web dashboard (using Streamlit or Dash) to allow users to explore the data and visualizations dynamically.
-   **Feature Enrichment:** Expanding the analysis by incorporating diverse external datasets, including unstructured data sources from fields like Computer Vision and Natural Language Processing (NLP).

---

## Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Plotly, Seaborn, Matplotlib
* **Environment:** Jupyter Notebook

---

## Contact

* **Author:** Rafael Masson
* **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/rafael-masson/)
* **GitHub:** [GitHub Profile](https://github.com/rafacmasson)