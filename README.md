# DavidCoulson2021.github.io
From Raw Data to Robust Models: Taming Variance and Zero-Inflation.
# Signal in the Noise: Advanced Pipelines, Shrinkage, and Hurdle Models
> **From Hygiene to Hurdle Models:** Rigorous Machine Learning in Python.

An end-to-end Python portfolio mastering data hygiene, Bayesian shrinkage, and two-stage hurdle modeling to conquer small-sample variance, class imbalance, and zero-inflation.

### 📂 Project Code
* **[View Full Python Pipeline Script (416 lines)](https://github.com/DavidCoulson2021/DavidCoulson2021.github.io/blob/main/your-filename.py)** — Explores data hygiene, Bayesian empirical smoothing, regularized Ridge regression, and the two-stage hurdle model.
* 
---

## Portfolio Overview

This repository features a unified, rigorous data science pipeline broken down into three progressive phases:

### 1. Project 1: Consumer Activity Data Processing & EDA
* **Focus:** Foundational data integrity and exploratory analysis.
* **Key Techniques:** Median imputation for missing income features, outlier detection via box plots, and feature multicollinearity auditing using correlation heatmaps.
<img width="1000" height="712" alt="image" src="https://github.com/user-attachments/assets/c980ae4c-b776-4de5-bac3-9a204283381f" />

### 2. Project 2: Amazon Musical Instruments Review Ranking
* **Focus:** Small-sample statistical correction and variance control.
* **Key Techniques:** Applying Bayesian empirical smoothing to shrink high-variance, low-volume ratings toward a global mean, paired with regularized Ridge regression (lambda) to penalize coefficient variance (theta).
<img width="609" height="392" alt="image" src="https://github.com/user-attachments/assets/9d85016f-275c-4cb3-9f9d-d31647895b23" />


### 3. E-Commerce Conversion & Session Worth Prediction
* **Focus:** Advanced predictive modeling under class imbalance and zero-inflation.
* **Key Techniques:**
  * **Strict 3-Way Partitioning:** Split data into Train (60%), Validation (20%), and Test (20%) to eliminate data leakage.
  * **Threshold Optimization:** Programmatically tuned decision thresholds on the validation set to maximize F1-score against severe class imbalance.
  * **Architecture Benchmarking:** Compared Raw Baseline Ridge, Log-Transformed Ridge, and a Two-Stage Hurdle Model.
  * **Robust Metrics & Segments:** Utilized Mean Absolute Error (MAE) alongside RMSE for outlier-robust error tracking, backed by a Segmented Evaluation isolating zero-value browsing sessions from active-value buyer sessions.
<img width="1000" height="856" alt="image" src="https://github.com/user-attachments/assets/ef3e1ae8-9670-4e33-bf54-ff5d29ef1e3a" />

---

## Technical Stack
* **Languages:** Python, SQL
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Core Concepts:** Regularization (Lambda/Theta tuning), 3-way cross-validation, hypermarket threshold tuning, MAE, and hurdle modeling.

---
*Created as part of the Python Data Products Specialization.*

## Data Citations & Attribution
* **Online Shoppers Purchasing Intention Dataset:** Used in Project 3, sourced from the UC Irvine Machine Learning Repository. 
  * *Creators:* C. Sahin and U. Sever (or the original publication: Sakar, C.O., Polat, S.O., Katircioglu, M., & Kiliç, R. (2019). Real-time prediction of online shoppers' purchasing intention. *Neural Computing and Applications*).
