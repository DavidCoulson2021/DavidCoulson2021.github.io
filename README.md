# DavidCoulson2021.github.io
From Raw Data to Robust Models: Taming Variance and Zero-Inflation.
# Signal in the Noise: Advanced Pipelines, Shrinkage, and Hurdle Models
> **From Hygiene to Hurdle Models:** Rigorous Machine Learning in Python.

An end-to-end Python portfolio mastering data hygiene, Bayesian shrinkage, and two-stage hurdle modeling to conquer small-sample variance, class imbalance, and zero-inflation.

---

## Portfolio Overview

This repository features a unified, rigorous data science pipeline broken down into three progressive phases:

### 1. Project 1: Consumer Activity Data Processing & EDA
* **Focus:** Foundational data integrity and exploratory analysis.
* **Key Techniques:** Median imputation for missing income features, outlier detection via box plots, and feature multicollinearity auditing using correlation heatmaps.

### 2. Project 2: Amazon Musical Instruments Review Ranking
* **Focus:** Small-sample statistical correction and variance control.
* **Key Techniques:** Applying Bayesian empirical smoothing to shrink high-variance, low-volume ratings toward a global mean, paired with regularized Ridge regression (lambda) to penalize coefficient variance (theta).

### 3. E-Commerce Conversion & Session Worth Prediction
* **Focus:** Advanced predictive modeling under class imbalance and zero-inflation.
* **Key Techniques:**
  * **Strict 3-Way Partitioning:** Split data into Train (60%), Validation (20%), and Test (20%) to eliminate data leakage.
  * **Threshold Optimization:** Programmatically tuned decision thresholds on the validation set to maximize F1-score against severe class imbalance.
  * **Architecture Benchmarking:** Compared Raw Baseline Ridge, Log-Transformed Ridge, and a Two-Stage Hurdle Model.
  * **Robust Metrics & Segments:** Utilized Mean Absolute Error (MAE) alongside RMSE for outlier-robust error tracking, backed by a Segmented Evaluation isolating zero-value browsing sessions from active-value buyer sessions.

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
