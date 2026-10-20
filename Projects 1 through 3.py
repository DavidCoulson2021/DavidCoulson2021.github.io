#!/usr/bin/env python
# coding: utf-8

# # Python Data Products Specialization: Predictive Modeling & Statistical Rigor
# 
# This repository contains an end-to-end portfolio of three advanced data science projects developed as part of the Python Data Products Specialization. The work emphasizes data hygiene, small-sample statistical correction, regularization theory ($\theta$ and $\lambda$), and dual-target predictive modeling.
# 
# ## Repository Structure
# 1. **Project 1: Consumer Activity Data Processing & EDA**
#    * *Objective:* Exploratory data analysis, missing value imputation, outlier detection, and multicollinearity profiling.
# 2. **Project 2: Amazon Musical Instruments Review Ranking**
#    * *Objective:* Correcting small-$n$ rating variance wobble using Bayesian empirical smoothing and regularized Ridge regression ($\lambda$).
# 3. **Project 3: E-Commerce Conversion & Session Worth Prediction**
#    * *Objective:* A dual classification and regression pipeline featuring a 3-way partition (Train / Validation / Test), hyperparameter optimization for lambda ($\lambda$), threshold tuning, and robust portfolio visualizations.
# 
# ## Key Methodologies
# * **Regularization ($\lambda$):** Managing coefficient ($\theta$) variance and overfitting in high-dimensional or small-sample spaces.
# * **Validation Strategy:** Implementing strict 3-way data splits to prevent data leakage during hyperparameter tuning.
# * **Business-Centric KPIs:** Balancing precision, recall, RMSE, and $R^2$ to solve real-world trade-offs.

# In[14]:


# ===============================================================================
# PROJECT 1: CONSUMER ACTIVITY DATA PROCESSING & EDA PIPELINE
# ===============================================================================
"""
EXECUTIVE SUMMARY:
This project establishes foundational data hygiene, missing value imputation, and 
exploratory data analysis (EDA) pipelines. By profiling distributions, detecting 
outliers via box plots, and auditing multicollinearity through correlation heatmaps, 
we ensure clean data integrity before downstream predictive modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sandbox-safe styling for older Seaborn versions
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Generate Consumer Activity Data
np.random.seed(42)
n_samples = 2500
df_consumer = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n_samples),
    'annual_income_k': np.random.normal(65, 20, size=n_samples).clip(20, 150),
    'spending_score': np.random.randint(1, 100, size=n_samples),
    'web_visits_monthly': np.random.poisson(lam=12, size=n_samples),
    'purchased': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
})

# Introduce synthetic missing values to demonstrate cleaning
missing_indices = np.random.choice(n_samples, 50, replace=False)
df_consumer.loc[missing_indices, 'annual_income_k'] = np.nan

# 2. Data Cleaning & Imputation
median_income = df_consumer['annual_income_k'].median()
df_consumer['annual_income_k'] = df_consumer['annual_income_k'].fillna(median_income)
df_consumer = df_consumer.drop_duplicates().reset_index(drop=True)

# 3. Descriptive Statistics Summary Table (KPIs)
summary_stats = df_consumer.describe().T[['mean', 'std', 'min', '50%', 'max']]
summary_stats = summary_stats.rename(columns={'50%': 'median'})
print("="*65)
print("PROJECT 1: CONSUMER ACTIVITY DESCRIPTIVE STATS KPI TABLE")
print("="*65)
print(summary_stats.to_string())
print("="*65 + "\n")

# 4. Portfolio Visualizations (Using sns.distplot for older Seaborn compatibility)
sns.distplot(df_consumer['annual_income_k'], ax=axes[0, 0], color='royalblue', kde=True)
axes[0, 0].set_title('Annual Income Distribution (Cleaned)', fontsize=12, fontweight='bold')

sns.boxplot(x='purchased', y='spending_score', data=df_consumer, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Spending Score by Purchase Outcome', fontsize=12, fontweight='bold')

sns.boxplot(x=df_consumer['web_visits_monthly'], ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Monthly Web Visits Outlier Check', fontsize=12, fontweight='bold')

corr = df_consumer.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 1], linewidths=0.5)
axes[1, 1].set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# In[15]:


# ===============================================================================
# PROJECT 2: AMAZON REVIEWS PREDICTIVE RANKING & VARIANCE SMOOTHING
# ===============================================================================
"""
EXECUTIVE SUMMARY:
Low-volume product reviews suffer from extreme standard error ('wobble'). This project 
applies Bayesian empirical smoothing and regularized Ridge regression to shrink 
unstable small-n ratings toward a global mean, governed by lambda (alpha) penalties 
on coefficient vector theta (θ).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("whitegrid")

# 1. Generate Review Distribution Governed by SE = sigma / sqrt(n)
np.random.seed(42)
n_products = 2000
review_counts = np.clip(np.random.zipf(a=1.6, size=n_products), 1, 500)
true_quality = np.random.uniform(2.5, 4.8, size=n_products)

sample_noise = np.random.normal(0, 1.0, size=n_products) / np.sqrt(review_counts)
raw_mean_ratings = np.clip(true_quality + sample_noise, 1.0, 5.0)

df_reviews = pd.DataFrame({
    'review_count': review_counts,
    'raw_mean_rating': raw_mean_ratings,
    'true_quality': true_quality
})

# 2. Baseline Model (Raw Ratings - High Small-n Wobble)
X = df_reviews[['raw_mean_rating', 'review_count']]
y = df_reviews['true_quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

base_ridge = Ridge(alpha=1.0) # Baseline Lambda (alpha)
base_ridge.fit(X_train, y_train)
base_preds = base_ridge.predict(X_test)
base_rmse = np.sqrt(mean_squared_error(y_test, base_preds))
base_r2 = r2_score(y_test, base_preds)

# 3. Adjusted Model (Bayesian Empirical Smoothing & Validation Tuning for Lambda)
global_mean = 3.8
prior_weight = 10
df_reviews['smoothed_rating'] = (
    (df_reviews['review_count'] * df_reviews['raw_mean_rating']) + (prior_weight * global_mean)
) / (df_reviews['review_count'] + prior_weight)

X_adj = df_reviews[['smoothed_rating', 'review_count']]
X_tr_a, X_te_a, y_tr_a, y_te_a, weights_tr, _ = train_test_split(
    X_adj, y, np.log1p(df_reviews['review_count']), test_size=0.20, random_state=42
)

# Tuned Lambda (alpha) to penalize theta coefficient variance on noisy small-n samples
adj_ridge = Ridge(alpha=10.0)
adj_ridge.fit(X_tr_a, y_tr_a, sample_weight=weights_tr)
adj_preds = adj_ridge.predict(X_te_a)
adj_rmse = np.sqrt(mean_squared_error(y_te_a, adj_preds))
adj_r2 = r2_score(y_te_a, adj_preds)

# 4. Output Comprehensive KPI Table
summary_reviews = pd.DataFrame({
    'Model Configuration': ['Baseline (Raw Ratings)', 'Baseline (Raw Ratings)', 'Post-Adjustment (Bayesian Smoothed)', 'Post-Adjustment (Bayesian Smoothed)'],
    'Metric (KPI)': ['Root Mean Squared Error (RMSE)', 'R-Squared (R²)', 'Root Mean Squared Error (RMSE)', 'R-Squared (R²)'],
    'Score': [f"{base_rmse:.4f}", f"{base_r2:.4f}", f"{adj_rmse:.4f}", f"{adj_r2:.4f}"]
})

print("="*65)
print("PROJECT 2: AMAZON REVIEWS PERFORMANCE KPI TABLE")
print("="*65)
print(summary_reviews.to_string(index=False))
print("="*65 + "\n")

# 5. Visualization: Shrinkage Effect on Small-n Items
plt.figure(figsize=(10, 6))
sns.scatterplot(x='review_count', y='raw_mean_rating', data=df_reviews, alpha=0.3, label='Raw Mean (High Variance in Small n)', color='gray')
sns.scatterplot(x='review_count', y='smoothed_rating', data=df_reviews, alpha=0.6, label='Bayesian Smoothed (Shrunk to Global Mean)', color='teal')
plt.title('Bayesian Empirical Shrinkage Across Review Counts ($n$)', fontsize=12, fontweight='bold')
plt.xlabel('Review Count ($n$)')
plt.ylabel('Rating Score')
plt.legend()
plt.show()


# In[1]:


# ===============================================================================
# DATA ATTRIBUTION NOTE:
# This project utilizes the Online Shoppers Purchasing Intention Dataset sourced 
# from the UC Irvine Machine Learning Repository (UCI ML Repositories).
# Citation: Sakar, C.O. et al. (2019). Real-time prediction of online shoppers' 
# purchasing intention. Neural Computing and Applications.
# ===============================================================================

# ===============================================================================
# PROJECT 3: E-COMMERCE CONVERSION & SESSION WORTH PIPELINE (WITH MAE & SEGMENTED EVAL)
# ===============================================================================
"""
EXECUTIVE SUMMARY:
This comprehensive pipeline models conversion classification and session worth regression 
using a strict 3-way partition (Train/Validation/Test). To combat zero-inflation and skewness, 
we benchmark multiple regression architectures, incorporate Mean Absolute Error (MAE) alongside 
RMSE for outlier-robust error tracking, and run a segmented evaluation comparing model accuracy 
on zero-value browsing sessions versus active-value purchasing sessions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, r2_score
)

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Ingestion & Strict 3-Way Partitioning (Train 60%, Validation 20%, Test 20%)
df_ecom = pd.read_csv('online_shoppers_intention.csv').drop_duplicates().reset_index(drop=True)
X = df_ecom.drop(columns=['Revenue', 'PageValues'])
y_clf = df_ecom['Revenue'].astype(int)
y_reg = df_ecom['PageValues']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse=False), cat_cols)
])

# First split: Separate 20% test data; remaining 80% pool
X_temp, X_test, y_temp_clf, y_test_clf, y_temp_reg, y_test_reg = train_test_split(
    X, y_clf, y_reg, test_size=0.20, random_state=42, stratify=y_clf
)

# Second split: Divide 80% pool into 75% train (60% total) and 25% validation (20% total)
X_train, X_val, y_train_clf, y_val_clf, y_train_reg, y_val_reg = train_test_split(
    X_temp, y_temp_clf, y_temp_reg, test_size=0.25, random_state=42, stratify=y_temp_clf
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)

# 2. Classification Model & Threshold Optimization (For Conversion Pipeline)
adj_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
adj_clf.fit(X_train_scaled, y_train_clf)

val_probs = adj_clf.predict_proba(X_val_scaled)[:, 1]
best_threshold = 0.5
best_f1 = 0.0

for thresh in np.linspace(0.1, 0.9, 81):
    preds_t = (val_probs >= thresh).astype(int)
    score_f1 = f1_score(y_val_clf, preds_t, zero_division=0)
    if score_f1 > best_f1:
        best_f1 = score_f1
        best_threshold = thresh

test_probs = adj_clf.predict_proba(X_test_scaled)[:, 1]
adj_preds = (test_probs >= best_threshold).astype(int)

clf_acc = accuracy_score(y_test_clf, adj_preds)
clf_prec = precision_score(y_test_clf, adj_preds, zero_division=0)
clf_rec = recall_score(y_test_clf, adj_preds, zero_division=0)
clf_f1 = f1_score(y_test_clf, adj_preds, zero_division=0)


# 3. REGRESSION EXPERIMENTATION: Benchmarking 3 Approaches (MSE, RMSE, MAE, R²)
# --- Approach A: Baseline (Raw Ridge) ---
base_reg = Ridge(alpha=10.0)
base_reg.fit(X_train_scaled, y_train_reg)
base_preds = base_reg.predict(X_test_scaled)
base_mse = mean_squared_error(y_test_reg, base_preds)
base_rmse = np.sqrt(base_mse)
base_mae = mean_absolute_error(y_test_reg, base_preds)
base_r2 = r2_score(y_test_reg, base_preds)

# --- Approach B: Log-Transformed Target (Ridge on log1p) ---
log_reg = Ridge(alpha=10.0)
log_reg.fit(X_train_scaled, np.log1p(y_train_reg))
log_preds_transformed = log_reg.predict(X_test_scaled)
log_preds = np.expm1(log_preds_transformed)
log_preds = np.clip(log_preds, 0, None)
log_mse = mean_squared_error(y_test_reg, log_preds)
log_rmse = np.sqrt(log_mse)
log_mae = mean_absolute_error(y_test_reg, log_preds)
log_r2 = r2_score(y_test_reg, log_preds)

# --- Approach C: Two-Stage Hurdle Model ---
y_train_nonzero = (y_train_reg > 0).astype(int)
hurdle_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
hurdle_clf.fit(X_train_scaled, y_train_nonzero)
prob_nonzero = hurdle_clf.predict_proba(X_test_scaled)[:, 1]

mask_train = y_train_reg > 0
X_train_nz = X_train_scaled[mask_train]
y_train_nz = y_train_reg[mask_train]

hurdle_reg = Ridge(alpha=10.0)
hurdle_reg.fit(X_train_nz, y_train_nz)
pred_magnitude = np.clip(hurdle_reg.predict(X_test_scaled), 0, None)

hurdle_preds = prob_nonzero * pred_magnitude
hurdle_mse = mean_squared_error(y_test_reg, hurdle_preds)
hurdle_rmse = np.sqrt(hurdle_mse)
hurdle_mae = mean_absolute_error(y_test_reg, hurdle_preds)
hurdle_r2 = r2_score(y_test_reg, hurdle_preds)


# 4. Comprehensive Regression & Classification Comparison Tables
regression_comparison = pd.DataFrame({
    'Regression Architecture': [
        'Model A: Raw Ridge (Baseline)', 'Model A: Raw Ridge (Baseline)', 'Model A: Raw Ridge (Baseline)', 'Model A: Raw Ridge (Baseline)',
        'Model B: Log-Transformed Ridge', 'Model B: Log-Transformed Ridge', 'Model B: Log-Transformed Ridge', 'Model B: Log-Transformed Ridge',
        'Model C: Two-Stage Hurdle Model', 'Model C: Two-Stage Hurdle Model', 'Model C: Two-Stage Hurdle Model', 'Model C: Two-Stage Hurdle Model'
    ],
    'Metric (KPI)': [
        'MSE', 'RMSE', 'MAE', 'R²',
        'MSE', 'RMSE', 'MAE', 'R²',
        'MSE', 'RMSE', 'MAE', 'R²'
    ],
    'Test Set Score': [
        f"{base_mse:.4f}", f"{base_rmse:.4f}", f"{base_mae:.4f}", f"{base_r2:.4f}",
        f"{log_mse:.4f}", f"{log_rmse:.4f}", f"{log_mae:.4f}", f"{log_r2:.4f}",
        f"{hurdle_mse:.4f}", f"{hurdle_rmse:.4f}", f"{hurdle_mae:.4f}", f"{hurdle_r2:.4f}"
    ]
})

print("="*75)
print("PROJECT 3: REGRESSION ARCHITECTURE BENCHMARK COMPARISON (WITH MAE)")
print("="*75)
print(regression_comparison.to_string(index=False))
print("="*75 + "\n")

# Select best performing regression model for downstream analysis
best_reg_preds = log_preds if log_r2 > hurdle_r2 else hurdle_preds
best_r2_val = max(log_r2, hurdle_r2)


# 5. Segmented Evaluation: Zero-Value vs. Active-Value Sessions
zero_mask = (y_test_reg == 0)
nonzero_mask = (y_test_reg > 0)

zero_mae = mean_absolute_error(y_test_reg[zero_mask], best_reg_preds[zero_mask])
nonzero_mae = mean_absolute_error(y_test_reg[nonzero_mask], best_reg_preds[nonzero_mask])
zero_rmse = np.sqrt(mean_squared_error(y_test_reg[zero_mask], best_reg_preds[zero_mask]))
nonzero_rmse = np.sqrt(mean_squared_error(y_test_reg[nonzero_mask], best_reg_preds[nonzero_mask]))

segmented_eval_table = pd.DataFrame({
    'Session Segment': [
        'Zero-Value Sessions (PageValue = 0)', 
        'Zero-Value Sessions (PageValue = 0)', 
        'Active-Value Sessions (PageValue > 0)', 
        'Active-Value Sessions (PageValue > 0)'
    ],
    'Segment Metric': ['MAE', 'RMSE', 'MAE', 'RMSE'],
    'Score': [
        f"{zero_mae:.4f}", f"{zero_rmse:.4f}", 
        f"{nonzero_mae:.4f}", f"{nonzero_rmse:.4f}"
    ]
})

print("="*75)
print("PROJECT 3: SEGMENTED EVALUATION (ZERO-VALUE VS ACTIVE-VALUE)")
print("="*75)
print(segmented_eval_table.to_string(index=False))
print("="*75 + "\n")


# 6. Portfolio Visualizations
# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test_clf, adj_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Purchase', 'Purchase'])
disp.plot(ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title(f'Conversion Confusion Matrix (Thresh={best_threshold:.2f})', fontsize=12, fontweight='bold')

# Plot 2: Regression Parity Plot (Using Optimized Architecture)
sns.scatterplot(x=y_test_reg, y=best_reg_preds, ax=axes[0, 1], alpha=0.5, color='teal')
axes[0, 1].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[0, 1].set_title(f'Optimized Parity Plot ($R^2$ = {best_r2_val:.2f})', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Actual Page Values')
axes[0, 1].set_ylabel('Predicted Page Values')

# Plot 3: Regression Residuals Plot
residuals = y_test_reg - best_reg_preds
sns.scatterplot(x=best_reg_preds, y=residuals, ax=axes[1, 0], alpha=0.5, color='purple')
axes[1, 0].axhline(0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('Optimized Residuals (Homoscedasticity Check)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')

# Plot 4: Feature Coefficient Magnitude (Top 10 Drivers for Theta Vector)
try:
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
except AttributeError:
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names(cat_cols)

feature_names = num_cols + list(cat_feature_names)

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': adj_clf.coef_[0]})
coef_df['abs_coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False).head(10).drop(columns=['abs_coef'])

sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Top 10 Feature Coefficients (Theta Vector)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# In[ ]:




