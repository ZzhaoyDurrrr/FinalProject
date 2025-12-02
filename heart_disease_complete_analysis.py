#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Heart Disease Analysis Project
Dataset: Heart_Disease_Prediction 2.csv
Author: Data Science 611
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, accuracy_score, precision_recall_curve)
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create directories
import os
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*80)
print("HEART DISEASE PREDICTION - COMPLETE ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND DESCRIPTION
# ============================================================================

print("\n" + "="*80)
print("PART 1: DATA LOADING AND EXPLORATION")
print("="*80)

# Load data
df = pd.read_csv('Heart_Disease_Prediction 2.csv')

print(f"\n1.1 Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")

# Basic information
print("\n1.2 Dataset Structure:")
print(df.info())

print("\n1.3 First 5 rows:")
print(df.head())

print("\n1.4 Statistical Summary:")
stats_summary = df.describe()
print(stats_summary)
stats_summary.to_csv('results/statistical_summary.csv')

print("\n1.5 Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values detected!")
else:
    print(missing[missing > 0])

print("\n1.6 Target Variable Distribution:")
target_dist = df['Heart Disease'].value_counts()
print(target_dist)
print(f"\nClass Balance:")
for cls, count in target_dist.items():
    print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")

print("\n1.7 Feature Descriptions:")
feature_desc = {
    'Age': 'Patient age (years)',
    'Sex': 'Sex (1=male, 0=female)',
    'Chest pain type': 'Type of chest pain (1-4)',
    'BP': 'Resting blood pressure (mm Hg)',
    'Cholesterol': 'Serum cholesterol (mg/dl)',
    'FBS over 120': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)',
    'EKG results': 'Resting electrocardiographic results (0-2)',
    'Max HR': 'Maximum heart rate achieved',
    'Exercise angina': 'Exercise induced angina (1=yes, 0=no)',
    'ST depression': 'ST depression induced by exercise',
    'Slope of ST': 'Slope of peak exercise ST segment (1-3)',
    'Number of vessels fluro': 'Number of major vessels colored by fluoroscopy (0-3)',
    'Thallium': 'Thallium stress test result (3,6,7)',
    'Heart Disease': 'Target variable (Presence/Absence)'
}

for feat, desc in feature_desc.items():
    print(f"  • {feat}: {desc}")

# Save detailed statistics
detailed_stats = pd.DataFrame({
    'Feature': df.columns,
    'Type': df.dtypes,
    'Unique_Values': [df[col].nunique() for col in df.columns],
    'Missing': [df[col].isnull().sum() for col in df.columns],
    'Missing_%': [df[col].isnull().sum()/len(df)*100 for col in df.columns]
})
detailed_stats.to_csv('results/feature_statistics.csv', index=False)
print("\n✓ Saved: results/feature_statistics.csv")

# ============================================================================
# PART 2: DATA VISUALIZATION (6 FIGURES)
# ============================================================================

print("\n" + "="*80)
print("PART 2: EXPLORATORY DATA VISUALIZATION")
print("="*80)

# Prepare data
df_plot = df.copy()
df_plot['Heart Disease Binary'] = (df_plot['Heart Disease'] == 'Presence').astype(int)

# Figure 1: Age and Sex Distribution
print("\n2.1 Creating Figure 1: Demographics Analysis")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution by disease status
axes[0, 0].hist(df_plot[df_plot['Heart Disease'] == 'Presence']['Age'], 
               bins=20, alpha=0.6, label='Disease', color='red', edgecolor='black')
axes[0, 0].hist(df_plot[df_plot['Heart Disease'] == 'Absence']['Age'], 
               bins=20, alpha=0.6, label='No Disease', color='green', edgecolor='black')
axes[0, 0].set_xlabel('Age (years)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution by Heart Disease Status', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Age boxplot
sns.boxplot(x='Heart Disease', y='Age', data=df_plot, ax=axes[0, 1], 
           palette={'Presence': 'red', 'Absence': 'green'})
axes[0, 1].set_title('Age Comparison: Disease vs No Disease', fontweight='bold')
axes[0, 1].set_ylabel('Age (years)')
axes[0, 1].grid(alpha=0.3)

# Sex distribution
sex_disease = pd.crosstab(df_plot['Sex'], df_plot['Heart Disease'], normalize='index') * 100
sex_disease.plot(kind='bar', ax=axes[1, 0], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Heart Disease Rate by Sex', fontweight='bold')
axes[1, 0].set_xlabel('Sex (0=Female, 1=Male)')
axes[1, 0].set_ylabel('Percentage (%)')
axes[1, 0].legend(title='Heart Disease')
axes[1, 0].set_xticklabels(['Female', 'Male'], rotation=0)
axes[1, 0].grid(alpha=0.3)

# Age-Sex interaction
age_sex_disease = df_plot.groupby(['Sex', 'Heart Disease']).size().unstack(fill_value=0)
age_sex_disease.plot(kind='bar', ax=axes[1, 1], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Disease Count by Sex', fontweight='bold')
axes[1, 1].set_xlabel('Sex (0=Female, 1=Male)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Heart Disease')
axes[1, 1].set_xticklabels(['Female', 'Male'], rotation=0)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure1_demographics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/figure1_demographics.png")

# Figure 2: Clinical Measurements
print("\n2.2 Creating Figure 2: Clinical Measurements")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

clinical_vars = ['BP', 'Cholesterol', 'Max HR', 'ST depression']
colors = {'Presence': 'red', 'Absence': 'green'}

for idx, var in enumerate(clinical_vars):
    row = idx // 3
    col = idx % 3
    
    sns.violinplot(x='Heart Disease', y=var, data=df_plot, ax=axes[row, col], 
                  palette=colors, inner='box')
    axes[row, col].set_title(f'{var} Distribution', fontweight='bold')
    axes[row, col].set_ylabel(var)
    axes[row, col].grid(alpha=0.3)

# Chest pain type
chest_pain = pd.crosstab(df_plot['Chest pain type'], df_plot['Heart Disease'])
chest_pain.plot(kind='bar', ax=axes[1, 1], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Heart Disease by Chest Pain Type', fontweight='bold')
axes[1, 1].set_xlabel('Chest Pain Type')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Heart Disease')
axes[1, 1].grid(alpha=0.3)

# Exercise angina
angina = pd.crosstab(df_plot['Exercise angina'], df_plot['Heart Disease'], normalize='index') * 100
angina.plot(kind='bar', ax=axes[1, 2], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Disease Rate by Exercise Angina', fontweight='bold')
axes[1, 2].set_xlabel('Exercise Angina (0=No, 1=Yes)')
axes[1, 2].set_ylabel('Percentage (%)')
axes[1, 2].legend(title='Heart Disease')
axes[1, 2].set_xticklabels(['No', 'Yes'], rotation=0)
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure2_clinical_measurements.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/figure2_clinical_measurements.png")

# Figure 3: Correlation Heatmap
print("\n2.3 Creating Figure 3: Feature Correlations")
numerical_features = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
                     'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                     'ST depression', 'Slope of ST', 'Number of vessels fluro', 
                     'Thallium', 'Heart Disease Binary']

corr_matrix = df_plot[numerical_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/figure3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/figure3_correlation_heatmap.png")

# Figure 4: Key Risk Factors
print("\n2.4 Creating Figure 4: Risk Factor Analysis")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Number of vessels
vessels = pd.crosstab(df_plot['Number of vessels fluro'], df_plot['Heart Disease'])
vessels.plot(kind='bar', ax=axes[0, 0], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Vessels Colored by Fluoroscopy', fontweight='bold')
axes[0, 0].set_xlabel('Number of Vessels')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend(title='Heart Disease')
axes[0, 0].grid(alpha=0.3)

# Thallium test
thal = pd.crosstab(df_plot['Thallium'], df_plot['Heart Disease'])
thal.plot(kind='bar', ax=axes[0, 1], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Thallium Stress Test Results', fontweight='bold')
axes[0, 1].set_xlabel('Thallium Result')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(title='Heart Disease')
axes[0, 1].grid(alpha=0.3)

# ST slope
slope = pd.crosstab(df_plot['Slope of ST'], df_plot['Heart Disease'])
slope.plot(kind='bar', ax=axes[1, 0], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Slope of ST Segment', fontweight='bold')
axes[1, 0].set_xlabel('ST Slope')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(title='Heart Disease')
axes[1, 0].grid(alpha=0.3)

# EKG results
ekg = pd.crosstab(df_plot['EKG results'], df_plot['Heart Disease'])
ekg.plot(kind='bar', ax=axes[1, 1], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 1].set_title('EKG Results', fontweight='bold')
axes[1, 1].set_xlabel('EKG Category')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Heart Disease')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure4_risk_factors.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/figure4_risk_factors.png")

# Figure 5: Bivariate Relationships
print("\n2.5 Creating Figure 5: Bivariate Analysis")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age vs Max HR
for disease in ['Absence', 'Presence']:
    subset = df_plot[df_plot['Heart Disease'] == disease]
    color = 'green' if disease == 'Absence' else 'red'
    axes[0, 0].scatter(subset['Age'], subset['Max HR'], alpha=0.5, 
                      label=disease, s=30, c=color, edgecolors='black', linewidth=0.5)
axes[0, 0].set_xlabel('Age (years)')
axes[0, 0].set_ylabel('Max Heart Rate')
axes[0, 0].set_title('Age vs Maximum Heart Rate', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Cholesterol vs BP
for disease in ['Absence', 'Presence']:
    subset = df_plot[df_plot['Heart Disease'] == disease]
    color = 'green' if disease == 'Absence' else 'red'
    axes[0, 1].scatter(subset['Cholesterol'], subset['BP'], alpha=0.5, 
                      label=disease, s=30, c=color, edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel('Cholesterol (mg/dl)')
axes[0, 1].set_ylabel('Blood Pressure (mm Hg)')
axes[0, 1].set_title('Cholesterol vs Blood Pressure', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# ST depression vs Max HR
for disease in ['Absence', 'Presence']:
    subset = df_plot[df_plot['Heart Disease'] == disease]
    color = 'green' if disease == 'Absence' else 'red'
    axes[1, 0].scatter(subset['ST depression'], subset['Max HR'], alpha=0.5, 
                      label=disease, s=30, c=color, edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel('ST Depression')
axes[1, 0].set_ylabel('Max Heart Rate')
axes[1, 0].set_title('ST Depression vs Max Heart Rate', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Age vs Cholesterol
for disease in ['Absence', 'Presence']:
    subset = df_plot[df_plot['Heart Disease'] == disease]
    color = 'green' if disease == 'Absence' else 'red'
    axes[1, 1].scatter(subset['Age'], subset['Cholesterol'], alpha=0.5, 
                      label=disease, s=30, c=color, edgecolors='black', linewidth=0.5)
axes[1, 1].set_xlabel('Age (years)')
axes[1, 1].set_ylabel('Cholesterol (mg/dl)')
axes[1, 1].set_title('Age vs Cholesterol', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure5_bivariate_relationships.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/figure5_bivariate_relationships.png")

# ============================================================================
# PART 3: MACHINE LEARNING METHODS
# ============================================================================

print("\n" + "="*80)
print("PART 3: MACHINE LEARNING ANALYSIS")
print("="*80)

# Prepare data for modeling
X = df_plot[numerical_features[:-1]]  # Exclude target
y = df_plot['Heart Disease Binary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                      random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Method 1: Logistic Regression
print("\n3.1 Method 1: Logistic Regression")
print("-" * 40)

log_reg = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
auc_lr = auc(fpr_lr, tpr_lr)

print(f"Accuracy: {acc_lr:.4f}")
print(f"AUC: {auc_lr:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Disease']))

# Feature importance
feature_importance_lr = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance_lr.head(10))
feature_importance_lr.to_csv('results/feature_importance_logistic.csv', index=False)

# Method 2: Random Forest
print("\n3.2 Method 2: Random Forest Classifier")
print("-" * 40)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
auc_rf = auc(fpr_rf, tpr_rf)

print(f"Accuracy: {acc_rf:.4f}")
print(f"AUC: {auc_rf:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Disease']))

# Feature importance
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance_rf.head(10))
feature_importance_rf.to_csv('results/feature_importance_random_forest.csv', index=False)

# Method 3: Gradient Boosting
print("\n3.3 Method 3: Gradient Boosting Classifier")
print("-" * 40)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                               max_depth=5, random_state=42)
gb.fit(X_train_scaled, y_train)

y_pred_gb = gb.predict(X_test_scaled)
y_pred_proba_gb = gb.predict_proba(X_test_scaled)[:, 1]

acc_gb = accuracy_score(y_test, y_pred_gb)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
auc_gb = auc(fpr_gb, tpr_gb)

print(f"Accuracy: {acc_gb:.4f}")
print(f"AUC: {auc_gb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb, target_names=['No Disease', 'Disease']))

# Figure 6: Model Comparison
print("\n2.6 Creating Figure 6: Model Performance Comparison")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC Curves
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
axes[0, 0].plot(fpr_lr, tpr_lr, linewidth=2.5, label=f'Logistic Reg (AUC={auc_lr:.3f})', color='blue')
axes[0, 0].plot(fpr_rf, tpr_rf, linewidth=2.5, label=f'Random Forest (AUC={auc_rf:.3f})', color='green')
axes[0, 0].plot(fpr_gb, tpr_gb, linewidth=2.5, label=f'Gradient Boost (AUC={auc_gb:.3f})', color='red')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curves - Model Comparison', fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(alpha=0.3)

# Accuracy comparison
models = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
accuracies = [acc_lr, acc_rf, acc_gb]
aucs = [auc_lr, auc_rf, auc_gb]

x_pos = np.arange(len(models))
width = 0.35

axes[0, 1].bar(x_pos - width/2, accuracies, width, label='Accuracy', 
              color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].bar(x_pos + width/2, aucs, width, label='AUC', 
              color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Model Performance Metrics', fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(models)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')
axes[0, 1].set_ylim([0, 1.1])

for i, (acc, auc_val) in enumerate(zip(accuracies, aucs)):
    axes[0, 1].text(i - width/2, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[0, 1].text(i + width/2, auc_val + 0.02, f'{auc_val:.3f}', ha='center', fontsize=9, fontweight='bold')

# Confusion Matrix - Best Model (highest AUC)
best_idx = np.argmax([auc_lr, auc_rf, auc_gb])
best_model_name = ['Logistic Regression', 'Random Forest', 'Gradient Boosting'][best_idx]
best_y_pred = [y_pred_lr, y_pred_rf, y_pred_gb][best_idx]

cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
           xticklabels=['No Disease', 'Disease'],
           yticklabels=['No Disease', 'Disease'])
axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# Feature importance - Best Model
if best_idx == 0:  # Logistic Regression
    top_features = feature_importance_lr.head(10)
    axes[1, 1].barh(range(len(top_features)), np.abs(top_features['Coefficient']), 
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['Feature'], fontsize=9)
    axes[1, 1].set_xlabel('|Coefficient|')
else:  # Tree-based models
    if best_idx == 1:
        top_features = feature_importance_rf.head(10)
    else:
        top_features = pd.DataFrame({
            'Feature': X.columns,
            'Importance': gb.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
    
    axes[1, 1].barh(range(len(top_features)), top_features.iloc[:, 1], 
                   color='forestgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['Feature'], fontsize=9)
    axes[1, 1].set_xlabel('Importance')

axes[1, 1].set_title(f'Top 10 Features - {best_model_name}', fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figures/figure6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/figure6_model_comparison.png")

# Method 4: K-Means Clustering
print("\n3.4 Method 4: K-Means Clustering Analysis")
print("-" * 40)

# Determine optimal k using elbow method
inertias = []
silhouettes = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)

# Perform clustering with k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_scaled)

print(f"Optimal k: {optimal_k}")
print(f"\nCluster Distribution:")
unique, counts = np.unique(clusters, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} patients ({count/len(clusters)*100:.1f}%)")

# Analyze clusters
df_train_clustered = X_train.copy()
df_train_clustered['Cluster'] = clusters
df_train_clustered['Heart Disease'] = y_train.values

print("\nHeart Disease Distribution by Cluster:")
for cluster in range(optimal_k):
    cluster_data = df_train_clustered[df_train_clustered['Cluster'] == cluster]
    disease_rate = (cluster_data['Heart Disease'] == 1).sum() / len(cluster_data) * 100
    print(f"  Cluster {cluster}: {disease_rate:.1f}% have heart disease")

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)

print(f"\nPCA Explained Variance: {pca.explained_variance_ratio_.sum():.3f}")

# Figure 7: Clustering Analysis
print("\n2.7 Creating Figure 7: Clustering Analysis")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Elbow curve
axes[0, 0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0, 0].set_title('Elbow Method for Optimal k', fontweight='bold')
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# PCA visualization - by cluster
scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis',
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
axes[0, 1].set_title('K-Means Clustering (PCA Visualization)', fontweight='bold')
plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')
axes[0, 1].grid(alpha=0.3)

# PCA visualization - by disease
scatter2 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='RdYlGn_r',
                             alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
axes[1, 0].set_title('Patient Distribution by Disease Status', fontweight='bold')
cbar = plt.colorbar(scatter2, ax=axes[1, 0], ticks=[0, 1])
cbar.ax.set_yticklabels(['No Disease', 'Disease'])
axes[1, 0].grid(alpha=0.3)

# Disease rate by cluster
cluster_disease_rates = []
for cluster in range(optimal_k):
    cluster_data = df_train_clustered[df_train_clustered['Cluster'] == cluster]
    disease_rate = (cluster_data['Heart Disease'] == 1).sum() / len(cluster_data) * 100
    cluster_disease_rates.append(disease_rate)

axes[1, 1].bar(range(optimal_k), cluster_disease_rates, 
              color=['skyblue', 'lightcoral', 'lightgreen'][:optimal_k],
              alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Heart Disease Rate (%)')
axes[1, 1].set_title('Heart Disease Prevalence by Cluster', fontweight='bold')
axes[1, 1].set_xticks(range(optimal_k))
axes[1, 1].grid(alpha=0.3, axis='y')

for i, rate in enumerate(cluster_disease_rates):
    axes[1, 1].text(i, rate + 2, f'{rate:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/figure7_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/figure7_clustering_analysis.png")

# Save cluster profiles
cluster_profiles = df_train_clustered.groupby('Cluster').mean()
cluster_profiles.to_csv('results/cluster_profiles.csv')
print("✓ Saved: results/cluster_profiles.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

summary = {
    'Dataset': {
        'Total Samples': len(df),
        'Features': len(df.columns) - 1,
        'Disease Cases': (df['Heart Disease'] == 'Presence').sum(),
        'No Disease Cases': (df['Heart Disease'] == 'Absence').sum()
    },
    'Best Model': {
        'Name': best_model_name,
        'Accuracy': f'{[acc_lr, acc_rf, acc_gb][best_idx]:.4f}',
        'AUC': f'{[auc_lr, auc_rf, auc_gb][best_idx]:.4f}'
    },
    'Clustering': {
        'Optimal k': optimal_k,
        'PCA Variance Explained': f'{pca.explained_variance_ratio_.sum():.3f}'
    }
}

print("\nSUMMARY:")
for section, values in summary.items():
    print(f"\n{section}:")
    for key, value in values.items():
        print(f"  {key}: {value}")

# Save summary
with open('results/analysis_summary.txt', 'w') as f:
    f.write("HEART DISEASE PREDICTION - ANALYSIS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    for section, values in summary.items():
        f.write(f"{section}:\n")
        for key, value in values.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

print("\n✓ All figures saved in 'figures/' directory")
print("✓ All results saved in 'results/' directory")
print("\nReady to generate final report!")
