# Heart Disease Prediction Analysis Project

A comprehensive machine learning project analyzing heart disease prediction using clinical and diagnostic data, achieving 92% AUC with logistic regression.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project performs an in-depth analysis of a heart disease dataset containing 270 patients with 13 clinical features. Through exploratory data analysis, statistical modeling, and machine learning techniques, we achieved excellent predictive performance and identified key risk factors for cardiovascular disease.

**Key Achievements:**
- 🎯 **92% AUC** with Logistic Regression (best model)
- 📊 **7 comprehensive visualizations** exploring data patterns
- 🤖 **4 machine learning methods** (Logistic Regression, Random Forest, Gradient Boosting, K-Means Clustering)
- 📈 **3 distinct patient subgroups** identified with disease prevalence ranging from 20% to 86%

## 🗂️ Project Structure

```
Heart-Disease-Prediction/
│
├── README.md                              # Project documentation (this file)
├── Heart_Disease_Analysis_Report.md       # Complete analysis report (15 pages)
├── Heart_Disease_Prediction 2.csv         # Dataset (270 patients, 14 variables)
├── heart_disease_complete_analysis.py     # Main analysis script
│
├── figures/                               # Generated visualizations
│   ├── figure1_demographics.png
│   ├── figure2_clinical_measurements.png
│   ├── figure3_correlation_heatmap.png
│   ├── figure4_risk_factors.png
│   ├── figure5_bivariate_relationships.png
│   ├── figure6_model_comparison.png
│   └── figure7_clustering_analysis.png
│
└── results/                               # Analysis outputs
    ├── statistical_summary.csv
    ├── feature_importance_logistic.csv
    ├── feature_importance_random_forest.csv
    ├── cluster_profiles.csv
    └── analysis_summary.txt
```

## 📊 Dataset Description

**Source:** Heart Disease Prediction Dataset  
**Size:** 270 patients × 14 variables  
**Target:** Heart Disease (Presence/Absence)  
**Class Distribution:** 44.4% disease, 55.6% no disease (relatively balanced)

### Features

| Category | Features |
|----------|----------|
| **Demographics** | Age, Sex |
| **Clinical Measurements** | Blood Pressure, Cholesterol, Max Heart Rate, ST Depression |
| **Diagnostic Tests** | Chest Pain Type, FBS over 120, EKG Results, Exercise Angina, Slope of ST, Number of Vessels (Fluoroscopy), Thallium Stress Test |
| **Target** | Heart Disease Status (Presence/Absence) |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Or using requirements.txt (if created):
```bash
pip install -r requirements.txt
```

### Running the Analysis

```bash
python heart_disease_complete_analysis.py
```

This will:
- Load and explore the dataset
- Generate 7 visualization figures (saved to `figures/`)
- Train 4 machine learning models
- Perform K-means clustering analysis
- Save results to `results/` directory

**Expected Runtime:** 10-30 seconds

## 📈 Key Results

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| **Logistic Regression** | **88.24%** | **0.9202** | 82% | 93% | 0.88 |
| Random Forest | 86.76% | 0.8965 | 84% | 87% | 0.85 |
| Gradient Boosting | 80.88% | 0.8930 | 74% | 87% | 0.80 |

### Top Predictive Features

1. **Number of Vessels (Fluoroscopy)** - Coefficient: 1.11
2. **Sex** - Coefficient: 0.91
3. **Chest Pain Type** - Coefficient: 0.70
4. **Thallium Stress Test** - Coefficient: 0.52
5. **ST Slope** - Coefficient: 0.46

### Patient Clustering Results

K-means clustering identified three distinct patient subgroups:

- **Cluster 0 (Low Risk):** 28% of patients, 20% disease prevalence
- **Cluster 1 (Moderate Risk):** 37% of patients, 24% disease prevalence
- **Cluster 2 (High Risk):** 35% of patients, 86% disease prevalence

## 📖 Detailed Analysis Report

For comprehensive analysis including:
- In-depth data exploration
- Feature correlation analysis
- Detailed model evaluation
- Clinical interpretation
- Future directions

👉 **See [Heart_Disease_Analysis_Report.md](Heart_Disease_Analysis_Report.md)** (15 pages)

## 🔍 Visualizations

### Sample Outputs

All visualizations are generated automatically and saved to the `figures/` directory:

1. **Demographics Analysis** - Age and sex distribution by disease status
2. **Clinical Measurements** - Blood pressure, cholesterol, heart rate distributions
3. **Correlation Heatmap** - Feature relationships and multicollinearity assessment
4. **Risk Factors** - Diagnostic test results (fluoroscopy, thallium, ST slope, EKG)
5. **Bivariate Relationships** - Scatter plots of key variable pairs
6. **Model Comparison** - ROC curves, accuracy metrics, confusion matrices
7. **Clustering Analysis** - PCA visualization and cluster characterization

## 🛠️ Technologies Used

- **Python 3.x** - Programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **SciPy** - Scientific computing

## 📊 Methodology

### 1. Data Exploration
- Statistical summary and missing value analysis
- Feature distribution visualization
- Correlation analysis

### 2. Feature Engineering
- Standardization using StandardScaler
- Train-test split (75%-25%)
- No missing values (100% complete dataset)

### 3. Model Training
- **Logistic Regression** - Linear probabilistic classifier
- **Random Forest** - Ensemble of decision trees
- **Gradient Boosting** - Sequential boosting algorithm
- **K-Means Clustering** - Unsupervised patient grouping

### 4. Model Evaluation
- ROC-AUC curves
- Confusion matrices
- Precision, recall, F1-scores
- Feature importance analysis

## 💡 Key Findings

1. **Excellent Predictive Performance**: Logistic regression achieved 0.92 AUC, significantly outperforming typical clinical risk scores (AUC ≈ 0.70-0.75)

2. **Invasive Testing Dominates**: Coronary angiography (vessel fluoroscopy) emerged as the strongest single predictor, validating its gold-standard status

3. **Functional Testing Valuable**: Exercise stress test parameters (max HR, ST depression) provided substantial predictive power

4. **Patient Heterogeneity**: Clustering revealed a high-risk subgroup (35% of patients) with 86% disease prevalence requiring intensive intervention

5. **Traditional Risk Factors Limited**: Blood pressure and cholesterol showed weaker discrimination than expected, possibly due to medication confounding

## 🔮 Future Directions

- **External Validation**: Test on independent datasets
- **Feature Engineering**: Add interaction terms and polynomial features
- **Deep Learning**: Apply neural networks to raw ECG signals
- **Longitudinal Analysis**: Incorporate time-to-event modeling
- **Clinical Deployment**: Develop decision support system

## 📚 References

1. World Health Organization. (2021). Cardiovascular diseases (CVDs). Global Health Observatory.
2. D'Agostino RB Sr, et al. (2008). General cardiovascular risk profile: the Framingham Heart Study. *Circulation*, 117(6):743-753.
3. Breiman L. (2001). Random Forests. *Machine Learning*, 45(1):5-32.
4. Hastie T, Tibshirani R, Friedman J. (2009). *The Elements of Statistical Learning*. Springer.

## 👥 Author

**Data Science 611 Course Project**  
Date: December 2025

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YOUR_USERNAME/Heart-Disease-Prediction/issues).

## 📞 Contact

For questions or collaboration opportunities:
- Open an issue on GitHub
- Email: [Your Email]

## 🙏 Acknowledgments

- Dataset contributors
- Data Science 611 course instructors
- Open-source community (scikit-learn, pandas, matplotlib teams)

---

**⭐ If you find this project useful, please consider giving it a star!**

## 📝 Citation

If you use this code or analysis in your research, please cite:

```bibtex
@misc{heart_disease_analysis_2025,
  author = {Data Science 611},
  title = {Heart Disease Prediction: A Comprehensive Machine Learning Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/Heart-Disease-Prediction}
}
```
