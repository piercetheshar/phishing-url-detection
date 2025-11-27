# ⭐ Phishing URL Detection 

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Analysis-success.svg)]()
[![Pandas](https://img.shields.io/badge/EDA-Pandas-orange.svg)]()
[![scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-yellow.svg)]()

A data-intensive project that detects **phishing vs. legitimate websites** using only **URL-based features**, combined with custom feature engineering and machine learning models.  
This repository includes a full pipeline for:

- **Data cleaning & preprocessing**  
- **Feature engineering**  
- **Exploratory Data Analysis (EDA)**  
- **ML model training & evaluation**


---

# 📊 Dataset Overview

- **Source:** UCI Machine Learning Repository – Phishing Websites Dataset  
- **Size:** ≈ 235,000 URLs  
- **Classes:**
  - `0` → Legitimate  
  - `1` → Phishing  
- **Key columns from your dataset:**

| Column | Description |
|--------|-------------|
| URL | The website URL |
| URLLength | Length of URL |
| Domain | Extracted domain |
| IsDomainIP | Whether domain is IP |
| TLD | Top-Level Domain |
| NoOfSubDomain | Count of subdomains |
| HasObfuscation | Obfuscation indicator |
| NoOfLettersInURL | Letters count |
| DegitRatioInURL | Digit-to-total ratio |
| label | Target variable |

The file used in this repo: **phishing.csv**

---

# 🧼 Phase 1 – Data Cleaning & Preprocessing

Phase 1 focuses on preparing the raw dataset for modeling.

### ✔ Missing Values
All numeric columns are imputed using mean:

```python
df.fillna(df.mean(numeric_only=True), inplace=True)
```

### ✔ Duplicate Removal

```python
df_cleaned = df.drop_duplicates()
```

### ✔ Data Type Fixes
- Binary columns converted to `bool`
- String/categorical values normalized

### ✔ Outlier Removal
IQR-based filtering applied to numeric columns.

### ✔ Feature Scaling
Min–Max scaling to range `[0, 1]`:

\[
X_\text{scaled} = \frac{X - X_\min}{X_\max - X_\max}
\]

---

# 🧠 Phase 1 – Feature Engineering

Custom URL-based features added to enrich the dataset:

### 1️⃣ CharContinuationRate  
Measures character transition smoothness to catch abnormal patterns.

### 2️⃣ URLTitleMatchScore  
Checks similarity between URL and HTML title  
(legitimate pages tend to match more closely).

### 3️⃣ URLCharProb  
Statistical probability of URL character sequences.

### 4️⃣ TLDLegitimateProb  
Scores TLDs based on known phishing vs. legitimate likelihood.

These features are created in:

```
src/feature_engineering.py
```

---

# 📈 Phase 1 – Exploratory Data Analysis (EDA)

The notebook explores:

- Histograms of key features  
- Correlation heatmap  
- Outlier patterns  
- Class distribution:  
  **64% legitimate | 36% phishing**
- Boxplots of numeric features  
- Domain/TLD distribution plots  

---

# 🤖 Phase 2 – Machine Learning Modeling

Phase 2 focuses on building full ML pipeline and evaluating models.

The script:

```
python -m src.train_phase2
```

runs:

1. Load dataset  
2. Apply all feature engineering  
3. Preprocess the data  
4. Split into train/test  
5. Train several baseline models  
6. Print evaluation metrics  

---

# 🧩 ML Models Trained

The following models are trained using scikit-learn:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- SVM (RBF kernel)

Models are defined in:

```
src/modeling.py
```

---

# 📊 Phase 2 – Evaluation Metrics

For each model, the script prints:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  
- Full Classification Report  

Example output:

```
==========================
Training model: RandomForest
==========================
Accuracy : 0.9643
Precision: 0.9532
Recall   : 0.9451
F1-score : 0.9491
ROC AUC  : 0.9827
```

---

# 🥇 Model Comparison Summary

At the end of training:

```
🎯 Summary of models by F1-score:
RandomForest: F1=0.9491, Acc=0.9643, Precision=0.9532, Recall=0.9451, ROC-AUC=0.9827
SVM_rbf:     F1=0.9324, Acc=0.9485, Precision=0.9391, Recall=0.9260, ROC-AUC=0.9731
DecisionTree:F1=0.8912, Acc=0.9023, Precision=0.8854, Recall=0.8972, ROC-AUC=0.9040
LogisticRegression: F1=0.8630, Acc=0.8801, Precision=0.8524, Recall=0.8740, ROC-AUC=0.9153
```

Random Forest typically performs best.

---

# 🚀 Future Extensions 

- Hyperparameter tuning (GridSearch, Bayesian Optimization)
- Feature importance + SHAP explainability
- FastAPI/Flask deployment
- Streamlit web app for real-time URL classification
- URL live scanning + real-time monitoring pipeline

---

# 📦 How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run full ML modeling pipeline
```
python -m src.train_phase2
```

### 3. Open EDA notebook
```
notebook/phase1_analysis.ipynb
```

---

# 🙌 Acknowledgements

Dataset:  
UCI Machine Learning Repository – Phishing Websites Dataset  
Models & pipeline built using **Python, Pandas, NumPy, Scikit-learn**.

---
