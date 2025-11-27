# 🔐 Phishing URL Detection 

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A data-intensive project for detecting **phishing vs. legitimate websites** using URL-based features.  
This repository includes **data cleaning, preprocessing, exploratory data analysis (EDA), and feature engineering** to prepare a high-quality dataset for machine learning classification.

---

## 🚀 Project Objectives

- Clean and preprocess a large-scale phishing URL dataset.
- Explore statistical patterns, correlations, and class imbalance.
- Engineer new semantic and character-based URL features.
- Prepare modeling-ready data for downstream ML classification.

---

## 📊 Dataset Description

- **Source:** UCI Phishing Websites / URL Dataset  
- **Samples:** 235,794 URLs  
- **Classes:**  
  - `0` → Legitimate  
  - `1` → Phishing  
- **Data File:** `phishing.csv`

The dataset includes structural, lexical, and heuristic URL features such as:
- URL length, domain length  
- Suspicious characters (`@`, `//`, `-`, `%`)  
- TLD information  
- Digit/letter ratios  
- Entropy measures  

---

## 🧹 Data Cleaning & Preprocessing

### ✔ Missing Values  
Checked using `df.isnull().sum()` and imputed numeric features using mean:

```python
df.fillna(df.mean(numeric_only=True), inplace=True)
```

### ✔ Duplicate Removal  
```python
df = df.drop_duplicates()
```

### ✔ Data Type Normalization  
- Columns with two unique values → converted to `bool`  
- Numeric types standardized for consistency  

### ✔ Outlier Removal (IQR Method)  
Used for skewed numeric features to reduce noise.

### ✔ Feature Scaling  
Min–Max scaling applied to bring all features to the `[0, 1]` range:

```python
X_scaled = (X - X.min()) / (X.max() - X.min())
```

---

## 🧠 Feature Engineering

New custom URL-derived features were added:

### **`CharContinuationRate`**
Measures how naturally characters transition through the URL.  
Phishing URLs often show abnormal jumps or symbol-heavy transitions.

### **`URLTitleMatchScore`**
Computes similarity between the page `<title>` and the URL.  
Legitimate websites usually maintain strong alignment.

### **`URLCharProb`**
Character sequence probability based on expected lexical patterns.  
Useful for detecting random or obfuscated URLs.

### **`TLDLegitimateProb`**
A statistical lookup of TLD reputation.  
Certain TLDs are disproportionately used by phishing domains.

These are appended as new columns in the dataset for later ML modeling.

---

## 📈 Exploratory Data Analysis (EDA)

The notebook performs:

- **Descriptive statistics** (`df.describe()`)
- **Histograms** of key numeric features  
- **Correlation heatmap** (to identify multicollinearity)
- **Class balance visualization**
  - Approximately **64% legitimate**  
  - Approximately **36% phishing**
- **Boxplots** for distribution and outlier inspection  
- **Scatterplots & density plots** for pattern discovery  

These insights guide preprocessing decisions and model selection.

---

## 📁 Repository Structure

```text
.
├── data/
│   └── phishing.csv                    # dataset (or a sample)
├── notebook/
│   └── phishing_eda.ipynb              # full EDA + preprocessing notebook
├── reports/
│   └── phishing_report.pdf             # project report
├── src/
│   ├── data_loader.py                  # dataset loading utilities
│   ├── preprocessing.py                # cleaning, handling missing data, scaling
│   └── feature_engineering.py          # custom URL-based feature engineering
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the notebook
```bash
jupyter notebook notebook/phishing_eda.ipynb
```

---

## 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
notebook
```

---

## 🔮 Future Enhancements

- Build ML classifiers (Random Forest, XGBoost, Logistic Regression)
- Add hyperparameter tuning and cross-validation  
- Feature selection & PCA  
- Deploy model with FastAPI or Streamlit  
- Real-time URL scanning using a live API  

---

## 📚 References

- UCI ML Repository – Phishing Websites Data  
- *Phishing URL Detection Report* (included in `/reports/`)

