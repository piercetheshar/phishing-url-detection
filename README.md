# Phishing URL Detection – Data Intensive Computing Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Phase%201-green.svg)]()
[![Pandas](https://img.shields.io/badge/Pandas-EDA-orange.svg)]()
[![scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-yellow.svg)]()

A data-intensive project to detect **phishing vs. legitimate websites** using only their URL-based features.  
The project uses the **UCI Phishing Websites dataset** (~235k URLs, 56+ features) and focuses on **data cleaning, preprocessing, exploratory data analysis (EDA), and feature engineering** to build stronger machine learning models. :contentReference[oaicite:0]{index=0}

---

## 🚀 Project Goals

- Clean and preprocess a large phishing URL dataset.
- Explore feature distributions, correlations, and class imbalance.
- Engineer **new URL-based features** to better capture phishing patterns.
- Prepare a modeling-ready dataset for downstream ML classifiers.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository – Phishing Websites dataset  
- **Samples:** ~235,794 URLs  
- **Classes:**  
  - `0` – Legitimate  
  - `1` – Phishing  
- **File used in this repo:** `phishing.csv`

The dataset includes both original and derived URL features, such as URL length, domain characteristics, presence of special characters, and more. :contentReference[oaicite:1]{index=1}

---

## 🛠️ Data Cleaning & Preprocessing

Key steps implemented:

1. **Missing values**
   - Checked with `df.isnull().sum()`
   - Imputed using column-wise mean for numeric features:
     ```python
     df.fillna(df.mean(numeric_only=True), inplace=True)
     ```

2. **Duplicate records**
   - Detected and removed to avoid bias:
     ```python
     df_cleaned = df.drop_duplicates()
     ```

3. **Data types & binary features**
   - Columns with only 2 unique values converted to `bool` for consistency.

4. **Outlier removal**
   - Used **IQR (Interquartile Range)** per numeric column to detect and drop extreme outliers.

5. **Feature scaling**
   - Applied Min–Max scaling to bring all features to `[0, 1]`:
     \[
     X_\text{scaled} = \frac{X - X_\min}{X_\max - X_\min}
     \]

---

## 🧠 Feature Engineering

New features were engineered to better capture phishing patterns in URLs: :contentReference[oaicite:2]{index=2}

1. **`CharContinuationRate`**  
   Measures how smoothly characters transition in the URL, flagging abnormal character sequences.

2. **`URLTitleMatchScore`**  
   Quantifies similarity between the URL and the page title (legitimate sites usually align).

3. **`URLCharProb`**  
   Estimates how likely character sequences in the URL are under a normal distribution of characters.

4. **`TLDLegitimateProb`**  
   Scores the top-level domain (TLD) based on its historical use in phishing vs. legitimate URLs.

These engineered features are added as extra columns to the dataset for downstream modeling.

---

## 📈 Exploratory Data Analysis (EDA)

The notebook (and scripts) perform:

- **Descriptive statistics** via `df.describe()`
- **Histograms** of key features to inspect skewness and spread
- **Correlation heatmap** to spot redundant features and multicollinearity
- **Class distribution plots**:
  - Bar chart of counts for class 0 vs. class 1
  - Pie chart showing ~64% legitimate vs. ~36% phishing URLs :contentReference[oaicite:3]{index=3}
- **Boxplots** of numeric features to visualize outliers and variability

---

## 🗂️ Repository Structure

Suggested structure for this project:

```text
phishing-url-detection/
├─ data/
│  └─ phishing.csv
├─ notebooks/
│  └─ phase_1_exploration.ipynb
├─ src/
│  ├─ data_preprocessing.py
│  ├─ feature_engineering.py
│  ├─ eda.py
│  └─ utils.py
├─ reports/
│  └─ phase_1_report.pdf
├─ docs/
│  └─ workflow_diagram.png   # (optional, add later)
├─ README.md
├─ requirements.txt
└─ .gitignore
