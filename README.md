# 🔐 Phishing URL Detection 

This repository contains a complete **phishing URL detection project**, including dataset preparation, cleaning, feature engineering, and exploratory data analysis (EDA). The goal is to prepare high-quality data for building machine learning models that classify URLs as **phishing** or **legitimate**. :contentReference[oaicite:0]{index=0}

---

## 📌 Overview

Phishing websites attempt to trick users into revealing sensitive information by mimicking legitimate sites. Detecting these URLs early is critical to preventing cyber-attacks, identity theft, and financial loss.

This project performs:

- ✔ **Data loading & preprocessing**  
- ✔ **Handling missing values & duplicates**  
- ✔ **Feature scaling & outlier removal**  
- ✔ **Exploratory data analysis (EDA)**  
- ✔ **Custom URL-based feature engineering**  

This forms the foundation for training high-performance ML classifiers (e.g., Random Forest, XGBoost, Logistic Regression, etc.) in future work.

---

## 📊 Dataset

- **Samples:** 235,794 URLs  
- **Classes:**  
  - **Phishing:** 100,945  
  - **Legitimate:** 134,850  
- **Features:** 56 original + 4 engineered features  
- **Format:** Cleaned CSV file

### 🔧 Engineered Features (important!)
Based on URL behavior, we engineered the following additional attributes: :contentReference[oaicite:1]{index=1}  

- **CharContinuationRate** – Measures irregular character transitions  
- **URLTitleMatchScore** – How closely the HTML `<title>` matches the URL  
- **URLCharProb** – Statistical probability of character sequences  
- **TLDLegitimateProb** – Likelihood the top-level domain is legitimate  

These features significantly improve URL-based threat detection.

---

## 🧹 Data Preprocessing

Key preprocessing steps implemented:

### ✔ Missing values
```python
df.fillna(df.mean(numeric_only=True), inplace=True)
```

### ✔ Duplicate removal
```python
df = df.drop_duplicates()
```

### ✔ Data type normalization
- Convert binary columns → `bool`
- Normalize numeric types for modeling

### ✔ Outlier removal (IQR)
Used for skewed features to reduce noise.

### ✔ Min-Max scaling
```python
X_scaled = (X - X.min()) / (X.max() - X.min())
```

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes:

- 📉 **Histograms** for numeric features  
- 🔥 **Correlation heatmaps**  
- 🧊 **Boxplots** for distribution and outliers  
- 🧮 **Class distribution plots (bar/pie)**  
- 🔗 High correlation between `URLLength` and `DomainLength` observed  
- 🧾 Summary statistics for all columns  

These insights directly influence model selection and preprocessing strategy. :contentReference[oaicite:2]{index=2}

---

## 📁 Repository Structure

```text
.
├── data/
│   └── phishing.csv                  # dataset
├── notebook/
│   └── phishing_eda.ipynb            # main analysis notebook
├── reports/
│   └── phishing_report.pdf           # project report
├── src/
│   ├── data_loader.py                # loading, basic validation
│   ├── preprocessing.py              # cleaning, scaling, outliers
│   └── feature_engineering.py        # URL-level engineered features
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/phishing-url-detection.git
cd phishing-url-detection
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Open the Jupyter notebook
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

## 📚 References

- UCI ML Repository – Phishing Websites / URL datasets  
- *Phishing URL Detection Report* (included in `/reports/`) :contentReference[oaicite:3]{index=3}

