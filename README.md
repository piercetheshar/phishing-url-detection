Phishing URL Detection Using Machine Learning

A machine learning–based system to classify URLs as legitimate or phishing, built using a large dataset and custom engineered features. This project demonstrates end-to-end ML workflow: data cleaning, preprocessing, feature engineering, EDA visualizations, and phishing detection insights.

🔍 Project Overview

Phishing remains one of the most prevalent cybersecurity threats. Attackers deceive users using fake websites designed to steal personal information.
This project builds a data-driven phishing URL classifier using 235,794 samples and 56 features extracted from URL structure, domain metadata, and character patterns.

The goal is simple:
👉 Automate phishing detection using machine learning–ready features.

📊 Key Features
✔ Extensive Data Cleaning

Missing value imputation

Duplicate removal

Data type standardization

Outlier removal using the IQR method

Feature scaling using Min–Max normalization

✔ Custom Feature Engineering

Created four additional features to improve phishing detection accuracy:

CharContinuationRate – captures unnatural character transitions

URLTitleMatchScore – measures similarity between URL text & page title

URLCharProb – probability score for URL character pattern legitimacy

TLDLegitimateProb – evaluates legitimacy probability of URL TLD

✔ Exploratory Data Analysis (EDA)

Histograms

Correlation heatmap

Boxplots for outlier detection

Class distribution charts

✔ Dataset

Source: UCI Machine Learning Repository – Phishing Websites Dataset
Samples: 235,794 URLs
Classes: Legitimate (0) vs Phishing (1)

🛠️ Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-Learn

Jupyter Notebook

