# Phishing_Email_Detection_System
This project uses machine learning to classify emails as phishing or legitimate based on their content. It includes model training, evaluation, and a web interface.
# 🛡️ Phishing Email Detection System

This project is a machine learning-based system to detect phishing emails. It classifies emails as either **phishing** or **legitimate** using natural language processing (NLP) and a trained logistic regression model.

---

## 📌 Project Overview

Phishing is a major cybersecurity threat where attackers trick users into revealing sensitive information. This project aims to mitigate that threat using a supervised learning model trained on email content.

---

## 🎯 Objectives

- Develop an ML model to classify emails as phishing or legitimate
- Extract key features from email text
- Train and evaluate the model on a labeled dataset
- Deploy a user interface for real-time predictions

---

## 🛠️ Technologies Used

- **Language**: Python  
- **Libraries**:
  - `scikit-learn` – ML model and evaluation
  - `pandas`, `numpy` – data handling
  - `joblib` – model serialization
  - `gradio` – web-based user interface

---

## 📂 Files

| File Name                           | Description                                 |
|------------------------------------|---------------------------------------------|
| `phishing_detection.ipynb`         | Jupyter Notebook with the full pipeline     |
| `emails_dataset.csv`               | Labeled dataset of phishing and legit emails|
| `phishing_model.pkl`               | Trained ML model                            |
| `vectorizer.pkl`                   | TF-IDF vectorizer for feature extraction    |
| `Phishing_Email_Detection_System.pdf` | Project report and synopsis                |
| `README.md`                        | Project documentation                       |

---

## 🔍 Model Details

- **Preprocessing**: Cleaning text, replacing URLs/emails/numbers
- **Feature Extraction**: TF-IDF Vectorizer (top 5000 features)
- **Model**: Logistic Regression
- **Accuracy**: 100% on test data (sample size: 10)

---

## 🚀 How to Use

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
