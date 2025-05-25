
# Phishing Email Detection System

This project is focused on using machine learning to detect phishing emails. It classifies emails as either **phishing** or **legitimate** using natural language processing (NLP) and a trained logistic regression model.

---

## Project Overview

Phishing is one of the most common and dangerous types of cyberattacks. This system uses a supervised ML model to automatically analyze the content of emails and classify them in real time, providing a first line of defense against phishing threats.

---

## Objectives

- Build a machine learning model to detect phishing emails
- Extract text features using NLP techniques
- Evaluate model performance on labeled data
- Provide an interactive interface using Gradio

---

## Technologies Used

- **Python**
- **Libraries**:
  - `scikit-learn` – for machine learning
  - `pandas`, `numpy` – for data processing
  - `joblib` – for saving/loading the trained model
  - `gradio` – for building a simple web interface

---

## Project Files

| File Name                           | Description                                 |
|------------------------------------|---------------------------------------------|
| `phishing_detection.ipynb`         | Main Jupyter Notebook with code and steps   |
| `emails_dataset.csv`               | Labeled dataset used for training/testing   |
| `phishing_model.pkl`               | Saved trained model                         |
| `vectorizer.pkl`                   | Saved TF-IDF vectorizer                     |
| `Phishing_Email_Detection_System.pdf` | Project synopsis/report (PDF)            |
| `README.md`                        | Project documentation (this file)           |

---

## Model Summary

- **Vectorization**: TF-IDF (Top 5000 words)
- **Classifier**: Logistic Regression
- **Evaluation**:
  - **Accuracy**: 100% (on small test set)
  - **Precision/Recall**: Perfect on sample data

---

## How to Use

### 📁 Upload Required Files in Colab:
- `phishing_model.pkl`
- `vectorizer.pkl`

### ▶Run the Following Code in Colab:

```python
!pip install gradio joblib scikit-learn pandas numpy --quiet

import gradio as gr
import joblib
import re
import string

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "url", text)
    text = re.sub(r"\S+@\S+", "email", text)
    text = re.sub(r"\d+", "number", text)
    text = re.sub(rf"[{string.punctuation}]", " ", text)
    return text

def predict_email(email_text):
    cleaned = preprocess_text(email_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return "Phishing Email" if prediction == 1 else "Legitimate Email"

gr.Interface(
    fn=predict_email,
    inputs=gr.Textbox(lines=5, placeholder="Paste email content here..."),
    outputs=gr.Text(label="Prediction"),
    title="Phishing Email Detection System",
    description="Classify emails as Phishing or Legitimate using a trained Machine Learning model."
).launch()
