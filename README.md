# 🧠 Stroke Prediction Using Data Mining

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-brightgreen?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📖 Project Overview

**Stroke Prediction Using Data Mining** is a machine learning project that aims to build a predictive model to classify whether an individual is likely to suffer a stroke based on their healthcare and demographic data.

This project involved:
- Comprehensive data preprocessing and EDA
- Implementation of **5 ML models** (Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, SVM)
- Usage of techniques like **SMOTE** (for class imbalance) and **Recursive Feature Elimination (RFE)**
- Creation of an advanced **Stacked Ensemble Model** for improved predictive accuracy

---

## 📊 Dataset

- Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Records: **5,110** instances
- Features:
  - **Categorical**: Gender, Ever Married, Work Type, Residence Type, Smoking Status
  - **Numerical**: Age, Hypertension, Heart Disease, Avg Glucose Level, BMI

*Note: BMI missing values were handled via mean imputation.*

---

## 🛠️ Techniques Used

| Technique | Purpose |
|:---------:|:-------:|
| **Label Encoding** | Convert categorical variables |
| **SMOTE** | Handle class imbalance |
| **Standard Scaler** | Normalize numerical features |
| **Recursive Feature Elimination (RFE)** | Feature selection |
| **Ensemble Learning** | Boost accuracy with multiple models |

---

## 📈 Models Evaluated

| Model | Accuracy | F1 Score | ROC AUC |
|:------|:--------:|:--------:|:-------:|
| Logistic Regression | 0.79 | 0.80 | 0.85 |
| Decision Tree | 0.91 | 0.91 | 0.91 |
| Random Forest | **0.96** | **0.96** | **0.99** |
| K-Nearest Neighbors (KNN) | 0.90 | 0.90 | 0.95 |
| Support Vector Machines (SVM) | 0.84 | 0.85 | 0.91 |

🚀 **Best Model**: Random Forest  
🚀 **Even Better**: Stacked Ensemble achieved **96.66% accuracy**!

---

## 📊 Additional Observations

- **Tuned SVM** improved accuracy and ROC AUC through hyperparameter optimization.
- **Stacked Ensemble** combined base models and gave the best generalization.

---


---

## 🔥 Key Takeaways

- **Data Preprocessing** and **Feature Engineering** significantly affect performance.
- **SMOTE** improved stroke class detection.
- **Ensemble models** (Random Forest, Stacked) outperform individual models.
- **Feature Selection (RFE)** simplifies models with minor performance trade-offs.

---

## 📚 References

- 📂 Dataset: [Stroke Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 📑 Project Notebook: [Google Colab Notebook](https://colab.research.google.com/drive/1jgS5OhntiZSgC0nOAxXHaTfI-8YK6H8t?usp=sharing)

---

## 📬 Contact

- 👤 Hetu Virajkumar Patel | [GitHub](https://github.com/hetuvpatel) | [LinkedIn](https://www.linkedin.com/in/hetu-patel-toronto/)
- 👤 Nilay Thakorbhai Patel

---

> 🎯 **Developed as part of CPS 844 - Data Mining, under Prof. Cherie Ding at Toronto Metropolitan University.**

---

