# 🍷 Wine Classification

A machine learning project that classifies wine samples into 3 cultivar classes using the classic **Wine dataset** (`sklearn.datasets.load_wine`). Five models were trained, tuned, and compared to find the best performing classifier.

## 📊 Dataset

- **Source:** `sklearn.datasets.load_wine`
- **Samples:** 178
- **Features:** 13 chemical properties (alcohol, flavanoids, color intensity, proline, etc.)
- **Classes:** 3 wine cultivars

## 🧠 Models Trained & Tuned

Each model was tuned using `GridSearchCV` (5-fold cross-validation):

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Decision Tree | 0.9374 | 1.0000 |
| AdaBoost | 0.9719 | 0.9444 |
| **Gradient Boosting** | **0.9862** | **1.0000** |
| XGBoost | 0.9862 | 0.9722 |
| LightGBM | 0.9724 | 1.0000 |

**Final model:** Gradient Boosting — highest cross-validated accuracy, tied with XGBoost, plus perfect classification on the held-out test set.

## 🛠️ Tech Stack

- Python, pandas, numpy
- scikit-learn (Decision Tree, AdaBoost, Gradient Boosting, GridSearchCV)
- XGBoost, LightGBM
- SHAP (model interpretability)
- matplotlib, seaborn

## 🔍 Project Workflow

1. Exploratory Data Analysis (class distribution, feature distributions, correlation heatmap)
2. Train/test split with stratification
3. Baseline models for each algorithm
4. Hyperparameter tuning via loops + `GridSearchCV`
5. Model evaluation: accuracy, confusion matrices, cross-validated scores
6. SHAP analysis for feature importance and interpretability
7. Final model selection based on CV score (not just test accuracy)

## 📸 Screenshots

**Class Distribution**
<img width="568" height="454" alt="image" src="https://github.com/user-attachments/assets/1ad5c6f1-86fe-4e75-910d-23865b42c244" />


**Feature Correlation Heatmap**
<img width="1113" height="886" alt="image" src="https://github.com/user-attachments/assets/ccc59ed2-6a2a-4006-84c0-e4f5aa7a8852" />


**Confusion Matrices (All Models)**
<img width="1456" height="823" alt="image" src="https://github.com/user-attachments/assets/b0734d43-7e53-4cfe-9f7f-749e97c7f401" />


**SHAP Feature Importance**
<img width="913" height="498" alt="image" src="https://github.com/user-attachments/assets/4a2d434f-4e1e-4f73-acc9-8ce646cf183f" />
<img width="913" height="498" alt="image" src="https://github.com/user-attachments/assets/43352277-a492-4013-ab13-bc464ccd7baf" />
<img width="913" height="498" alt="image" src="https://github.com/user-attachments/assets/132a625c-0e40-4d02-a73d-115d1d581359" />


> Replace the placeholders above with actual image paths once you add screenshots to a `screenshots/` folder in the repo.

## 🚀 How to Run

```bash
git clone https://github.com/gunjan01-source/wine-classification-ml.git
cd wine-classification-ml
pip install -r requirements.txt
jupyter notebook model.ipynb
```

## 📁 Requirements

```
scikit-learn
xgboost
lightgbm
shap
pandas
numpy
matplotlib
seaborn
```

## 📌 Key Insight

The Decision Tree scored a perfect 100% on the test set, but had the *lowest* cross-validated score (93.74%) among all five models — a reminder that a small test set (36 samples) can be misleading, and cross-validation is a more reliable basis for model selection.
