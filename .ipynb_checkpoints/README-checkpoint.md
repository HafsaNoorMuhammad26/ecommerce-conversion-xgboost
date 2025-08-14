# 🛒 Ecommerce Conversion Prediction with XGBoost

This project predicts whether an online shopper will complete a purchase based on their session behavior using **XGBoost**.

---

## 📌 Project Overview

Understanding online shopper behavior is key for improving e-commerce conversions.  
In this project, we:
- Perform **Exploratory Data Analysis (EDA)** to uncover purchase behavior patterns.
- Train an **XGBoost classifier** to predict purchase intent.
- Tune hyperparameters for better performance.
- Visualize results with confusion matrix, ROC curve, and feature importance.

---

## 📂 Dataset

- **File**: `online_shoppers_intention.csv`
- **Source**: [UCI ML Repository – Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- **Target Variable**: `Revenue` (True = Purchase made, False = No purchase)

---

## 📓 Notebook

- **`HafsaNoorMuhammad_OnlineShoppersPurchaseIntentions_P5.ipynb`**  
  Steps included:
  - Data loading
  - EDA and visualization
  - Data preprocessing
  - XGBoost model training
  - Hyperparameter tuning
  - Model evaluation

---

## 🚀 Quick Start

### Install Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
````

### Run the Notebook

```bash
jupyter notebook HafsaNoorMuhammad_OnlineShoppersPurchaseIntentions_P5.ipynb
```

---

## 🧠 What is XGBoost?

**XGBoost** (Extreme Gradient Boosting) is a powerful machine learning library optimized for:

* **Speed** — parallel computation and efficient memory usage.
* **Accuracy** — built-in regularization to avoid overfitting.
* **Flexibility** — works for classification, regression, and ranking problems.

We chose XGBoost because:

* It handles **structured/tabular data** very well.
* It supports **feature importance ranking**.
* It’s highly efficient for large datasets.

---

## 🎯 Hyperparameter Tuning

We tuned:

* `max_depth` — Tree depth.
* `learning_rate` — Step size shrinkage.
* `n_estimators` — Number of boosting rounds.
* `subsample` — Fraction of rows per tree.
* `colsample_bytree` — Fraction of features per tree.
* `gamma` — Minimum loss reduction for a split.

**Methods used**:

* **GridSearchCV** — Exhaustive parameter search.
* **RandomizedSearchCV** — Faster random parameter sampling.

---

## 📊 Model Evaluation & Visuals

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)

---

## 📁 Project Structure

```
ecommerce-conversion-xgboost/
│
├── online_shoppers_intention.csv
├── HafsaNoorMuhammad_OnlineShoppersPurchaseIntentions_P5.ipynb
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── README.md
```

---

## 🤝 Contributions

**Author**: Hafsa Noor Muhammad
* 🌐 [LinkedIn](https://www.linkedin.com/in/hafsa-noor-muhammad-67b96331a/)
* 📁 [GitHub](https://github.com/HafsaNoorMuhammad26)
---

## 📚 References

* [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Scikit-learn Documentation](https://scikit-learn.org/)

```
