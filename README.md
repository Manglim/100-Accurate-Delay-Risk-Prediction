# 📦 Shipment Delay Risk Prediction using XGBoost

This project builds a machine learning pipeline to **predict shipment delays** using real-world logistics data and **XGBoost**, a powerful gradient boosting framework. The notebook includes complete preprocessing, feature engineering, hyperparameter tuning, model evaluation, and synthetic data prediction.

---

## 🚀 Features

- Preprocessing of logistics dataset with one-hot encoding and scaling
- Hyperparameter tuning using `GridSearchCV`
- Model evaluation using:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC
  - Confusion Matrix
- Feature importance visualization using XGBoost's built-in plot
- Synthetic prediction generation for 200 simulated shipments

---

## 📂 Dataset

The dataset used is sourced from:
`/kaggle/input/smart-logistics-supply-chain-dataset/smart_logistics_dataset.csv`

Features include environmental variables, asset IDs, shipment details, and delay labels.

---

## 🛠️ ML Stack

- **Model**: `XGBClassifier` from `xgboost`
- **Preprocessing**: `StandardScaler`, `pandas.get_dummies`
- **Hyperparameter Tuning**: `GridSearchCV`
- **Visualization**: `matplotlib`, `seaborn`

---

## 📊 Output Highlights

- 📈 Optimized model performance metrics (F1, ROC-AUC)
- 🔍 Confusion Matrix heatmap
- 🌟 Feature importance ranking
- 🧪 Delay predictions for synthetic shipment data (200 samples)
- 📉 Histogram of predicted delay probabilities

---

## 🔮 Synthetic Prediction Example

```python
synthetic_data[["Temperature", "Humidity", "Waiting_Time", "Traffic_Status", 
                "Predicted_Delay", "Predicted_Probability_Delay"]].head()
