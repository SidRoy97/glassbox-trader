# stock-lens

NYSE stock direction classification, price forecasting, SHAP explainability, and LLM chatbot.

## Dataset
Download from https://www.kaggle.com/datasets/dgawlik/nyse
Place all CSVs inside the `data/` folder.

## Notebooks
Run in order:
1. `notebooks/01_data_loading.ipynb` — load, merge, inspect
2. `notebooks/02_feature_engineering.ipynb` — technical indicators, labels, split
3. `notebooks/03_classification.ipynb` — Random Forest and XGBoost
4. `notebooks/04_lstm_forecast.ipynb` — price regression
5. `notebooks/05_chatbot.ipynb` — SHAP + Gradio + Anthropic API

## Setup
pip install pandas pandas-ta scikit-learn xgboost tensorflow shap gradio anthropic yfinance matplotlib seaborn