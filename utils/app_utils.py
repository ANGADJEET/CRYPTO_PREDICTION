import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_processing import load_and_preprocess_data

def load_models():
    directory = 'data'
    crypto_data = load_and_preprocess_data(directory)

    X_clean = crypto_data  # Placeholder for outlier detection logic

    X = X_clean[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap']]
    y = X_clean['Close']

    models_dir = 'models'
    with open(os.path.join(models_dir, 'linear_regression_model.pkl'), 'rb') as f:
        lr_model = pickle.load(f)

    with open(os.path.join(models_dir, 'random_forest_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, lr_model, rf_model

def evaluate_models(X_test, y_test, lr_model, rf_model):
    lr_predictions = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)

    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

    return lr_predictions, lr_mse, lr_r2, rf_predictions, rf_mse, rf_r2

def make_predictions(lr_model, rf_model):
    import streamlit as st

    st.sidebar.header('Make New Predictions')

    day = st.sidebar.slider('Day', min_value=1, max_value=31, value=15)
    month = st.sidebar.slider('Month', min_value=1, max_value=12, value=6)
    year = st.sidebar.slider('Year', min_value=2020, max_value=2023, value=2022)
    high = st.sidebar.number_input('High', min_value=0.0, max_value=50000.0, value=25000.0)
    low = st.sidebar.number_input('Low', min_value=0.0, max_value=50000.0, value=20000.0)
    open_price = st.sidebar.number_input('Open', min_value=0.0, max_value=50000.0, value=23000.0)
    volume = st.sidebar.number_input('Volume', min_value=0, max_value=1000000, value=500000)
    marketcap = st.sidebar.number_input('Marketcap', min_value=0, max_value=1000000000, value=500000000)

    user_input = pd.DataFrame({
        'Day': [day],
        'Month': [month],
        'Year': [year],
        'High': [high],
        'Low': [low],
        'Open': [open_price],
        'Volume': [volume],
        'Marketcap': [marketcap]
    })

    lr_prediction = lr_model.predict(user_input)
    rf_prediction = rf_model.predict(user_input)

    st.sidebar.subheader('Prediction Results')
    st.sidebar.write(f'Linear Regression Prediction: ${lr_prediction[0]:,.2f}')
    st.sidebar.write(f'Random Forest Prediction: ${rf_prediction[0]:,.2f}')
