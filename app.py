import streamlit as st
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from utils.data_processing import load_and_preprocess_data
from utils.visualization import plot_feature_importance, plot_correlation_heatmap, plot_moving_average, plot_volume_over_time, plot_price_distribution, plot_actual_vs_predicted_plotly_3d

# Main Streamlit app
def main():
    st.title('Cryptocurrency Price Prediction App')
    
    # Introduction
    st.write("""
    Welcome to the Cryptocurrency Price Prediction App! 
    Predict cryptocurrency prices using machine learning models trained on historical data sourced from Kaggle.
    """)
    
    # Load data
    directory = 'data'
    crypto_data = load_and_preprocess_data(directory)
    
    # Load trained models
    models_dir = 'models'
    with open(os.path.join(models_dir, 'linear_regression_model.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'random_forest_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'decision_tree_model.pkl'), 'rb') as f:
        dt_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'gradient_boosting_model.pkl'), 'rb') as f:
        gb_model = pickle.load(f)
    
    # Split data into features and target
    X = crypto_data[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap']]
    y = crypto_data['Close']
    
    # Sidebar - Prediction inputs
    st.sidebar.header('Set Prediction Inputs')
    day = st.sidebar.slider('Day', min_value=1, max_value=31, value=15)
    month = st.sidebar.slider('Month', min_value=1, max_value=12, value=6)
    year = st.sidebar.slider('Year', min_value=2010, max_value=2023, value=2022)
    high = st.sidebar.number_input('High Price', min_value=0.0, max_value=100000.0, value=50000.0)
    low = st.sidebar.number_input('Low Price', min_value=0.0, max_value=100000.0, value=45000.0)
    open_price = st.sidebar.number_input('Open Price', min_value=0.0, max_value=100000.0, value=48000.0)
    volume = st.sidebar.number_input('Volume', min_value=0, value=1000)
    marketcap = st.sidebar.number_input('Market Cap', min_value=0.0, max_value=1000000000.0, value=500000000.0)
    
    # Predict button
    if st.sidebar.button('Predict'):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Day': [day],
            'Month': [month],
            'Year': [year],
            'High': [high],
            'Low': [low],
            'Open': [open_price],
            'Volume': [volume],
            'Marketcap': [marketcap]
        })
        
        # Predict with models
        lr_prediction = lr_model.predict(input_data)[0]
        rf_prediction = rf_model.predict(input_data)[0]
        dt_prediction = dt_model.predict(input_data)[0]
        gb_prediction = gb_model.predict(input_data)[0]
        
        # Display predictions
        st.subheader('Predictions')
        st.write(f'Linear Regression Predicted Price: ${lr_prediction:.2f}')
        st.write(f'Random Forest Predicted Price: ${rf_prediction:.2f}')
        st.write(f'Decision Tree Predicted Price: ${dt_prediction:.2f}')
        st.write(f'Gradient Boosting Predicted Price: ${gb_prediction:.2f}')
    
    # Split data into training and testing sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate models
    lr_predictions = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    lr_expl_var = explained_variance_score(y_test, lr_predictions)
    
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_expl_var = explained_variance_score(y_test, rf_predictions)
    
    dt_predictions = dt_model.predict(X_test)
    dt_mse = mean_squared_error(y_test, dt_predictions)
    dt_r2 = r2_score(y_test, dt_predictions)
    dt_mae = mean_absolute_error(y_test, dt_predictions)
    dt_expl_var = explained_variance_score(y_test, dt_predictions)
    
    gb_predictions = gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_predictions)
    gb_r2 = r2_score(y_test, gb_predictions)
    gb_mae = mean_absolute_error(y_test, gb_predictions)
    gb_expl_var = explained_variance_score(y_test, gb_predictions)
    
    # Display evaluation metrics in a table
    st.subheader('Model Evaluation Metrics')
    eval_metrics = {
        'Linear Regression': [lr_mse, lr_r2, lr_mae, lr_expl_var],
        'Random Forest': [rf_mse, rf_r2, rf_mae, rf_expl_var],
        'Decision Tree': [dt_mse, dt_r2, dt_mae, dt_expl_var],
        'Gradient Boosting': [gb_mse, gb_r2, gb_mae, gb_expl_var]
    }
    eval_df = pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Mean Squared Error', 'R-squared', 'Mean Absolute Error', 'Explained Variance'])
    st.write(eval_df)
    
  # Generate and display actual vs predicted prices plot using Plotly 3D
    predictions_dict = {
        'Linear Regression': {'predictions': lr_predictions, 'r2': lr_r2},
        'Random Forest': {'predictions': rf_predictions, 'r2': rf_r2},
        'Decision Tree': {'predictions': dt_predictions, 'r2': dt_r2},
        'Gradient Boosting': {'predictions': gb_predictions, 'r2': gb_r2}
    }
    fig = plot_actual_vs_predicted_plotly_3d(y_test, predictions_dict)
    st.subheader('Actual vs Predicted Prices')
    st.plotly_chart(fig)
    st.write("This plot shows the comparison between actual and predicted closing prices for each regression model in 3D.")

    # Generate and display feature importance plot for Random Forest
    plot_feature_importance({
        'Random Forest': rf_model
    }, X_train, 'plots/feature_importance.png')
    st.subheader('Feature Importance')
    st.image('plots/feature_importance.png', use_column_width=True)
    st.write("This plot shows the importance of each feature in predicting the closing prices, as determined by the Random Forest model.")
    
    crypto_features = crypto_data[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap','Close']]
    # Generate and display correlation heatmap
    plot_correlation_heatmap(crypto_features)
    st.subheader('Correlation Heatmap')
    st.image('plots/correlation_heatmap.png', use_column_width=True)
    st.write("This heatmap displays the correlation between different features in the dataset. Higher absolute values indicate stronger correlations.")
    
    # Generate and display moving average plot
    plot_moving_average(crypto_data.set_index('Date'))
    st.subheader('30-Day Moving Average of Closing Prices')
    st.image('plots/moving_average.png', use_column_width=True)
    st.write("This plot shows the 30-day moving average of closing prices, highlighting the trend and smoothing out short-term fluctuations.")
    
    # Generate and display volume over time plot
    plot_volume_over_time(crypto_data.set_index('Date'))
    st.subheader('Trading Volume Over Time')
    st.image('plots/volume_over_time.png', use_column_width=True)
    st.write("This plot shows the trading volume over time, indicating the level of trading activity for the cryptocurrency.")
    
    # Generate and display price distribution plot
    plot_price_distribution(crypto_features)
    st.subheader('Distribution of Closing Prices')
    st.image('plots/price_distribution.png', use_column_width=True)
    st.write("This plot shows the distribution of closing prices, providing insights into the range and frequency of different price levels.")
    
    # Display dataset information
    st.subheader('Dataset Information')
    st.markdown("""
    This dataset was sourced from Kaggle and contains historical cryptocurrency price data. 
    It includes the following features:
    
    - Date: The date of the data point
    - Day: The day of the month
    - Month: The month of the year
    - Year: The year of the data point
    - High: The highest price of the cryptocurrency on that day
    - Low: The lowest price of the cryptocurrency on that day
    - Open: The opening price of the cryptocurrency on that day
    - Volume: The trading volume of the cryptocurrency on that day
    - Marketcap: The market capitalization of the cryptocurrency on that day
    - Close: The closing price of the cryptocurrency on that day
    """)

# Run the app
if __name__ == '__main__':
    main()
