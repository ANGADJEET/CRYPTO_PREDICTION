# Cryptocurrency Price Prediction App

Welcome to the Cryptocurrency Price Prediction App! This project utilizes machine learning models to predict cryptocurrency prices based on historical data sourced from Kaggle. The app provides interactive features for predicting prices using both Linear Regression and Random Forest models, evaluating model performance, and visualizing predictions.

## Features

- **Prediction Inputs**: Users can set prediction inputs such as day, month, year, high price, low price, open price, volume, and market cap using interactive sliders and input fields.
- **Predict Button**: Clicking the "Predict" button triggers predictions based on the selected inputs, displaying results for both Linear Regression and Random Forest models.
- **Model Evaluation**: Displays evaluation metrics including Mean Squared Error, R-squared, Mean Absolute Error, Median Absolute Error, and Explained Variance for both models.
- **Visualizations**: Provides visual feedback with actual vs predicted prices plot and feature importance plot for the Random Forest model.
- **Dataset Information**: Offers insights into the dataset used, including the features available and their descriptions.

## Getting Started

To run the app locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/cryptocurrency-price-prediction.git
   cd cryptocurrency-price-prediction
   ```

2. **Install dependencies**:

   Ensure you have Python installed. Then, install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   Execute the following command to start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   This will open a new tab in your default web browser with the Cryptocurrency Price Prediction App running locally.

## Project Structure

- **app.py**: Main application file containing the Streamlit app code.
- **data/**: Directory containing the dataset files sourced from Kaggle.
- **models/**: Directory storing the trained machine learning models (`.pkl` files).
- **plots/**: Directory where generated plots (actual vs predicted prices, feature importance) are saved.
- **utils/**:
  - **data_processing.py**: Module for data loading and preprocessing functions.
  - **visualization.py**: Module for visualization functions.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building interactive web apps with Python.
- **Pandas, NumPy**: Data manipulation and numerical computing.
- **scikit-learn**: Machine learning models and evaluation metrics.
- **Matplotlib, Seaborn**: Data visualization.

## Dataset Information

The dataset used in this project was sourced from Kaggle and contains historical cryptocurrency price data. It includes features such as Date, Day, Month, Year, High, Low, Open, Volume, Marketcap, and Close.

## Future Enhancements

- Incorporate more advanced machine learning models.
- Enhance user interface with additional interactive features.
- Deploy the app on a cloud platform for wider accessibility.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or feature requests, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to customize this README to fit your specific project details and style preferences. It should serve as a comprehensive introduction and guide for anyone interested in understanding, using, or contributing to your Cryptocurrency Price Prediction App.
