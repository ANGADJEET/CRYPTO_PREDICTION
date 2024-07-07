import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from data_processing import load_and_preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
#decision tree
from sklearn.tree import DecisionTreeRegressor


# svr gradient  boosting

def train_and_save_models(X, y, save_dir):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and save models
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    save_model(lr_model, os.path.join(save_dir, 'linear_regression_model.pkl'))
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    save_model(rf_model, os.path.join(save_dir, 'random_forest_model.pkl'))
    
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=10)
    gb_model.fit(X_train, y_train)
    save_model(gb_model, os.path.join(save_dir, 'gradient_boosting_model.pkl'))
    
    dt_model = DecisionTreeRegressor(max_depth=10)
    dt_model.fit(X_train, y_train)
    save_model(dt_model, os.path.join(save_dir, 'decision_tree_model.pkl'))

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


# Load data
directory = 'data'
crypto_data = load_and_preprocess_data(directory)

# #scale the data
# scaler = StandardScaler()
# crypto_data[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap']] = scaler.fit_transform(crypto_data[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap']])


# #normalize the data
# scaler = MinMaxScaler()
# crypto_data[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap']] = scaler.fit_transform(crypto_data[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap']])


# Split data into features and target
X = crypto_data[['Day', 'Month', 'Year', 'High', 'Low', 'Open', 'Volume', 'Marketcap']]
y = crypto_data['Close']

# Train and save models
models_dir = 'models'
train_and_save_models(X, y, models_dir)
