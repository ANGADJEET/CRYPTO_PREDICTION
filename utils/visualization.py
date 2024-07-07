import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go

def plot_actual_vs_predicted(y_test, predictions_dict, save_path):
    plt.figure(figsize=(12, 6))
    for model_name, results in predictions_dict.items():
        predictions = results['predictions']
        r2 = results['r2']
        plt.scatter(y_test, predictions, alpha=0.5, label=f'{model_name} (R-squared: {r2:.2f})')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(models, X_train, save_path):
    # Assuming rf_model is the Random Forest model
    rf_model = models['Random Forest']

    # Get feature importances from Random Forest
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns

    # Plot the feature importances of the Random Forest model
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances - Random Forest")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

# Function to plot moving average
def plot_moving_average(df, window=30):
    plt.figure(figsize=(12, 6))
    df['Close'].rolling(window=window).mean().plot()
    plt.title(f'{window}-Day Moving Average of Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Moving Average Price')
    plt.grid()
    plt.savefig('plots/moving_average.png')
    plt.close()

# Function to plot volume over time
def plot_volume_over_time(df):
    plt.figure(figsize=(12, 6))
    df['Volume'].plot()
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid()
    plt.savefig('plots/volume_over_time.png')
    plt.close()

# Function to plot price distribution
def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Close'], kde=True, log_scale=(True, False))
    plt.title('Distribution of Closing Prices')
    plt.xlabel('Closing Price (Log Scale)')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig('plots/price_distribution.png')
    plt.close()

# Function to plot actual vs predicted prices using Plotly
def plot_actual_vs_predicted_plotly(y_test, predictions_dict):
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='markers', name='Actual', marker=dict(color='black')))
    
    # Add predicted prices for each model
    for model_name, data in predictions_dict.items():
        fig.add_trace(go.Scatter(x=y_test.index, y=data['predictions'], mode='markers', name=model_name))
    
    fig.update_layout(
        title="Actual vs Predicted Prices",
        xaxis_title="Index",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white"
    )
    
    return fig

# Function to plot actual vs predicted prices using Plotly in 3D
def plot_actual_vs_predicted_plotly_3d(y_test, predictions_dict):
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter3d(
        x=y_test.index, y=y_test, z=[0]*len(y_test),
        mode='markers',
        name='Actual',
        marker=dict(size=5, color='black')
    ))
    
    # Add predicted prices for each model
    for i, (model_name, data) in enumerate(predictions_dict.items(), start=1):
        fig.add_trace(go.Scatter3d(
            x=y_test.index, y=data['predictions'], z=[i]*len(y_test),
            mode='markers',
            name=model_name,
            marker=dict(size=5)
        ))
    
    fig.update_layout(
        title="Actual vs Predicted Prices",
        scene=dict(
            xaxis_title='Index',
            yaxis_title='Price',
            zaxis_title='Model',
            zaxis=dict(
                tickvals=list(range(len(predictions_dict)+1)),
                ticktext=['Actual'] + list(predictions_dict.keys())
            )
        ),
        legend_title="Legend",
        template="plotly_white"
    )
    
    return fig