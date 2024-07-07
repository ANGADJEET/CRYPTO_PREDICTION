import os
import pandas as pd

def load_and_preprocess_data(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
