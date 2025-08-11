
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

ASSETS = ['TSLA', 'BND', 'SPY']

def preprocess(asset):

    file_path = os.path.join(RAW_DATA_DIR, f'{asset}.csv')
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = df.reset_index().rename(columns={'index': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])


    # Ensure 'Adj Close' exists (map from 'Close' if needed)
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    # Make sure 'Adj Close' is numeric
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

    # Sort and set index
    df = df.sort_values('Date').reset_index(drop=True)

    # Handle missing values (use ffill/bfill methods)
    df = df.ffill().bfill()

    # Feature engineering
    df['Return'] = df['Adj Close'].pct_change()
    df['LogReturn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Volatility'] = df['Return'].rolling(window=21).std()
    df['RollingMean'] = df['Adj Close'].rolling(window=21).mean()
    df['RollingStd'] = df['Adj Close'].rolling(window=21).std()

    # Outlier detection (z-score)
    df['z_score'] = (df['Return'] - df['Return'].mean()) / df['Return'].std()


    # Normalization (for ML)
    scaler = StandardScaler()
    scaled_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    # Only scale columns that exist in the DataFrame
    cols_to_scale = [col for col in scaled_cols if col in df.columns]
    # Convert to numeric and drop rows with NaN in these columns
    for col in cols_to_scale:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cols_to_scale)
    df[[f'{col}_scaled' for col in cols_to_scale]] = scaler.fit_transform(df[cols_to_scale])

    # Drop initial rows with NaN after rolling
    df = df.dropna().reset_index(drop=True)

    # Save processed data
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, f'{asset}_processed.csv'), index=False)
    print(f"Processed {asset}.")

    # Print basic stats
    print(df.describe())

if __name__ == '__main__':
    for asset in ASSETS:
        preprocess(asset)
