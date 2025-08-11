import yfinance as yf
import pandas as pd
import os

ASSETS = ['TSLA', 'BND', 'SPY']
START = '2015-07-01'
END = '2025-07-31'
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_and_save(asset):
    df = yf.download(asset, start=START, end=END)
    df.to_csv(os.path.join(RAW_DATA_DIR, f'{asset}.csv'))
    print(f"Saved {asset} data.")

if __name__ == '__main__':
    for asset in ASSETS:
        fetch_and_save(asset)
