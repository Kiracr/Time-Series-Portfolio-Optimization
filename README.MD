# Time Series Forecasting for Portfolio Management Optimization

## Introduction

Guide Me in Finance (GMF) Investments is a forward-thinking financial advisory firm specializing in personalized portfolio management. This project leverages advanced time series forecasting models to predict market trends, optimize asset allocation, and enhance portfolio performance for GMF clients.

## Objective

Apply time series forecasting to historical financial data (TSLA, BND, SPY) to enhance portfolio management strategies. This involves data extraction, cleaning, exploratory analysis, model building (ARIMA/SARIMA and LSTM), forecasting, portfolio optimization, and backtesting.

## Challenge Overview

- **Data Extraction:** Download historical data for TSLA, BND, and SPY from Yahoo Finance (2015-07-01 to 2025-07-31).
- **Data Preparation:** Clean, preprocess, and engineer features (returns, volatility, etc.).
- **EDA:** Visualize trends, check stationarity, and calculate risk metrics.
- **Modeling:** Build and evaluate ARIMA/SARIMA and LSTM models for forecasting.
- **Forecasting Analysis:** Compare model forecasts and interpret results.
- **Portfolio Optimization:** Use forecasts to optimize asset allocation using Modern Portfolio Theory.
- **Backtesting:** Simulate and compare strategy performance against a benchmark.

## Repository Structure

```
Time-Series-Forecasting-for-Portfolio-Management-Optimization/
│
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_arima_modeling.ipynb
│   ├── 03_lstm_modeling.ipynb
│   ├── 04_forecasting_analysis.ipynb
│   ├── 05_portfolio_optimization.ipynb
│   └── 06_backtesting.ipynb
├── scripts/
│   ├── fetch_data.py
│   ├── preprocess.py
│   ├── arima_model.py
│   ├── lstm_model.py
│   ├── portfolio_optimization.py
│   └── backtest.py
├── references/
│   └── key_references.md
└── tutorials/
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/fetch_data.py
```

### 3. Preprocess Data

```bash
python scripts/preprocess.py
```

### 4. Run Notebooks

Open each notebook in order and run all cells:
- `01_data_exploration.ipynb`
- `02_arima_modeling.ipynb`
- `03_lstm_modeling.ipynb`
- `04_forecasting_analysis.ipynb`
- `05_portfolio_optimization.ipynb`
- `06_backtesting.ipynb`

## Key Results

- Data cleaned and features engineered for TSLA, BND, SPY
- EDA revealed trends, volatility, and risk metrics
- ARIMA and LSTM models built and evaluated
- Efficient frontier and optimal portfolios identified
- Backtesting compared strategy vs. benchmark


## References

- [YFinance Documentation](https://github.com/ranaroussi/yfinance)
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/)
-