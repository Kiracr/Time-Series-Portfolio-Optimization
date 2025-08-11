
import pandas as pd
import numpy as np
import os
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../reports')
os.makedirs(RESULTS_DIR, exist_ok=True)

def optimize_portfolio():
	assets = ['TSLA', 'BND', 'SPY']
	dfs = []
	for asset in assets:
		df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{asset}_processed.csv'))
		df['Date'] = pd.to_datetime(df['Date'])
		df = df.set_index('Date')
		dfs.append(df['Adj Close'])
	prices = pd.concat(dfs, axis=1)
	prices.columns = assets

	# Calculate expected returns and sample covariance
	mu = expected_returns.mean_historical_return(prices)
	S = risk_models.sample_cov(prices)

	# Efficient Frontier
	ef = EfficientFrontier(mu, S)
	raw_weights = ef.max_sharpe()
	cleaned_weights = ef.clean_weights()
	perf = ef.portfolio_performance(verbose=True)

	# Minimum volatility portfolio
	ef_min = EfficientFrontier(mu, S)
	ef_min.min_volatility()
	min_vol_weights = ef_min.clean_weights()
	min_vol_perf = ef_min.portfolio_performance(verbose=True)

	# Plot Efficient Frontier
	fig, ax = plt.subplots()
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
	plt.title('Efficient Frontier')
	plt.savefig(os.path.join(RESULTS_DIR, 'efficient_frontier.png'))
	plt.close()

	# Save weights and performance
	pd.DataFrame({'Max Sharpe': cleaned_weights, 'Min Volatility': min_vol_weights}).to_csv(os.path.join(RESULTS_DIR, 'portfolio_weights.csv'))

	return cleaned_weights, perf, min_vol_weights, min_vol_perf

if __name__ == '__main__':
	optimize_portfolio()
