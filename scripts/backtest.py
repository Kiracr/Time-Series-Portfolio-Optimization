
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../reports')
os.makedirs(RESULTS_DIR, exist_ok=True)

def backtest(strategy_weights, start_date='2024-08-01', end_date='2025-07-31'):
	assets = ['TSLA', 'BND', 'SPY']
	dfs = []
	for asset in assets:
		df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{asset}_processed.csv'))
		df['Date'] = pd.to_datetime(df['Date'])
		df = df.set_index('Date')
		dfs.append(df['Adj Close'])
	prices = pd.concat(dfs, axis=1)
	prices.columns = assets
	prices = prices.loc[start_date:end_date]

	returns = prices.pct_change().dropna()

	# Strategy portfolio
	strat_returns = (returns * np.array([strategy_weights[a] for a in assets])).sum(axis=1)
	strat_cum = (1 + strat_returns).cumprod()

	# Benchmark: 60% SPY, 40% BND
	bench_weights = np.array([0, 0.4, 0.6])
	bench_returns = (returns * bench_weights).sum(axis=1)
	bench_cum = (1 + bench_returns).cumprod()

	# Plot
	plt.figure(figsize=(10,6))
	plt.plot(strat_cum, label='Strategy')
	plt.plot(bench_cum, label='Benchmark (60% SPY, 40% BND)')
	plt.title('Backtest: Cumulative Returns')
	plt.legend()
	plt.savefig(os.path.join(RESULTS_DIR, 'backtest_cumulative_returns.png'))
	plt.close()

	# Sharpe Ratio
	sharpe_strat = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
	sharpe_bench = bench_returns.mean() / bench_returns.std() * np.sqrt(252)

	# Total return
	total_strat = strat_cum.iloc[-1] - 1
	total_bench = bench_cum.iloc[-1] - 1

	print(f"Strategy Sharpe: {sharpe_strat:.2f}, Total Return: {total_strat:.2%}")
	print(f"Benchmark Sharpe: {sharpe_bench:.2f}, Total Return: {total_bench:.2%}")

	# Save results
	pd.DataFrame({'Strategy': strat_cum, 'Benchmark': bench_cum}).to_csv(os.path.join(RESULTS_DIR, 'backtest_cum_returns.csv'))

	return strat_cum, bench_cum

if __name__ == '__main__':
	# Example: use equal weights for demonstration
	strategy_weights = {'TSLA': 0.33, 'BND': 0.33, 'SPY': 0.34}
	backtest(strategy_weights)
