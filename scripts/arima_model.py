
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../reports')
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_arima(asset, test_years=2):
	df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{asset}_processed.csv'))
	df['Date'] = pd.to_datetime(df['Date'])
	df = df.set_index('Date')
	series = df['Adj Close']

	# Chronological split
	split_date = series.index[-1] - pd.DateOffset(years=test_years)
	train = series[series.index < split_date]
	test = series[series.index >= split_date]

	# Parameter tuning
	print('Running auto_arima...')
	stepwise_model = auto_arima(train, start_p=1, start_q=1,
								max_p=3, max_q=3, m=1,
								start_P=0, seasonal=False,
								d=None, trace=True,
								error_action='ignore', suppress_warnings=True, stepwise=True)
	print(f'Best ARIMA order: {stepwise_model.order}')

	# Fit SARIMAX
	model = SARIMAX(train, order=stepwise_model.order, enforce_stationarity=False, enforce_invertibility=False)
	model_fit = model.fit(disp=False)

	# Forecast
	forecast = model_fit.forecast(steps=len(test))

	# Evaluation
	mae = mean_absolute_error(test, forecast)
	rmse = np.sqrt(mean_squared_error(test, forecast))
	mape = np.mean(np.abs((test - forecast) / test)) * 100

	print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')

	# Save results
	results = pd.DataFrame({'Actual': test, 'Forecast': forecast})
	results.to_csv(os.path.join(RESULTS_DIR, f'{asset}_arima_results.csv'))

	return model_fit, results

if __name__ == '__main__':
	train_arima('TSLA')
