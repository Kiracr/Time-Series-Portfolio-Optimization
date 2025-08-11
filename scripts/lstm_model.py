
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../reports')
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_sequences(data, seq_length):
	xs, ys = [], []
	for i in range(len(data) - seq_length):
		xs.append(data[i:i+seq_length])
		ys.append(data[i+seq_length])
	return np.array(xs), np.array(ys)

def train_lstm(asset, seq_length=30, epochs=20, batch_size=32, test_years=2):
	df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{asset}_processed.csv'))
	df['Date'] = pd.to_datetime(df['Date'])
	df = df.set_index('Date')
	series = df['Adj Close'].values.reshape(-1, 1)

	# Chronological split
	split_idx = len(series) - 252 * test_years  # ~252 trading days per year
	train, test = series[:split_idx], series[split_idx:]

	# Scale data
	scaler = MinMaxScaler()
	train_scaled = scaler.fit_transform(train)
	test_scaled = scaler.transform(test)

	# Create sequences
	X_train, y_train = create_sequences(train_scaled, seq_length)
	X_test, y_test = create_sequences(np.concatenate([train_scaled[-seq_length:], test_scaled]), seq_length)

	# Build model
	model = Sequential([
		LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
		Dropout(0.2),
		LSTM(50),
		Dropout(0.2),
		Dense(1)
	])
	model.compile(optimizer='adam', loss='mse')

	# Train
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

	# Predict
	y_pred_scaled = model.predict(X_test)
	y_pred = scaler.inverse_transform(y_pred_scaled)
	y_test_inv = scaler.inverse_transform(y_test)

	# Evaluation
	mae = mean_absolute_error(y_test_inv, y_pred)
	rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
	mape = np.mean(np.abs((y_test_inv - y_pred) / y_test_inv)) * 100

	print(f'LSTM MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')

	# Save results
	results = pd.DataFrame({'Actual': y_test_inv.flatten(), 'Forecast': y_pred.flatten()})
	results.to_csv(os.path.join(RESULTS_DIR, f'{asset}_lstm_results.csv'))

	return model, results

if __name__ == '__main__':
	train_lstm('TSLA')
