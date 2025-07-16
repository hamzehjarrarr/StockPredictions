import numpy as np
import os
import random
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

class StockPredictionPipeline: #start was 2019-07-15
    def __init__(self, symbol="AAPL", start_date="2019-07-15", end_date="2025-07-16", #includes data upto 14 so im predicting 15
                 window_size=20, features_config=None, train_split=0.7, val_split=0.15, test_split=0.15):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.features_config = features_config or {
            'ohlcv': True, 'return_1d': True, 'return_5d': True,
            'volatility_10d': True, 'high_low_range': True, 'body_size': True,
            'volume_ratio_20d': True, 'sma_20d': True, 'rsi_14d': True,
            'macd':True, 'sma_10d': True, 'sma_50d': True, 'ema_10d': True, 'ema_20d': True, 'ema_50d': True,
            'stoch_osc': True
        }

        self.data = None
        self.processed_data = None
        self.scaler = None
        self.target_scaler = None
        self.model = None
        self.feature_names = []
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    def load_data(self):
        data = yf.Ticker(self.symbol).history(start=self.start_date, end=self.end_date)
        data.columns = [col.lower() for col in data.columns]
        self.data = data[['open', 'high', 'low', 'close', 'volume']].copy()
        return self.data

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    def calculate_stochastic_oscillator(self, df, k_period=14, d_period=3):
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        percent_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        percent_d = percent_k.rolling(window=d_period).mean()
        return percent_k, percent_d
    def engineer_features(self):
        df = self.data.copy()
        self.feature_names = []

        if self.features_config['ohlcv']:
            self.feature_names.extend(['open', 'high', 'low', 'close', 'volume'])

        if self.features_config['return_1d']:
            df['return_1d'] = df['close'].pct_change().shift(1)
            self.feature_names.append('return_1d')

        if self.features_config['return_5d']:
            df['return_5d'] = df['close'].pct_change(periods=5).shift(1)
            self.feature_names.append('return_5d')

        if self.features_config['volatility_10d']:
            df['volatility_10d'] = df['close'].rolling(window=10).std().shift(1)
            self.feature_names.append('volatility_10d')

        if self.features_config['high_low_range']:
            df['high_low_range'] = (df['high'] - df['low']).shift(1)
            self.feature_names.append('high_low_range')

        if self.features_config['body_size']:
            df['body_size'] = abs(df['close'] - df['open']).shift(1)
            self.feature_names.append('body_size')

        if self.features_config['volume_ratio_20d']:
            df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio_20d'] = (df['volume'] / df['volume_ma_20d']).shift(1)
            df.drop('volume_ma_20d', axis=1, inplace=True)
            self.feature_names.append('volume_ratio_20d')

        if self.features_config['sma_20d']:
            df['sma_20d'] = df['close'].rolling(window=20).mean().shift(1)
            self.feature_names.append('sma_20d')

        if self.features_config['rsi_14d']:
            df['rsi_14d'] = self.calculate_rsi(df['close'], 14).shift(1)
            self.feature_names.append('rsi_14d')
        if self.features_config.get('macd', False):
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            self.feature_names.append('macd')
            self.feature_names.append('macd_signal')

        if self.features_config.get('sma_10d'):
            df['sma_10d'] = df['close'].rolling(window=10).mean().shift(1)
            self.feature_names.append('sma_10d')
        if self.features_config['sma_50d']:
            df['sma_50d'] = df['close'].rolling(window=50).mean().shift(1)
            self.feature_names.append('sma_50d') 
        if self.features_config['ema_10d']:
            df['ema_10d'] = df['close'].ewm(span=10, adjust=False).mean().shift(1)
            self.feature_names.append('ema_10d')
        if self.features_config['ema_20d']:
            df['ema_20d'] = df['close'].ewm(span=20, adjust=False).mean().shift(1)
            self.feature_names.append('ema_20d')
        if self.features_config['ema_50d']:
            df['ema_50d'] = df['close'].ewm(span=50, adjust=False).mean().shift(1)
            self.feature_names.append('ema_50d')

        if self.features_config.get('stoch_osc', False):
            df['stoch_k'], df['stoch_d'] = self.calculate_stochastic_oscillator(df)
            self.feature_names.append('stoch_k')
            self.feature_names.append('stoch_d')


        df['target'] = df['close'].pct_change().shift(-1)
        self.processed_data = df.dropna()
        return self.processed_data

    def create_sequences(self, data):
        features = data[self.feature_names].values
        targets = data['target'].values
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:i + self.window_size])
            y.append(targets[i + self.window_size])
        return np.array(X), np.array(y)

    def split_data(self):
        data = self.processed_data.iloc[:-1].copy()
        X, y = self.create_sequences(data)
        total = len(X)
        train_end = int(total * self.train_split)
        val_end = int(total * (self.train_split + self.val_split))

        self.X_train, self.y_train = X[:train_end], y[:train_end]
        self.X_val, self.y_val = X[train_end:val_end], y[train_end:val_end]
        self.X_test, self.y_test = X[val_end:], y[val_end:]
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def scale_features(self):
        self.scaler = StandardScaler()
        X_train_reshaped = self.X_train.reshape(-1, self.X_train.shape[-1])
        self.X_train = self.scaler.fit_transform(X_train_reshaped).reshape(self.X_train.shape)
        self.X_val = self.scaler.transform(self.X_val.reshape(-1, self.X_val.shape[-1])).reshape(self.X_val.shape)
        self.X_test = self.scaler.transform(self.X_test.reshape(-1, self.X_test.shape[-1])).reshape(self.X_test.shape)

        self.target_scaler = StandardScaler()
        self.y_train = self.target_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_val = self.target_scaler.transform(self.y_val.reshape(-1, 1)).flatten()
        self.y_test = self.target_scaler.transform(self.y_test.reshape(-1, 1)).flatten()
    def build_model(self, units=128, dropout=0.3, lr=0.0005):
        model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(self.window_size, len(self.feature_names))),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        self.model = model
        return model

    def train_model(self, epochs=100, batch_size=32, patience=20): 
        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        return self.model.fit(self.X_train, self.y_train,
                              validation_data=(self.X_val, self.y_val),
                              epochs=epochs, batch_size=batch_size,
                              callbacks=[es], verbose=1)

    def predict_next_day(self, last_sequence):
        scaled_seq = self.scaler.transform(last_sequence.reshape(-1, last_sequence.shape[-1])).reshape(1, self.window_size, -1)
        pred_return_scaled = self.model.predict(scaled_seq, verbose=0)
        pred_return = self.target_scaler.inverse_transform(pred_return_scaled)[0, 0]
        last_close_scaled = last_sequence[-1][self.feature_names.index('close')]
        last_close_actual = self.scaler.inverse_transform(last_sequence)[-1][self.feature_names.index('close')]
        return last_close_actual * (1 + pred_return)

    def evaluate_model(self):
        test_pred_scaled = self.model.predict(self.X_test).flatten()
        test_pred = self.target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        test_actual = self.target_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()

        last_closes = self.scaler.inverse_transform(self.X_test[:, -1])[:, self.feature_names.index('close')]
        predicted_prices = last_closes * (1 + test_pred)
        actual_prices = last_closes * (1 + test_actual)

        mse = mean_squared_error(actual_prices, predicted_prices)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)

        print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {'test_mae': mae, 'test_rmse': rmse}

    def plot_predictions(self):
        test_pred_scaled = self.model.predict(self.X_test).flatten()
        test_pred = self.target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        test_actual = self.target_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()

        last_closes = self.scaler.inverse_transform(self.X_test[:, -1])[:, self.feature_names.index('close')]
        predicted_prices = last_closes * (1 + test_pred)
        actual_prices = last_closes * (1 + test_actual)

        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, label='Actual Closing Price')
        plt.plot(predicted_prices, label='Predicted Closing Price')
        plt.legend()
        plt.title(f'Actual vs. Predicted Closing Prices ({self.symbol})')
        plt.show()

if __name__ == "__main__":
    pipeline = StockPredictionPipeline(symbol="NVDA", window_size=15)
    pipeline.load_data()
    pipeline.engineer_features()
    pipeline.split_data()
    pipeline.scale_features()
    pipeline.build_model(units=128, dropout=0.5, lr= 0.000001)
    history = pipeline.train_model(batch_size=32, patience=20)
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Early stopping: model weights restored from epoch {best_epoch} with val_loss={history.history['val_loss'][best_epoch-1]:.4f}")
    pipeline.evaluate_model()
    pipeline.plot_predictions()
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    test_pred_scaled = pipeline.model.predict(pipeline.X_test).flatten()
    test_pred = pipeline.target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    test_actual = pipeline.target_scaler.inverse_transform(pipeline.y_test.reshape(-1, 1)).flatten()
    actual_direction = (test_actual[1:] > 0).astype(int)
    predicted_direction = (test_pred[:-1] > 0).astype(int)
    directional_accuracy = np.mean(actual_direction == predicted_direction)
    print(f'Directional Accuracy: {directional_accuracy:.2%}')
    next_price = pipeline.predict_next_day(pipeline.X_test[-1])
    print(f"Predicted next day closing price: {next_price:.2f}")