import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES_CONFIG, MODEL_PARAMS, DATA_SPLIT


def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(start=start_date, end=end_date)
    df.columns = [col.lower() for col in df.columns]
    return df[['open', 'high', 'low', 'close', 'volume']]


def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def engineer_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    feature_cols = []

    if config.get("ohlcv"):
        feature_cols += ['open', 'high', 'low', 'close', 'volume']

    if config.get("return_1d"):
        df['return_1d'] = df['close'].pct_change().shift(1)
        feature_cols.append('return_1d')

    if config.get("return_5d"):
        df['return_5d'] = df['close'].pct_change(5).shift(1)
        feature_cols.append('return_5d')

    if config.get("volatility_10d"):
        df['volatility_10d'] = df['close'].rolling(window=10).std().shift(1)
        feature_cols.append('volatility_10d')

    if config.get("high_low_range"):
        df['high_low_range'] = (df['high'] - df['low']).shift(1)
        feature_cols.append('high_low_range')

    if config.get("body_size"):
        df['body_size'] = abs(df['open'] - df['close']).shift(1)
        feature_cols.append('body_size')

    if config.get("volume_ratio_20d"):
        df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio_20d'] = (df['volume'] / df['volume_ma_20d']).shift(1)
        df.drop('volume_ma_20d', axis=1, inplace=True)
        feature_cols.append('volume_ratio_20d')

    if config.get("sma_20d"):
        df['sma_20d'] = df['close'].rolling(window=20).mean().shift(1)
        feature_cols.append('sma_20d')

    if config.get("rsi_14d"):
        df['rsi_14d'] = calculate_rsi(df).shift(1)
        feature_cols.append('rsi_14d')

    if config.get("macd"):
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd'] = df['macd'].shift(1)
        df['macd_signal'] = df['macd_signal'].shift(1)
        feature_cols.append('macd')
        feature_cols.append('macd_signal')

    if config.get("sma_10d"):
        df['sma_10d'] = df['close'].rolling(window=10).mean().shift(1)
        feature_cols.append('sma_10d')

    if config.get("sma_50d"):
        df['sma_50d'] = df['close'].rolling(window=50).mean().shift(1)
        feature_cols.append('sma_50d')

    if config.get("ema_10d"):
        df['ema_10d'] = df['close'].ewm(span=10, adjust=False).mean().shift(1)
        feature_cols.append('ema_10d')

    if config.get("ema_20d"):
        df['ema_20d'] = df['close'].ewm(span=20, adjust=False).mean().shift(1)
        feature_cols.append('ema_20d')

    if config.get("ema_50d"):
        df['ema_50d'] = df['close'].ewm(span=50, adjust=False).mean().shift(1)
        feature_cols.append('ema_50d')

    if config.get("stoch_osc"):
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_osc'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_osc'] = df['stoch_osc'].shift(1)
        feature_cols.append('stoch_osc')

    df['target'] = df['close'].pct_change().shift(-1)  

    df.dropna(inplace=True)

    return df, feature_cols


def create_sequences(data: pd.DataFrame, feature_cols: list[str], window_size: int) -> tuple[np.ndarray, np.ndarray]:
    features = data[feature_cols].values
    targets = data['target'].values
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)


def split_data(data: pd.DataFrame, feature_cols: list[str], window_size: int) -> tuple:
    trimmed_data = data.iloc[:-1].copy()
    X, y = create_sequences(trimmed_data, feature_cols, window_size)

    total = len(X)
    train_end = int(total * DATA_SPLIT['train_split'])
    val_end = int(total * (DATA_SPLIT['train_split'] + DATA_SPLIT['val_split']))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test, y_train, y_val, y_test):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, feature_scaler, target_scaler


def prepare_data(symbol: str, start_date: str, end_date: str,
                 features_config: dict, window_size: int, split_config: dict):
    df = load_data(symbol, start_date, end_date)
    df, feature_cols = engineer_features(df, features_config)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols, window_size)
    X_train, X_val, X_test, y_train, y_val, y_test, fs, ts = scale_data(X_train, X_val, X_test, y_train, y_val, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, fs, ts
