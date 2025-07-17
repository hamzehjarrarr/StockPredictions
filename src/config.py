
MODEL_PARAMS = {
    'window_size': 15,
    'units': 128,
    'dropout': 0.5,
    'lr': 0.000001,
    'batch_size': 32,
    'patience': 20,
    'epochs': 100
}

DATA_SPLIT = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

FEATURES_CONFIG = {
    'ohlcv': True,
    'return_1d': True,
    'return_5d': True,
    'volatility_10d': True,
    'high_low_range': True,
    'body_size': True,
    'volume_ratio_20d': True,
    'sma_20d': True,
    'rsi_14d': True,
    'macd': True,
    'sma_10d': True,
    'sma_50d': True,
    'ema_10d': True,
    'ema_20d': True,
    'ema_50d': True,
    'stoch_osc': True
}

DEFAULT_START_DATE = "2019-07-15"
DEFAULT_END_DATE = "2025-07-16"