import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
from fastapi import FastAPI
from fastapi import Query
from typing import Optional

from src.data_pipeline import load_data, engineer_features, split_data, scale_data
from src.model_training import (
    build_model,
    train_model,
    evaluate_model,
    predict_next_day,
    directional_accuracy
)
from src.config import FEATURES_CONFIG, DEFAULT_START_DATE, DEFAULT_END_DATE, MODEL_PARAMS

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Stock Prediction API.",
        "usage": "Use the /predict endpoint with query parameters: symbol, start_date, end_date, window_size. Example: /predict?symbol=AAPL&start_date=2023-01-01&end_date=2024-01-01"
    }

def run_pipeline(symbol, start_date, end_date, window_size=15):
    raw_data = load_data(symbol, start_date, end_date)

    data_with_features, feature_names = engineer_features(raw_data, FEATURES_CONFIG)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_with_features, feature_names, window_size)

    last_sequence_unscaled = X_test[-1]

    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, feature_scaler, target_scaler = scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = build_model(input_shape=(window_size, len(feature_names)))

    history = train_model(model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)

    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Early stopping: model weights restored from epoch {best_epoch} with val_loss={history.history['val_loss'][best_epoch - 1]:.4f}")

    evaluate_model(model, X_test_scaled, y_test_scaled, feature_scaler, target_scaler, feature_names)

    dir_acc = directional_accuracy(model, X_test_scaled, y_test_scaled, target_scaler)

    next_price = predict_next_day(model, last_sequence_unscaled, feature_scaler, target_scaler, feature_names)
    print(f"Predicted next day closing price: {next_price:.2f}")

    test_pred_scaled = model.predict(X_test_scaled).flatten()
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    test_actual = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    last_features_scaled = X_test_scaled[:, -1]
    last_features_actual = feature_scaler.inverse_transform(last_features_scaled)
    last_closes_actual = last_features_actual[:, feature_names.index('close')]
    predicted_prices = last_closes_actual * (1 + test_pred)
    actual_prices = last_closes_actual * (1 + test_actual)

    return next_price, actual_prices.tolist(), predicted_prices.tolist(), dir_acc


@app.get("/predict")
def predict(
    symbol: str = Query(..., description="Stock symbol"),
    start_date: str = Query(DEFAULT_START_DATE, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(DEFAULT_END_DATE, description="End date (YYYY-MM-DD)"),
    window_size: int = Query(MODEL_PARAMS['window_size'], description="Window size")
):
    next_price, actual_prices, predicted_prices, dir_acc = run_pipeline(symbol, start_date, end_date, window_size)

    return {
        "predicted_next_close": next_price,
        "actual_prices": actual_prices,
        "predicted_prices": predicted_prices,
        "directional_accuracy": dir_acc
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)