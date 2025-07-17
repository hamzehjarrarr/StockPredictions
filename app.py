import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_pipeline import load_data, engineer_features, split_data, scale_data
from src.model_training import (
    build_model,
    train_model,
    evaluate_model,
    plot_predictions,
    plot_training_loss,
    predict_next_day,
    directional_accuracy
)
from src.config import FEATURES_CONFIG, DEFAULT_START_DATE, DEFAULT_END_DATE, MODEL_PARAMS

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

def main(symbol="AAPL", start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, window_size=15):
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

    plot_predictions(model, X_test_scaled, y_test_scaled, feature_scaler, target_scaler, feature_names, symbol)

    plot_training_loss(history)

    directional_accuracy(model, X_test_scaled, y_test_scaled, target_scaler)

    next_price = predict_next_day(model, last_sequence_unscaled, feature_scaler, target_scaler, feature_names)
    print(f"Predicted next day closing price: {next_price:.2f}")

if __name__ == "__main__":
    main()
