import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from fastapi.responses import StreamingResponse
import io
from src.config import MODEL_PARAMS


def build_model(input_shape,
                units=128,
                dropout=0.5,
                lr=0.000001):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def train_model(model,
                X_train, y_train,
                X_val, y_val,
                batch_size=32,
                epochs=100,
                patience=20):
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, feature_scaler, target_scaler, feature_names):
    test_pred_scaled = model.predict(X_test).flatten()
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    last_features_scaled = X_test[:, -1]
    last_features_actual = feature_scaler.inverse_transform(last_features_scaled)
    last_closes_actual = last_features_actual[:, feature_names.index('close')]

    predicted_prices = last_closes_actual * (1 + test_pred)
    actual_prices = last_closes_actual * (1 + test_actual)

    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)

    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return {'test_mae': mae, 'test_rmse': rmse}

def predict_next_day(model,last_sequence,feature_scaler,target_scaler,feature_names):        

    window_size = last_sequence.shape[0]                 
    seq_scaled = feature_scaler.transform(last_sequence.reshape(-1, last_sequence.shape[-1])).reshape(1, window_size, -1)                        

    pred_return_scaled = model.predict(seq_scaled, verbose=0)         
    pred_return = target_scaler.inverse_transform(pred_return_scaled)[0, 0]

    last_close_actual = last_sequence[-1, feature_names.index('close')]

    return last_close_actual * (1 + pred_return)




def directional_accuracy(model, X_test, y_test, target_scaler):
    test_pred_scaled = model.predict(X_test).flatten()
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    actual_direction = (test_actual[1:] > 0).astype(int)
    predicted_direction = (test_pred[:-1] > 0).astype(int)
    accuracy = np.mean(actual_direction == predicted_direction)

    return accuracy
