# Stock Prediction Pipeline

This repository provides a modular pipeline for stock price prediction using deep learning and technical indicators. The project is implemented in Python and leverages yfinance, pandas, scikit-learn, and TensorFlow/Keras.

## Features

- Modular codebase for easy maintenance and extension
- Feature engineering with configurable technical indicators (OHLCV, returns, volatility, moving averages, RSI, MACD, stochastic oscillator, etc.)
- Sequence creation for time series modeling
- Data scaling and splitting (train/validation/test)
- Model training and evaluation (LSTM)
- FastAPI endpoint for predictions

## Project Structure

```
src/
    config.py            # Configuration for features, model, and data split
    data_pipeline.py     # Data loading, feature engineering, sequence creation, scaling
    model_training.py    # Model building, training, and evaluation
app.py                   # FastAPI server
...
```

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the pipeline:
   ```
   python main.py
   ```


## Running with Docker Compose

1. Build and start the services:
   ```
   docker compose up --build
   ```

2. Access the apps:
   - FastAPI: [http://localhost:8000](http://localhost:8000)
   - Streamlit: [http://localhost:8501](http://localhost:8501)

3. To stop the services:
   ```
   docker compose down
   ```

## Configuration

Edit `src/config.py` to adjust model parameters, feature selection, and data split ratios.


## Usage

- Modify `FEATURES_CONFIG` in `config.py` to select features.
- Use `app.py` to serve predictions via API.
- Use `app_streamlit.py` for the Streamlit UI (via Docker Compose).

## License

MIT License
