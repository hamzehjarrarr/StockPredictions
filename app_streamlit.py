import streamlit as st
import requests
import datetime
import matplotlib.pyplot as plt
st.title("Stock Prediction App")

symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", value=datetime.date(2019, 7, 15))
max_market_date = datetime.date.today()
if datetime.datetime.now().hour < 16:
    max_market_date -= datetime.timedelta(days=1)
while max_market_date.weekday() > 4:
    max_market_date -= datetime.timedelta(days=1)

end_date = st.date_input(
    "End Date",
    value=max_market_date,
    max_value=max_market_date
)
window_size = st.number_input("Window Size", min_value=1, value=15)

if st.button("Predict Next Close Price"):
    params = {
        "symbol": symbol,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "window_size": window_size
    }
    response = requests.get("http://fastapi:8000/predict", params=params)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted next day closing price: {result['predicted_next_close']:.2f}")
        st.info(f"Directional Accuracy: {result['directional_accuracy']:.2%}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result['actual_prices'], label='Actual Closing Price')
        ax.plot(result['predicted_prices'], label='Predicted Closing Price')
        ax.legend()
        ax.set_title('Actual vs. Predicted Closing Prices')
        st.pyplot(fig)
    else:
        st.error("Error fetching prediction. Please check the server logs.")

