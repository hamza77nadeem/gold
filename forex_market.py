import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request
import io
import base64
from statsmodels.tsa.arima.model import ARIMA
import matplotlib

# Use 'Agg' backend for Matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Define the options for period and interval
period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
interval_options = ['1m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

@app.route("/", methods=["GET", "POST"])
def index():
    period = request.form.get("period", "5d")
    interval = request.form.get("interval", "1h")
    plot_url, prediction_plot_url, demand_text, prediction_text, latest_price_text = fetch_and_display_data(period, interval)
    return render_template("index.html", period_options=period_options, interval_options=interval_options, plot_url=plot_url, prediction_plot_url=prediction_plot_url, demand_text=demand_text, prediction_text=prediction_text, latest_price_text=latest_price_text)

def fetch_and_display_data(period, interval):
    gold = yf.Ticker("GC=F")
    data = gold.history(period=period, interval=interval)
    
    # Calculate OBV
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    
    # Determine colors for line segments
    colors = ['green' if data['Close'].iloc[i] > data['Close'].iloc[i-1] else 'red' for i in range(1, len(data))]
    colors.insert(0, 'green')  # First line segment
    
    # Plot price data with colored line segments
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot the closing prices with colors
    for i in range(1, len(data)):
        axs[0].plot(data.index[i-1:i+1], data['Close'].iloc[i-1:i+1], color=colors[i-1])
    axs[0].set_title('Gold Price and Volume Analysis')
    axs[0].set_ylabel('Price (USD)')
    axs[0].legend(['Close Price'], loc='upper left')
    
    # Plot the volume data with colored line segments
    for i in range(1, len(data)):
        axs[1].plot(data.index[i-1:i+1], data['Volume'].iloc[i-1:i+1], color=colors[i-1])
    axs[1].set_ylabel('Volume')
    axs[1].legend(['Volume'], loc='upper left')
    
    # Plot OBV
    axs[2].plot(data.index, data['OBV'], label='On-Balance Volume (OBV)', color='purple')
    axs[2].set_ylabel('OBV')
    axs[2].legend(loc='upper left')
    axs[2].set_xlabel('Time')
    
    # Save plot to a string in base64 format
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    # ARIMA model for prediction
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast_result = model_fit.get_forecast(steps=10)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Plot prediction
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(data.index, data['Close'], label='Actual')
    forecast_index = pd.date_range(start=data.index[-1], periods=10, freq='B')
    ax2.plot(forecast_index, forecast, label='Forecast')
    ax2.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.1)
    ax2.set_title('Gold Price Prediction')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    
    # Save prediction plot to a string in base64 format
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png')
    img2.seek(0)
    prediction_plot_url = base64.b64encode(img2.getvalue()).decode()
    plt.close(fig2)
    
    # Analyzing the results
    if data['OBV'].iloc[-1] > data['OBV'].iloc[0]:
        demand_text = "Overall, there was more buying demand."
    else:
        demand_text = "Overall, there was more selling demand."
    
    if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
        current_demand = "Buying"
        current_demand_text = "Currently, there is more buying demand."
    else:
        current_demand = "Selling"
        current_demand_text = "Currently, there is more selling demand."
    
    # Latest predicted price and its confidence interval
    predicted_price = forecast.iloc[-1]
    prediction_text = f"The predicted price for gold in the next period is ${predicted_price:.2f}."
    
    # Latest price
    latest_price = data['Close'].iloc[-1]
    latest_price_text = f"The latest gold price is ${latest_price:.2f}."

    return f"data:image/png;base64,{plot_url}", f"data:image/png;base64,{prediction_plot_url}", demand_text, prediction_text, latest_price_text

if __name__ == "__main__":
    app.run(debug=True)
