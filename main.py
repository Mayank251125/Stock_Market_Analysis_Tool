import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model # CHANGED: Corrected Keras import
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import yfinance as yf
import requests
import decimal # For handling Decimal type from yfinance

app = Flask(__name__)

# --- Load Model and Scaler ---
# Make sure you've run rnn.py once to save these files
model_path = 'models/google_stock_lstm_model.keras'
scaler_path = 'models/min_max_scaler.pkl'

model = None
scaler = None

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"LSTM model loaded successfully from {model_path}!")
    else:
        print(f"Error: Model file not found at {model_path}. Please run 'rnn.py' to save the model.")

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("MinMaxScaler loaded successfully!")
    else:
        print(f"Error: Scaler file not found at {scaler_path}. Please run 'rnn.py' to save the scaler.")

except Exception as e:
    print(f"An unexpected error occurred during model/scaler loading: {e}")

# --- Helper function to fetch real-time crypto data ---
def fetch_realtime_crypto_data(symbol):
    crypto_id_map = {
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum',
        'ADA-USD': 'cardano',
        'BNB-USD': 'binancecoin',
        'SOL-USD': 'solana',
        'XRP-USD': 'ripple',
        'DOGE-USD': 'dogecoin',
        'SHIB-USD': 'shiba-inu',
        'DOT-USD': 'polkadot',
        'LINK-USD': 'chainlink'
    }
    coin_id = crypto_id_map.get(symbol.upper())

    if not coin_id:
        return None

    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data and coin_id in data and 'usd' in data[coin_id]:
            price = data[coin_id]['usd']
            change_24hr = data[coin_id].get('usd_24h_change') # Returns None if not found

            return {
                'current_price': price,
                'change_24hr': change_24hr
            }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching crypto data for {symbol}: {e}")
    return None

# --- Helper function to fetch real-time stock data ---
def fetch_realtime_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        current_price = info.get('regularMarketPrice')
        previous_close = info.get('regularMarketPreviousClose')

        if current_price is not None and previous_close is not None:
            daily_change = ((current_price - previous_close) / previous_close) * 100
            return {
                'current_price': float(current_price),
                'change_24hr': float(daily_change)
            }
    except Exception as e:
        print(f"Error fetching real-time stock data for {symbol}: {e}")
    return None

# --- Routes ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_close', methods=['GET', 'POST'])
def predict_close():
    if request.method == 'POST':
        if not model or not scaler:
            return jsonify("Error: Prediction model or scaler not loaded on server. Please check server logs."), 500

        stock_ticker = request.form['stock_ticker'].upper()

        # Determine if it's a crypto ticker based on common suffixes
        is_crypto = '-USD' in stock_ticker or stock_ticker in ['BTC', 'ETH', 'ADA', 'BNB', 'SOL', 'XRP', 'DOGE', 'SHIB', 'DOT', 'LINK']

        required_timesteps = 60 # LSTM model requirement
        fetch_period = '90d' if not is_crypto else '70d' # Fetch slightly more data for robustness
        chart_period = '100d' # For historical chart display

        stock_data = pd.DataFrame() # Initialize stock_data as empty DataFrame
        
        try:
            # Fetch historical data
            stock_data = yf.download(stock_ticker, period=fetch_period, interval='1d')
            
            if stock_data.empty or 'Open' not in stock_data.columns:
                return jsonify(f"Error: Historical 'Open' price data not available for '{stock_ticker}'. Please check the ticker symbol or try a different one."), 400
            
            # Ensure we have enough data (at least 60 'Open' prices for the LSTM)
            # Take the last `required_timesteps` 'Open' prices
            inputs = stock_data['Open'].tail(required_timesteps).values
            
            if len(inputs) < required_timesteps:
                return jsonify(f"Error: Not enough historical trading days for '{stock_ticker}'. Need at least {required_timesteps} days, got {len(inputs)}. Try a longer-traded asset."), 400

            inputs = inputs.reshape(-1, 1) # Reshape for scaler
            inputs_scaled = scaler.transform(inputs)

            # Reshape for LSTM input (1 sample, 60 timesteps, 1 feature)
            X_predict = np.array([inputs_scaled[:, 0]])
            X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

            # Make prediction
            predicted_scaled_price = model.predict(X_predict)
            
            # Reshape predicted_scaled_price to be 2D for inverse_transform
            predicted_scaled_price_reshaped = predicted_scaled_price.reshape(-1, 1)

            # Inverse transform to get the actual price
            predicted_price = scaler.inverse_transform(predicted_scaled_price_reshaped)[0][0]
            print(f"Predicted price for {stock_ticker}: {predicted_price}")

        except Exception as e:
            # Catch errors during prediction itself (e.g., model predict issues)
            return jsonify(f"An error occurred during prediction for {stock_ticker}: {e}. Please check the stock ticker and try again."), 500

        # --- Fetch Company/Crypto Info ---
        company_info = {
            'longName': 'N/A',
            'symbol': stock_ticker,
            'marketCap': 'N/A',
            'volume': 'N/A',
            'sector': 'N/A', # Removed from frontend display but kept in backend
            'industry': 'N/A', # Removed from frontend display but kept in backend
            'website': 'N/A' # Removed from frontend display but kept in backend
        }
        real_time_crypto_data = None
        real_time_stock_data = None


        if is_crypto:
            real_time_crypto_data = fetch_realtime_crypto_data(stock_ticker)
            # For crypto, market cap and volume are fetched separately or may not be available via yfinance.info
            # Attempt to get marketCap and volume for crypto from yfinance info if available, else N/A
            try:
                ticker_info = yf.Ticker(stock_ticker).info
                company_info['marketCap'] = ticker_info.get('marketCap', 'N/A')
                company_info['volume'] = ticker_info.get('volume', 'N/A')
            except Exception as e:
                print(f"Warning: Could not fetch detailed yfinance info for crypto {stock_ticker}: {e}")

        else: # It's a traditional stock
            real_time_stock_data = fetch_realtime_stock_data(stock_ticker)
            try:
                ticker_info = yf.Ticker(stock_ticker).info
                for key in company_info.keys():
                    value = ticker_info.get(key, 'N/A')
                    # Handle Decimal objects for JSON serialization
                    if isinstance(value, decimal.Decimal):
                        company_info[key] = float(value)
                    # Convert large integers (like marketCap, volume) to string to prevent issues if too big
                    elif isinstance(value, int) and (key == 'marketCap' or key == 'volume'):
                        company_info[key] = str(value) # Convert large numbers to string
                    else:
                        company_info[key] = value
            except Exception as e:
                print(f"Warning: Could not fetch company info for stock {stock_ticker}: {e}")
                # Set all relevant fields to N/A if info fetch fails
                company_info = {k: 'N/A' if k not in ['symbol'] else stock_ticker for k in company_info.keys()}


        # --- Prepare Chart Data ---
        chart_labels = []
        chart_prices = []
        try:
            chart_data = yf.download(stock_ticker, period=chart_period, interval='1d')
            if not chart_data.empty and 'Close' in chart_data.columns:
                # Ensure no NaN values and convert to list of floats
                chart_data_cleaned = chart_data[['Close']].dropna()
                chart_labels = [d.strftime('%Y-%m-%d') for d in chart_data_cleaned.index]
                chart_prices = chart_data_cleaned['Close'].astype(float).values.flatten().tolist()
            else:
                print(f"No valid 'Close' data for chart for {stock_ticker}.")
        except Exception as e:
            print(f"Error fetching chart data for {stock_ticker}: {e}")

        # --- Return JSON Response ---
        return jsonify({
            'prediction': f"${predicted_price:.2f}",
            'company_info': company_info,
            'chart_data': {
                'labels': chart_labels,
                'prices': chart_prices
            },
            'is_crypto': is_crypto,
            'real_time_crypto_data': real_time_crypto_data,
            'real_time_stock_data': real_time_stock_data
        })

    return render_template('stock.html') # For GET request, display the form

if __name__ == '__main__':
    app.run(debug=True)