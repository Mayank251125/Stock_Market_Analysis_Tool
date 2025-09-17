import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient
from textblob import TextBlob
import pickle
import os
import yfinance as yf
import requests
import decimal

app = Flask(__name__)

newsapi = NewsApiClient(api_key='45e35c6e7d62483cafb700977f31f972')

model_path = 'models/google_stock_lstm_model.keras'
model = None

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"LSTM model loaded successfully from {model_path}!")
    else:
        print(f"Error: Model file not found at {model_path}. Please run 'rnn.py' to save the model.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")

# --- Helper function to fetch real-time crypto data ---
def fetch_realtime_crypto_data(symbol):
    crypto_id_map = {
        'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'ADA-USD': 'cardano',
        'BNB-USD': 'binancecoin', 'SOL-USD': 'solana', 'XRP-USD': 'ripple',
        'DOGE-USD': 'dogecoin', 'SHIB-USD': 'shiba-inu', 'DOT-USD': 'polkadot',
        'LINK-USD': 'chainlink'
    }
    coin_id = crypto_id_map.get(symbol.upper())
    if not coin_id:
        return None
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and coin_id in data and 'usd' in data[coin_id]:
            price = data[coin_id]['usd']
            change_24hr = data[coin_id].get('usd_24h_change')
            # Keeping raw float value for client-side formatting
            return {'current_price': price, 'change_24hr': change_24hr}
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
            # Keeping raw float value for client-side formatting
            return {'current_price': float(current_price), 'change_24hr': float(daily_change)}
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
        if not model:
            return jsonify("Error: Prediction model not loaded on server. Please check server logs."), 500
        stock_ticker = request.form['stock_ticker'].upper()
        is_crypto = '-USD' in stock_ticker or stock_ticker in ['BTC', 'ETH', 'ADA', 'BNB', 'SOL', 'XRP', 'DOGE', 'SHIB', 'DOT', 'LINK']
        required_timesteps = 90
        chart_period = '100d'

# --- ADD THIS CODE ---
        # --- Sentiment Analysis Section ---
        sentiment_data = {'score': 0, 'label': "Neutral"}
        try:
            print(f"Fetching news for {stock_ticker}...")
            # Fetch news articles related to the stock ticker
            articles = newsapi.get_everything(q=stock_ticker, language='en', sort_by='relevancy', page_size=100)
            
            # Calculate sentiment for each article title
            sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles'] if article['title']]
            
            # Calculate the average sentiment
            if sentiments:
                average_sentiment = sum(sentiments) / len(sentiments)
                sentiment_data['score'] = f"{average_sentiment:.3f}"
                if average_sentiment > 0.1:
                    sentiment_data['label'] = "Bullish 🐂"
                elif average_sentiment < -0.1:
                    sentiment_data['label'] = "Bearish 🐻"
            print(f"Sentiment analysis for {stock_ticker}: Score={sentiment_data['score']} ({sentiment_data['label']})")

        except Exception as e:
            print(f"Could not fetch or process news for {stock_ticker}: {e}")
            sentiment_data['label'] = "Unavailable"
        # --- End of Sentiment Analysis Section ---
# --- END OF ADDED CODE ---

        try:
            fetch_period = '120d'
            stock_data_full = yf.download(stock_ticker, period=fetch_period, interval='1d')
            if stock_data_full.empty or 'Open' not in stock_data_full.columns:
                return jsonify(f"Error: Historical 'Open' price data not available for '{stock_ticker}'."), 400
            open_prices = stock_data_full['Open'].values.reshape(-1, 1)
            if len(open_prices) < required_timesteps:
                return jsonify(f"Error: Not enough historical trading days for '{stock_ticker}'. Need at least {required_timesteps} days, got {len(open_prices)}. Try a longer-traded asset."), 400
            dynamic_scaler = MinMaxScaler(feature_range=(0, 1))
            dynamic_scaler.fit(open_prices)
            inputs_for_prediction = open_prices[-required_timesteps:]
            inputs_scaled = dynamic_scaler.transform(inputs_for_prediction)
            X_predict = np.reshape(inputs_scaled, (1, required_timesteps, 1))
            predicted_scaled_price = model.predict(X_predict)
            predicted_price = dynamic_scaler.inverse_transform(predicted_scaled_price)[0][0]
            print(f"Predicted price for {stock_ticker}: {predicted_price}")
        except Exception as e:
            import traceback
            traceback.print_exc() 
            return jsonify(f"An error occurred during prediction for {stock_ticker}: {e}. Please check the stock ticker and try again."), 500
        company_info = {
            'longName': 'N/A', 'symbol': stock_ticker, 'marketCap': 'N/A',
            'volume': 'N/A', 'sector': 'N/A', 'industry': 'N/A', 'website': 'N/A'
        }
        real_time_crypto_data = None
        real_time_stock_data = None
        if is_crypto:
            # FIXED: Corrected function call from fetch_real_time_crypto_data to fetch_realtime_crypto_data
            real_time_crypto_data = fetch_realtime_crypto_data(stock_ticker)
            try:
                ticker_info = yf.Ticker(stock_ticker).info
                company_info['marketCap'] = ticker_info.get('marketCap', 'N/A')
                company_info['volume'] = ticker_info.get('volume', 'N/A')
            except Exception as e:
                print(f"Warning: Could not fetch detailed yfinance info for crypto {stock_ticker}: {e}")
        else:
            real_time_stock_data = fetch_realtime_stock_data(stock_ticker)
            try:
                ticker_info = yf.Ticker(stock_ticker).info
                for key in company_info.keys():
                    value = ticker_info.get(key, 'N/A')
                    if isinstance(value, decimal.Decimal):
                        company_info[key] = float(value)
                    elif isinstance(value, int) and (key == 'marketCap' or key == 'volume'):
                        company_info[key] = str(value)
                    else:
                        company_info[key] = value
            except Exception as e:
                print(f"Warning: Could not fetch company info for stock {stock_ticker}: {e}")
                company_info = {k: 'N/A' if k not in ['symbol'] else stock_ticker for k in company_info.keys()}
        chart_labels = []
        chart_prices = []
        try:
            chart_data = yf.download(stock_ticker, period=chart_period, interval='1d')
            if not chart_data.empty and 'Close' in chart_data.columns:
                chart_data_cleaned = chart_data[['Close']].dropna()
                chart_labels = [d.strftime('%Y-%m-%d') for d in chart_data_cleaned.index]
                chart_prices = chart_data_cleaned['Close'].astype(float).values.flatten().tolist()
            else:
                print(f"No valid 'Close' data for chart for {stock_ticker}.")
        except Exception as e:
            print(f"Error fetching chart data for {stock_ticker}: {e}")
        return jsonify({
            'prediction': float(predicted_price), 
            'company_info': company_info,
            'chart_data': {
                'labels': chart_labels,
                'prices': chart_prices
            },
            'is_crypto': is_crypto,
            'real_time_crypto_data': real_time_crypto_data,
            'real_time_stock_data': real_time_stock_data,
# --- ADD THIS LINE ---
            'sentiment': sentiment_data
# --- END OF ADDED LINE ---
        })
    return render_template('stock.html')
if __name__ == '__main__':
    app.run(debug=True)