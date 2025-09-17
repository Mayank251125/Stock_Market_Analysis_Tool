```markdown
[View Project Demo Video](https://drive.google.com/file/d/1Hg1ZxplE20vCEb0p8oXF31_iKRYuH7f1/view?usp=sharing)
# Stock and Cryptocurrency Price Prediction Web App

A comprehensive web application that predicts stock and cryptocurrency prices using LSTM neural networks, featuring real-time market data, historical charts, and sentiment analysis.

## Features

- **📈 Price Prediction**: Predict next day's closing price using LSTM neural networks
- **📊 Real-time Data**: Live price and daily percentage change display
- **📉 Historical Charts**: Interactive charts showing last 100 days of price data
- **📰 Sentiment Analysis**: Market sentiment analysis based on news headlines (Bullish 🐂, Bearish 🐻, or Neutral)
- **🏢 Company Information**: Detailed asset information including market cap and volume
- **💹 Multi-Asset Support**: Both stocks and cryptocurrencies supported

## Technologies Used

### Backend
- **Flask** - Web framework
- **TensorFlow/Keras** - LSTM model implementation
- **Scikit-learn** - Data preprocessing and scaling

### Data APIs
- **yfinance** - Stock market data
- **CoinGecko API** - Cryptocurrency data
- **NewsApiClient** - News headlines for sentiment analysis

### Frontend
- **HTML/CSS/JavaScript** - Frontend structure and styling
- **Chart.js** - Interactive data visualizations
- **Bootstrap** - Responsive UI components (optional)


## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- News API key (free from [newsapi.org](https://newsapi.org/))

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd stock-crypto-predictor
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Create a `config.py` file in the root directory:

```python
# config.py
NEWS_API_KEY = 'your_news_api_key_here'
```

Or update in `main.py`:
```python
newsapi = NewsApiClient(api_key='your_actual_api_key')
```

### Step 5: Train the Model

```bash
python rnn.py
```

This will:
1. Load training data from `data/Google_Stock_Price_Train.csv`
2. Preprocess and scale the data
3. Train the LSTM model
4. Save model to `models/google_stock_lstm_model.keras`
5. Save scaler to `models/min_max_scaler.pkl`

### Step 6: Run the Application

```bash
python main.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Usage

1. **Home Page**: Enter a stock ticker (e.g., AAPL, GOOGL) or cryptocurrency symbol (e.g., BTC-USD, ETH-USD)
2. **Click Predict**: The app will fetch real-time data and generate predictions
3. **View Results**:
   - Predicted price for next trading day
   - Current market data
   - Historical price chart
   - News sentiment analysis
   - Company/asset information

## API Reference

### Supported Stock Symbols
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- TSLA (Tesla)
- And many more...

### Supported Crypto Symbols
- BTC-USD (Bitcoin)
- ETH-USD (Ethereum)
- ADA-USD (Cardano)
- And other major cryptocurrencies

## Model Details

### Architecture
- **Type**: Recurrent Neural Network (RNN)
- **Layers**: Long Short-Term Memory (LSTM) layers
- **Input**: Historical closing prices (60-day window)
- **Output**: Next day's predicted closing price

### Training Data
- Google stock price data from specific period
- Data normalized using MinMaxScaler
- 60-time step sliding window for sequences

## Troubleshooting

### Common Issues

1. **News API Errors**: Ensure your API key is valid and has not exceeded free tier limits
2. **Model Not Found**: Run `python rnn.py` to train and save the model first
3. **Data Fetching Issues**: Check internet connection and API availability

### Dependencies Issues

```bash
# If you encounter dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---


```

