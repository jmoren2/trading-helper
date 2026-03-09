# Trading Helper

A personal swing trading analysis tool built with Streamlit. Uses a Hidden Markov Model (HMM) to classify market regimes and surface long entry signals across individual tickers or broad universes.

## Features

- **Dashboard** — HMM regime analysis, voting-based entry signal, backtest results, and equity curve for any ticker
- **Scanner** — Run the signal across a custom watchlist
- **Research** — Universe-wide scan (S&P 500, NASDAQ 100, S&P MidCap 400, S&P SmallCap 600, Crypto, ETFs) with optional 5-day momentum pre-filter
- **Parameter Optimizer** — Grid search over Stop Loss / Take Profit / Leverage / Min Votes combinations, scored by Calmar ratio
- **Presets** — Save and load strategy parameters per ticker

## Running locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

App runs at `http://localhost:8501`.

## Running with Docker

```bash
# Build and run locally
docker build -t trading-helper .
docker run -p 5050:8501 trading-helper

# Or pull the pre-built image
docker-compose up -d
```

App is exposed on port `5050`.

## How it works

1. OHLCV data is fetched from Yahoo Finance via `yfinance`
2. Technical indicators are computed (RSI, ADX, MACD, EMA50/200, Momentum, Volatility)
3. A `GaussianHMM` (7 states) is trained and states are labeled: **Bull Run**, **Bear**, **Crash**, **Sideways 1–N**
4. Entry fires when the current bar is in the Bull Run regime **and** passes a configurable number of indicator votes (out of 8)
5. Exit on Stop Loss, Take Profit, or regime flip to Bear/Crash
6. Test4
