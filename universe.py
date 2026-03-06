import pandas as pd
import streamlit as st
import yfinance as yf

TOP_CRYPTO: list[str] = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "MATIC-USD", "UNI-USD", "LTC-USD", "BCH-USD", "ATOM-USD",
    "XLM-USD", "ALGO-USD", "NEAR-USD", "ICP-USD", "FIL-USD",
    "APT-USD", "ARB-USD", "OP-USD", "INJ-USD", "SUI20947-USD",
]

POPULAR_ETFS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "VOO", "GLD", "SLV", "TLT", "HYG",
    "XLK", "XLF", "XLE", "XLV", "XLU",
    "ARKK", "BITO", "SQQQ", "TQQQ", "UVXY",
]

UNIVERSE_LABELS: dict[str, str] = {
    "sp500":  "S&P 500",
    "ndx100": "NASDAQ 100",
    "crypto": "Top 25 Crypto",
    "etfs":   "Popular ETFs",
}


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        return sorted(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                "BRK-B", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX"]


@st.cache_data(ttl=86400, show_spinner=False)
def get_ndx100_tickers() -> list[str]:
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            attrs={"id": "constituents"},
        )
        return sorted(tables[0]["Ticker"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
                "AVGO", "ASML", "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTC"]


def get_universe_tickers(selected_keys: list[str]) -> list[str]:
    tickers: list[str] = []
    if "sp500"  in selected_keys: tickers += get_sp500_tickers()
    if "ndx100" in selected_keys: tickers += get_ndx100_tickers()
    if "crypto" in selected_keys: tickers += TOP_CRYPTO
    if "etfs"   in selected_keys: tickers += POPULAR_ETFS
    seen: set[str] = set()
    result: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def batch_5d_change(tickers: tuple[str, ...]) -> dict[str, float]:
    """Batch-download 5d daily close prices, return {ticker: pct_change}.
    Accepts tuple (not list) so it is hashable for st.cache_data."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            list(tickers), period="5d", interval="1d",
            progress=False, group_by="ticker", auto_adjust=True,
        )
    except Exception:
        return {}

    results: dict[str, float] = {}

    if len(tickers) == 1:
        closes = raw["Close"].dropna()
        if len(closes) >= 2:
            results[tickers[0]] = float(
                (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
            )
        return results

    for ticker in tickers:
        try:
            closes = raw["Close"][ticker].dropna()
            if len(closes) >= 2:
                results[ticker] = float(
                    (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
                )
        except (KeyError, TypeError):
            continue

    return results
