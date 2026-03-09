import io

import pandas as pd
import requests
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
    "sp500":   "S&P 500",
    "ndx100":  "NASDAQ 100",
    "sp400":   "S&P MidCap 400",
    "sp600":   "S&P SmallCap 600",
    "crypto":  "Top 25 Crypto",
    "etfs":    "Popular ETFs",
}


_WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; trading-helper/1.0)"}


def _scrape_wiki_tickers(url: str, min_rows: int = 50) -> list[str]:
    try:
        html = requests.get(url, headers=_WIKI_HEADERS, timeout=15).text
        tables = pd.read_html(io.StringIO(html))
        candidates = []
        for t in tables:
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = [" ".join(str(c) for c in col).strip() for col in t.columns]
            for col in ("Symbol", "Ticker", "Ticker symbol"):
                if col in t.columns and len(t) >= min_rows:
                    candidates.append((len(t), col, t))
                    break
        if not candidates:
            return []
        _, col, t = max(candidates, key=lambda x: x[0])
        return sorted(t[col].astype(str).str.replace(".", "-", regex=False).tolist())
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    return _scrape_wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ) or ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
          "BRK-B", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX"]


@st.cache_data(ttl=86400, show_spinner=False)
def get_ndx100_tickers() -> list[str]:
    return _scrape_wiki_tickers(
        "https://en.wikipedia.org/wiki/Nasdaq-100"
    ) or ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
          "AVGO", "ASML", "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTC"]


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp400_tickers() -> list[str]:
    return _scrape_wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    )


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp600_tickers() -> list[str]:
    return _scrape_wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    )


def get_universe_tickers(selected_keys: list[str]) -> list[str]:
    tickers: list[str] = []
    if "sp500"  in selected_keys: tickers += get_sp500_tickers()
    if "ndx100" in selected_keys: tickers += get_ndx100_tickers()
    if "sp400"  in selected_keys: tickers += get_sp400_tickers()
    if "sp600"  in selected_keys: tickers += get_sp600_tickers()
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
