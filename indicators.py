import numpy as np
import pandas as pd


def _wilder(series, period):
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    rs = _wilder(gain, period) / _wilder(loss, period).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_adx(high, low, close, period=14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=close.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=close.index
    )

    atr = _wilder(tr, period)
    plus_di = 100 * _wilder(plus_dm, period) / atr
    minus_di = 100 * _wilder(minus_dm, period) / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return _wilder(dx, period)


def add_indicators(df):
    df = df.copy()

    df["Returns"] = df["Close"].pct_change()

    df["RSI"] = compute_rsi(df["Close"])

    # Momentum: % price change over 14 periods
    df["Momentum"] = (df["Close"] / df["Close"].shift(14) - 1) * 100

    # Volatility: rolling 20-period std of returns (%)
    df["Volatility"] = df["Returns"].rolling(20).std() * 100

    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()

    df["ADX"] = compute_adx(df["High"], df["Low"], df["Close"])

    df["EMA50"] = df["Close"].ewm(span=50, min_periods=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, min_periods=200, adjust=False).mean()

    ema12 = df["Close"].ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, min_periods=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, min_periods=9, adjust=False).mean()

    return df
