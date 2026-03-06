import itertools

import numpy as np
import pandas as pd

INDICATOR_COLS = [
    "RSI", "Momentum", "Volatility", "Volume_SMA20",
    "ADX", "EMA50", "EMA200", "MACD", "MACD_Signal",
]


def count_votes(row):
    """Return list of (label, passed) for all 8 voting conditions."""
    return [
        ("RSI < 90",        bool(row["RSI"] < 90)),
        ("Momentum > 1%",   bool(row["Momentum"] > 1.0)),
        ("Volatility < 6%", bool(row["Volatility"] < 6.0)),
        ("Volume > SMA20",  bool(row["Volume"] > row["Volume_SMA20"])),
        ("ADX > 25",        bool(row["ADX"] > 25)),
        ("Price > EMA50",   bool(row["Close"] > row["EMA50"])),
        ("Price > EMA200",  bool(row["Close"] > row["EMA200"])),
        ("MACD > Signal",   bool(row["MACD"] > row["MACD_Signal"])),
    ]


def run_backtest(
    df,
    bull_state,
    bear_crash_states,
    initial_capital=10_000,
    leverage=2.5,
    sl_pct=0.03,
    tp_pct=0.15,
    cooldown_hours=48,
    votes_required=7,
):
    capital = float(initial_capital)
    in_position = False
    entry_price = None
    entry_time = None
    entry_reason = ""
    cooldown_until = None

    trades = []
    equity_values = np.full(len(df), np.nan)
    equity_values[0] = capital

    for i in range(1, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        close = float(row["Close"])

        if in_position:
            price_change = (close - entry_price) / entry_price
            exit_reason = None

            if int(row["State"]) in bear_crash_states:
                exit_reason = "Regime Flip"
            elif price_change <= -sl_pct:
                exit_reason = "Stop Loss"
            elif price_change >= tp_pct:
                exit_reason = "Take Profit"

            if exit_reason:
                leveraged_return = price_change * leverage
                pnl = capital * leveraged_return
                capital = max(capital + pnl, 0.0)
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": ts,
                    "Entry Price": round(entry_price, 2),
                    "Exit Price": round(close, 2),
                    "Price Change %": round(price_change * 100, 3),
                    "Leveraged Return %": round(leveraged_return * 100, 3),
                    "PnL ($)": round(pnl, 2),
                    "Capital After ($)": round(capital, 2),
                    "Entry Reason": entry_reason,
                    "Exit Reason": exit_reason,
                })
                in_position = False
                cooldown_until = ts + pd.Timedelta(hours=cooldown_hours)

        else:
            if cooldown_until is not None and ts <= cooldown_until:
                equity_values[i] = capital
                continue

            if int(row["State"]) == bull_state:
                if any(pd.isna(row[col]) for col in INDICATOR_COLS):
                    equity_values[i] = capital
                    continue

                vote_results = count_votes(row)
                votes = sum(passed for _, passed in vote_results)
                if votes >= votes_required:
                    in_position = True
                    entry_price = close
                    entry_time = ts
                    entry_reason = ", ".join(label for label, passed in vote_results if passed)

        equity_values[i] = capital

    # Close open position at end of data
    if in_position:
        close = float(df["Close"].iloc[-1])
        price_change = (close - entry_price) / entry_price
        leveraged_return = price_change * leverage
        pnl = capital * leveraged_return
        capital = max(capital + pnl, 0.0)
        trades.append({
            "Entry Time": entry_time,
            "Exit Time": df.index[-1],
            "Entry Price": round(entry_price, 2),
            "Exit Price": round(close, 2),
            "Price Change %": round(price_change * 100, 3),
            "Leveraged Return %": round(leveraged_return * 100, 3),
            "PnL ($)": round(pnl, 2),
            "Capital After ($)": round(capital, 2),
            "Entry Reason": entry_reason,
            "Exit Reason": "End of Data",
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Entry Time", "Exit Time", "Entry Price", "Exit Price",
                 "Price Change %", "Leveraged Return %", "PnL ($)",
                 "Capital After ($)", "Entry Reason", "Exit Reason"]
    )

    equity_series = pd.Series(equity_values, index=df.index).ffill().fillna(initial_capital)

    total_return_pct = (capital - initial_capital) / initial_capital * 100
    bh_return_pct = (
        (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0]))
        / float(df["Close"].iloc[0]) * 100
    )
    alpha = total_return_pct - bh_return_pct
    win_rate = float((trades_df["PnL ($)"] > 0).mean() * 100) if not trades_df.empty else 0.0

    rolling_max = equity_series.cummax()
    max_drawdown = float(((equity_series - rolling_max) / rolling_max * 100).min())

    metrics = {
        "Total Return %": round(total_return_pct, 2),
        "Final Capital": round(capital, 2),
        "Alpha vs B&H %": round(alpha, 2),
        "Buy & Hold Return %": round(bh_return_pct, 2),
        "Win Rate %": round(win_rate, 1),
        "Total Trades": len(trades_df),
        "Max Drawdown %": round(max_drawdown, 2),
    }

    return trades_df, metrics, equity_series


def grid_search(
    df,
    bull_state,
    bear_crash_states,
    sl_values,
    tp_values,
    leverage_values,
    votes_values,
    cooldown_hours=48,
    progress_cb=None,
):
    combos = list(itertools.product(sl_values, tp_values, leverage_values, votes_values))
    results = []

    for idx, (sl, tp, lev, votes) in enumerate(combos):
        if progress_cb:
            progress_cb(idx, len(combos))

        _, metrics, _ = run_backtest(
            df, bull_state, bear_crash_states,
            sl_pct=sl / 100,
            tp_pct=tp / 100,
            leverage=lev,
            votes_required=votes,
            cooldown_hours=cooldown_hours,
        )

        ret = metrics["Total Return %"]
        dd = metrics["Max Drawdown %"]
        # Calmar-like score: return / max drawdown (higher = better)
        score = ret / max(abs(dd), 1.0)

        results.append({
            "SL %": sl,
            "TP %": tp,
            "Leverage": lev,
            "Min Votes": votes,
            "Total Return %": ret,
            "Alpha %": metrics["Alpha vs B&H %"],
            "Win Rate %": metrics["Win Rate %"],
            "Trades": metrics["Total Trades"],
            "Max Drawdown %": dd,
            "Score": round(score, 3),
        })

    return pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
