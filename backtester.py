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


def position_value(capital, risk_pct, sl_pct, leverage):
    """Size a position so that hitting the stop costs `risk_pct` of the account.

    `leverage` is a cap on exposure, not a fixed multiplier: a tight stop would
    otherwise imply a position many times the account.
    """
    if sl_pct <= 0:
        return capital * leverage
    return min(capital * risk_pct / sl_pct, capital * leverage)


def resolve_exit(row, entry_price, sl_pct, tp_pct, state, bear_crash_states):
    """Find this bar's exit, if any, as (reason, fill price).

    Stops and targets are intrabar, so they are checked against the bar's low and
    high rather than its close - a stop order does not wait for the closing print.
    A bar that gaps straight through a level fills at the open, not at the level.
    A regime flip is only knowable at the close, so it ranks last.
    """
    stop_price = entry_price * (1 - sl_pct)
    target_price = entry_price * (1 + tp_pct)
    open_, high, low = float(row["Open"]), float(row["High"]), float(row["Low"])

    # If a bar touches both, assume the stop came first - OHLC cannot say which
    # did, and that is the pessimistic reading.
    if low <= stop_price:
        return "Stop Loss", min(open_, stop_price)
    if high >= target_price:
        return "Take Profit", max(open_, target_price)
    if int(state) in bear_crash_states:
        return "Regime Flip", float(row["Close"])
    return None, None


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
    risk_pct=0.02,
):
    capital = float(initial_capital)
    in_position = False
    entry_price = None
    entry_time = None
    entry_reason = ""
    entry_size = 0.0
    cooldown_until = None

    trades = []
    equity_values = np.full(len(df), np.nan)
    equity_values[0] = capital

    for i in range(1, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        close = float(row["Close"])

        if in_position:
            exit_reason, exit_price = resolve_exit(
                row, entry_price, sl_pct, tp_pct, row["State"], bear_crash_states
            )

            if exit_reason:
                price_change = (exit_price - entry_price) / entry_price
                pnl = entry_size * price_change
                capital = max(capital + pnl, 0.0)
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": ts,
                    "Entry Price": round(entry_price, 2),
                    "Exit Price": round(exit_price, 2),
                    "Position ($)": round(entry_size, 2),
                    "Price Change %": round(price_change * 100, 3),
                    "Account Return %": round(pnl / (capital - pnl) * 100, 3)
                                        if capital - pnl > 0 else 0.0,
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
                    entry_size = position_value(capital, risk_pct, sl_pct, leverage)
                    entry_reason = ", ".join(label for label, passed in vote_results if passed)

        equity_values[i] = capital

    # Close open position at end of data
    if in_position:
        close = float(df["Close"].iloc[-1])
        price_change = (close - entry_price) / entry_price
        pnl = entry_size * price_change
        capital = max(capital + pnl, 0.0)
        trades.append({
            "Entry Time": entry_time,
            "Exit Time": df.index[-1],
            "Entry Price": round(entry_price, 2),
            "Exit Price": round(close, 2),
            "Position ($)": round(entry_size, 2),
            "Price Change %": round(price_change * 100, 3),
            "Account Return %": round(pnl / (capital - pnl) * 100, 3)
                                if capital - pnl > 0 else 0.0,
            "PnL ($)": round(pnl, 2),
            "Capital After ($)": round(capital, 2),
            "Entry Reason": entry_reason,
            "Exit Reason": "End of Data",
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Entry Time", "Exit Time", "Entry Price", "Exit Price",
                 "Position ($)", "Price Change %", "Account Return %", "PnL ($)",
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
    risk_pct=0.02,
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
            risk_pct=risk_pct,
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
