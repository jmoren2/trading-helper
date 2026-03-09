import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PRESETS_FILE = Path(__file__).parent / "presets.json"


def load_presets() -> dict:
    if PRESETS_FILE.exists():
        with open(PRESETS_FILE) as f:
            return json.load(f)
    return {}


def save_preset(ticker, sl, tp, lev, votes, cool, interval="1h", period="730d"):
    presets = load_presets()
    presets[ticker] = {"sl": sl, "tp": tp, "leverage": lev, "votes": votes, "cooldown": cool,
                       "interval": interval, "period": period}
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def delete_preset(ticker):
    presets = load_presets()
    presets.pop(ticker, None)
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def load_quick_tickers() -> list:
    presets = load_presets()
    return presets.get("__quick__", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])


def save_quick_tickers(tickers: list):
    presets = load_presets()
    presets["__quick__"] = tickers
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def load_recent_tickers() -> list:
    return load_presets().get("__recent__", [])


def add_recent_ticker(ticker: str, max_recent: int = 15):
    presets = load_presets()
    recent = presets.get("__recent__", [])
    if ticker in recent:
        recent.remove(ticker)
    recent.insert(0, ticker)
    presets["__recent__"] = recent[:max_recent]
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)

from backtester import INDICATOR_COLS, count_votes, grid_search, run_backtest
from data_loader import load_data
from hmm_model import train_hmm
from indicators import add_indicators

# ── Universe data (inlined to avoid working-directory import issues) ───────────
import yfinance as _yf

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
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        # Find the table with a Symbol column
        for t in tables:
            if "Symbol" in t.columns:
                return sorted(t["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
        "KO", "AVGO", "COST", "WMT", "DIS", "NFLX", "ADBE", "CRM", "AMD",
        "INTC", "QCOM", "TXN", "ORCL", "IBM", "GS", "MS", "BAC", "WFC",
        "UNH", "LLY", "TMO", "DHR", "AMGN", "GILD", "ISRG", "SYK", "MDT",
        "CAT", "DE", "HON", "MMM", "GE", "BA", "LMT", "RTX", "NOC",
        "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "NEE", "DUK", "SO",
    ]


@st.cache_data(ttl=86400, show_spinner=False)
def get_ndx100_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            for col in ("Ticker", "Symbol"):
                if col in t.columns:
                    return sorted(t[col].str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
        "ASML", "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTC", "TXN",
        "CSCO", "CMCSA", "INTU", "AMGN", "HON", "SBUX", "MDLZ", "GILD",
        "ADI", "REGN", "VRTX", "ISRG", "LRCX", "SNPS", "CDNS", "MU",
        "PANW", "KLAC", "MELI", "MNST", "FTNT", "CTAS", "ADP", "ORLY",
        "PCAR", "CPRT", "KDP", "FAST", "DXCM", "ODFL", "ROST", "BIIB",
        "IDXX", "MRNA", "ZS", "CRWD", "DDOG", "ABNB", "COIN", "PLTR",
    ]


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
    if not tickers:
        return {}
    try:
        raw = _yf.download(
            list(tickers), period="5d", interval="1d",
            progress=False, auto_adjust=True,
        )
    except Exception:
        return {}

    results: dict[str, float] = {}

    # Flatten MultiIndex columns (yfinance returns (field, ticker) MultiIndex for multi-ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"]  # DataFrame: rows=dates, cols=tickers
        for ticker in tickers:
            try:
                closes = close_df[ticker].dropna()
                if len(closes) >= 2:
                    results[ticker] = float(
                        (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
                    )
            except (KeyError, TypeError):
                continue
    else:
        # Single ticker — raw["Close"] is a Series
        closes = raw["Close"].dropna()
        if len(closes) >= 2:
            results[tickers[0]] = float(
                (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
            )

    return results

st.set_page_config(page_title="Swing Trading Dashboard", layout="wide")

st.markdown("""
<style>
/* Make the quick-select radio list scrollable */
div[data-testid="stRadio"] > div {
    max-height: 180px;
    overflow-y: auto;
    padding-right: 4px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_research_scan(
    universe_keys: tuple[str, ...],
    top_n: int,
    scan_interval: str,
    scan_period: str,
    min_votes: int,
    skip_stage1: bool = False,
) -> tuple[list[dict], dict[str, float], list[tuple[str, str]]]:
    """
    Two-stage research scan.
    Stage 1: batch 5d price change for the full universe, take top_n movers.
             Skipped when skip_stage1=True — all tickers go to Stage 2.
    Stage 2: run full HMM on each candidate — returns ALL results, not just LONG.
    Returns: results, stage1_changes (top_n slice, empty if skipped), errors
    """
    all_tickers = get_universe_tickers(list(universe_keys))
    if not all_tickers:
        return [], {}, []

    if skip_stage1:
        top_tickers = all_tickers
        stage1_changes = {}
    else:
        # Stage 1 — rank by 5d momentum
        changes = batch_5d_change(tuple(all_tickers))
        sorted_tickers = sorted(changes, key=lambda t: changes[t], reverse=True)
        top_tickers = sorted_tickers[:top_n]
        stage1_changes = {t: changes[t] for t in top_tickers}

    # Stage 2 — full HMM on top N
    results: list[dict] = []
    errors: list[tuple[str, str]] = []

    for t in top_tickers:
        try:
            s_df, s_bull, s_bears, s_labels, _ = get_hmm_analysis(t, scan_interval, scan_period)
            s_row = s_df.iloc[-1]
            s_state = int(s_row["State"])
            is_long = s_state == s_bull

            votes_ready = not any(pd.isna(s_row.get(col)) for col in INDICATOR_COLS)
            if votes_ready:
                vote_results = count_votes(s_row)
                votes_count = sum(p for _, p in vote_results)
            else:
                votes_count = None

            entry_ready = is_long and votes_count is not None and votes_count >= min_votes

            tail = 720 if scan_interval == "1h" else 30
            close_series = s_df["Close"].tail(tail)

            results.append({
                "ticker":       t,
                "is_long":      is_long,
                "regime":       s_labels[s_state],
                "votes":        votes_count,
                "entry_ready":  entry_ready,
                "price":        float(s_row["Close"]),
                "change_5d":    changes.get(t, 0.0),
                "close_series": close_series,
            })
        except Exception as e:
            errors.append((t, str(e)))

    # Sort: entry-ready first, then LONG, then by 5d change desc
    results.sort(key=lambda r: (not r["entry_ready"], not r["is_long"], -r["change_5d"]))
    return results, stage1_changes, errors


@st.cache_data(ttl=3600, show_spinner=False)
def get_hmm_analysis(ticker, interval="1h", period="730d"):
    """Expensive part — cached for 1h. Only re-runs on ticker/interval/period change or refresh."""
    df = load_data(ticker, period=period, interval=interval)
    df = add_indicators(df)
    df, bull_state, bear_crash_states, state_labels, summary_df = train_hmm(df)
    return df, bull_state, bear_crash_states, state_labels, summary_df


def regime_bg_color(state, bull_state, bear_crash_states):
    if state == bull_state:
        return "rgba(34,197,94,0.13)"
    if state in bear_crash_states:
        return "rgba(239,68,68,0.13)"
    return "rgba(150,150,150,0.04)"


def make_price_chart(df, bull_state, bear_crash_states, n_hours=500):
    plot_df = df.tail(n_hours).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.03,
    )

    plot_df["_grp"] = (plot_df["State"] != plot_df["State"].shift()).cumsum()
    for _, grp in plot_df.groupby("_grp"):
        state = int(grp["State"].iloc[0])
        fig.add_vrect(
            x0=grp.index[0], x1=grp.index[-1],
            fillcolor=regime_bg_color(state, bull_state, bear_crash_states),
            layer="below", line_width=0,
        )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],   close=plot_df["Close"],
        name="Price",
        increasing_line_color="#22c55e", increasing_fillcolor="#22c55e",
        decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["EMA50"],
        name="EMA 50", line=dict(color="#f59e0b", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["EMA200"],
        name="EMA 200", line=dict(color="#a78bfa", width=1.2),
    ), row=1, col=1)

    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(plot_df["Close"], plot_df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["Volume"],
        marker_color=vol_colors, showlegend=False, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#0f0f0f",
        font=dict(color="#e0e0e0"),
        height=580,
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis2=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
        yaxis2=dict(gridcolor="#2a2a2a"),
    )
    return fig


def make_mini_chart(close_series: pd.Series) -> go.Figure:
    color = "#22c55e" if close_series.iloc[-1] >= close_series.iloc[0] else "#ef4444"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close_series.index, y=close_series, mode="lines",
        line=dict(color=color, width=1.5), showlegend=False,
        hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="#0f0f0f", paper_bgcolor="#0f0f0f",
        font=dict(color="#e0e0e0"), height=90,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def make_equity_chart(equity_series, initial_capital, df, leverage):
    bh = initial_capital * (df["Close"] / float(df["Close"].iloc[0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_series.index, y=equity_series,
        name=f"Strategy ({leverage}x lev)",
        line=dict(color="#4f8ef7", width=2),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh,
        name="Buy & Hold", line=dict(color="#888", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#0f0f0f",
        font=dict(color="#e0e0e0"),
        height=280,
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a", tickprefix="$"),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=20, b=40),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")

    # active_ticker is the source of truth. _ticker_text is the widget key.
    # Sync must happen BEFORE the text_input renders to avoid the
    # "cannot modify session_state after widget is instantiated" error.
    if "active_ticker" not in st.session_state:
        st.session_state["active_ticker"] = "BTC-USD"
    if "_ticker_text" not in st.session_state:
        st.session_state["_ticker_text"] = st.session_state["active_ticker"]

    # If active_ticker was changed externally (e.g. radio), push it to the
    # text input key before the widget renders.
    if st.session_state["_ticker_text"] != st.session_state["active_ticker"]:
        st.session_state["_ticker_text"] = st.session_state["active_ticker"]

    def _on_text_change():
        val = st.session_state["_ticker_text"].strip().upper()
        if val:
            st.session_state["active_ticker"] = val

    st.text_input("Ticker", key="_ticker_text", on_change=_on_text_change)
    ticker = st.session_state["active_ticker"]
    if st.session_state.get("_last_viewed") != ticker:
        add_recent_ticker(ticker)
        st.session_state["_last_viewed"] = ticker

    # Quick select + Recently viewed side by side
    quick_tickers = load_quick_tickers()
    recent_tickers = [t for t in load_recent_tickers() if t != ticker]

    _qs_col, _rv_col = st.columns(2)

    with _qs_col:
        st.caption("Quick select")
        if quick_tickers:
            _qs_index = quick_tickers.index(ticker) if ticker in quick_tickers else None
            _qs_selected = st.radio(
                "Quick select", quick_tickers, index=_qs_index,
                label_visibility="collapsed",
            )
            if _qs_selected and _qs_selected != ticker:
                st.session_state["active_ticker"] = _qs_selected
                st.rerun()

    with _rv_col:
        st.caption("Recently viewed")
        for _rt in recent_tickers:
            st.markdown(f"<div style='padding:2px 0;font-size:13px'>{_rt}</div>", unsafe_allow_html=True)

    # Add / remove from quick select
    if ticker in quick_tickers:
        _rm_col, _ = st.columns([3, 2])
        if _rm_col.button("✕ Remove", key="rm_quick", use_container_width=True, width="stretch"):
            quick_tickers.remove(ticker)
            save_quick_tickers(quick_tickers)
            st.rerun()
    else:
        if st.button(f"+ Add {ticker} to quick select", use_container_width=True):
            quick_tickers.append(ticker)
            save_quick_tickers(quick_tickers)
            st.rerun()

    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()

    st.divider()
    st.subheader("Strategy Parameters")

    # Load preset for current ticker if one exists
    presets = load_presets()
    if ticker in presets:
        p = presets[ticker]
        st.success(f"Saved preset found for {ticker}")
        if st.button(f"Load {ticker} preset", use_container_width=True):
            st.session_state["apply_sl"] = p["sl"]
            st.session_state["apply_tp"] = p["tp"]
            st.session_state["apply_lev"] = p["leverage"]
            st.session_state["apply_votes"] = p["votes"]
            st.session_state["apply_cool"] = p["cooldown"]
            st.session_state["apply_interval"] = p.get("interval", "1h")
            st.session_state["apply_period"] = p.get("period", "730d")
            st.rerun()

    sl_pct = st.slider("Stop Loss %", 1, 15,
                        st.session_state.get("apply_sl", 3), 1, key="sl_slider",
                        help="Exit the trade if price drops this much from your entry price. "
                             "Example: 5% SL on a $100 entry exits at $95. "
                             "Tighter = fewer big losses but more frequent stop-outs. "
                             "BTC moves 2-4% hourly often, so very tight SLs get hit constantly.") / 100
    tp_pct = st.slider("Take Profit %", 5, 50,
                        st.session_state.get("apply_tp", 15), 5, key="tp_slider",
                        help="Exit the trade and lock in profit if price rises this much from entry. "
                             "Example: 20% TP on a $100 entry closes at $120. "
                             "Higher TP = bigger wins but fewer of them. "
                             "Note: leverage multiplies this gain (e.g. 20% TP × 2x leverage = 40% account gain).") / 100
    leverage = st.slider("Leverage", 1.0, 5.0,
                          st.session_state.get("apply_lev", 2.5), 0.5, key="lev_slider",
                          help="Multiplier applied to your gains AND losses. "
                               "2x leverage means a 10% price move = 20% account change. "
                               "Higher leverage = higher reward but dramatically higher risk. "
                               "At 3x leverage, a -33% price drop wipes your entire position.")
    votes_required = st.slider("Min Votes Required", 5, 8,
                                st.session_state.get("apply_votes", 7), 1, key="votes_slider",
                                help="How many of the 8 technical indicators must agree before entering a trade. "
                                     "7/8 = very strict, fewer but higher-confidence entries. "
                                     "5/8 = more lenient, more trades but noisier signals. "
                                     "The 8 conditions are: RSI, Momentum, Volatility, Volume, ADX, EMA50, EMA200, MACD.")
    cooldown = st.slider("Cooldown (hours)", 0, 96,
                          st.session_state.get("apply_cool", 48), 12, key="cool_slider",
                          help="After closing any trade (win or loss), the strategy waits this many hours "
                               "before it can enter a new one. "
                               "Prevents immediately re-entering a bad trade. "
                               "48h = 2 full days of forced patience after every exit.")

    _INTERVAL_OPTIONS = {
        "1d / 5y    (Daily, 5 years)":   ("1d", "5y"),
        "1h / 730d  (Hourly, ~2 years)": ("1h", "730d"),
    }
    _default_interval = st.session_state.get("apply_interval", "1h")
    _interval_default_idx = 0 if _default_interval == "1h" else 1
    _interval_label = st.selectbox(
        "Data Interval / Period",
        list(_INTERVAL_OPTIONS.keys()),
        index=_interval_default_idx,
        key="interval_selectbox",
        help="Hourly data gives finer swing signals but is capped at ~730 days by yfinance.\n\n"
             "Daily data covers 5 years — useful for slower-moving stocks (AAPL, TSLA) where "
             "hourly noise is irrelevant and you want more historical regime examples.\n\n"
             "Changing this re-trains the HMM on the new dataset.",
    )
    interval, period = _INTERVAL_OPTIONS[_interval_label]

    if st.button(f"Save settings for {ticker}", width="stretch", key="save_preset_btn"):
        save_preset(
            ticker,
            sl=int(sl_pct * 100),
            tp=int(tp_pct * 100),
            lev=leverage,
            votes=votes_required,
            cool=cooldown,
            interval=interval,
            period=period,
        )
        st.success(f"Saved {ticker} preset!")
        st.rerun()

    # Manage all saved presets
    presets = load_presets()
    if presets:
        st.divider()
        with st.expander(f"Saved Presets ({sum(1 for k in presets if not k.startswith('__'))})"):
            for t, p in presets.items():
                if t.startswith("__"):
                    continue
                col_l, col_d = st.columns([3, 1])
                interval_str = p.get("interval", "1h")
                col_l.markdown(
                    f"**{t}** — SL {p['sl']}% / TP {p['tp']}% / "
                    f"{p['leverage']}x / {p['votes']} votes / {interval_str}"
                )
                if col_d.button("X", key=f"del_{t}"):
                    delete_preset(t)
                    st.rerun()

    st.divider()
    st.caption(f"Data: {period} {interval} — yfinance")
    st.caption("Model: GaussianHMM, 7 states")

# ── Load HMM (cached) ─────────────────────────────────────────────────────────
with st.spinner(f"Loading HMM analysis for {ticker} ..."):
    try:
        df, bull_state, bear_crash_states, state_labels, summary_df = get_hmm_analysis(ticker, interval, period)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# Run backtest with current slider params (fast, not cached)
trades_df, metrics, equity_series = run_backtest(
    df, bull_state, bear_crash_states,
    sl_pct=sl_pct, tp_pct=tp_pct, leverage=leverage,
    votes_required=votes_required, cooldown_hours=cooldown,
)

current_row = df.iloc[-1]
current_state = int(current_row["State"])
regime_name = state_labels[current_state]
is_long = current_state == bull_state
signal = "LONG" if is_long else "CASH"

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_scan, tab_research, tab_opt, tab_guide = st.tabs(
    ["Dashboard", "Scanner", "Research", "Parameter Optimizer", "How to Use"]
)

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_dash:
    _dash_title_col, _dash_news_col = st.columns([6, 1])
    _dash_title_col.title(f"Swing Trading — {ticker}")
    _dash_news_col.link_button(
        "Yahoo News", f"https://finance.yahoo.com/quote/{ticker}/news",
        use_container_width=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal", signal,
              help="What the strategy would do right now.\n\n"
                   "LONG = all entry conditions are met — the model would open a trade.\n\n"
                   "CASH = sitting out. Either the regime isn't Bull Run, not enough votes pass, "
                   "or we're in the 48h cooldown after a recent exit.")
    c2.metric("Regime", regime_name,
              help="The market state the HMM model detected for the most recent hourly candle.\n\n"
                   "Bull Run = historically the highest-returning regime. Only state that allows entries.\n\n"
                   "Bear / Crash = the two worst-returning regimes. Any open trade exits immediately.\n\n"
                   "Sideways 1-4 = neutral regimes with mixed returns. No action taken.")
    c3.metric("Price", f"${float(current_row['Close']):,.2f}",
              help="Closing price of the most recent hourly candle from yfinance. "
                   "Not a live tick — typically 1-2 hours delayed depending on the exchange.")
    rsi_val = current_row.get("RSI")
    c4.metric("RSI", f"{float(rsi_val):.1f}" if pd.notna(rsi_val) else "N/A",
              help="Relative Strength Index (14-period). Ranges from 0 to 100.\n\n"
                   "Above 70 = overbought (price rose fast, may pull back).\n\n"
                   "Below 30 = oversold (price dropped fast, may bounce).\n\n"
                   "The strategy only requires RSI < 90 — a very loose filter that just blocks extreme blowoff tops.")

    st.divider()

    # Voting conditions
    indicators_ready = not any(pd.isna(current_row.get(col)) for col in INDICATOR_COLS)
    with st.expander("Voting Conditions (current bar)", expanded=True):
        if not indicators_ready:
            st.info("Indicators still warming up.")
        else:
            vote_results = count_votes(current_row)
            votes_passed = sum(passed for _, passed in vote_results)
            entry_ready = is_long and votes_passed >= votes_required

            status = f"{votes_passed}/8 conditions met (need {votes_required})"
            if entry_ready:
                st.success(f"{status} — Entry signal active")
            elif is_long:
                st.warning(f"{status} — In Bull regime but insufficient votes")
            else:
                st.info(f"{status} — Not in Bull regime")

            cols = st.columns(4)
            for i, (label, passed) in enumerate(vote_results):
                mark = "✓" if passed else "✗"
                color = "green" if passed else "red"
                cols[i % 4].markdown(
                    f"<span style='color:{color}'>{mark}</span> {label}",
                    unsafe_allow_html=True,
                )

    # Chart
    st.subheader("Price Chart — Last 500 Hours")
    st.caption("Background: green = Bull Run | red = Bear/Crash | grey = Neutral")
    st.plotly_chart(make_price_chart(df, bull_state, bear_crash_states), width="stretch")

    # Backtest metrics
    st.subheader(f"Backtest Results  ({period} {interval}, $10k starting capital)")
    m1, m2, m3, m4 = st.columns(4)
    pnl = metrics["Final Capital"] - 10_000
    pnl_str = f"-${abs(pnl):,.0f}" if pnl < 0 else f"+${pnl:,.0f}"
    m1.metric("Total Return", f"{metrics['Total Return %']:+.2f}%", pnl_str,
              help="Total profit or loss over the entire 730-day backtest period, starting with $10,000.\n\n"
                   "The dollar amount below shows the actual gain/loss in cash. "
                   "This already includes the leverage multiplier.")
    m2.metric(
        "Alpha vs Buy & Hold",
        f"{metrics['Alpha vs B&H %']:+.2f}%",
        f"vs B&H {metrics['Buy & Hold Return %']:+.2f}%",
        help="How much better or worse the strategy performed compared to simply buying BTC on day 1 and holding.\n\n"
             "Positive alpha = the strategy beat doing nothing. Negative = you'd have been better off just holding.\n\n"
             "The delta shows the buy & hold return for reference."
    )
    m3.metric(
        "Win Rate",
        f"{metrics['Win Rate %']:.1f}%",
        f"{metrics['Total Trades']} trades",
        delta_color="off",
        help="Percentage of closed trades that made money.\n\n"
             "20% win rate means 1 in 5 trades was profitable. This sounds bad, but "
             "a strategy can still be profitable with a low win rate if the winning trades "
             "are much larger than the losing ones (high reward-to-risk ratio).\n\n"
             "With a tight stop loss and high leverage, losses are frequent but capped, "
             "while wins (at 15% TP × 2.5x leverage) are large when they hit."
    )
    m4.metric(
        "Max Drawdown",
        f"{metrics['Max Drawdown %']:.2f}%",
        f"{metrics['Max Drawdown %']:+.2f}%",
        help="The largest peak-to-trough decline in account value during the backtest.\n\n"
             "Example: -68% means at some point the account fell from its highest value down by 68% "
             "before recovering. This is the 'worst case' you would have lived through.\n\n"
             "Lower (less negative) is better. Most professional traders consider anything beyond -25% unacceptable."
    )

    st.plotly_chart(
        make_equity_chart(equity_series, 10_000, df, leverage),
        use_container_width=True,
    )

    with st.expander("Regime Summary (all 7 states)"):
        disp = summary_df.copy()
        disp.index = [f"State {i} — {state_labels[i]}" for i in disp.index]
        disp["Mean Return %"] = (disp["Mean_Return"].astype(float) * 100).round(4)
        disp["Volatility %"] = (disp["Volatility"].astype(float) * 100).round(4)
        disp = disp[["Mean Return %", "Volatility %", "Count"]].sort_values(
            "Mean Return %", ascending=False
        )
        st.dataframe(disp, use_container_width=True)

    with st.expander(f"Trade Log ({metrics['Total Trades']} trades)"):
        if trades_df.empty:
            st.info("No trades executed in the backtest window.")
        else:
            st.dataframe(
                trades_df.sort_values("Exit Time", ascending=False),
                use_container_width=True,
            )

# ════════════════════════════════════════════════════════════════════════════
# SCANNER TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.title("Long Signal Scanner")
    st.caption("Run the HMM model across multiple tickers and surface any that are currently in a LONG signal.")

    _scan_col1, _scan_col2, _scan_col3 = st.columns([3, 1, 1])

    _default_scan_tickers = "\n".join(load_quick_tickers())
    _scan_tickers_raw = _scan_col1.text_area(
        "Tickers to scan (one per line or comma-separated)",
        value=_default_scan_tickers,
        height=160,
        help="Enter any valid yfinance ticker symbols. The scan reuses cached HMM results when available.",
    )

    _SCAN_INTERVAL_OPTIONS = {
        "1d / 5y    (Daily, 5 years)":   ("1d", "5y"),
        "1h / 730d  (Hourly, ~2 years)": ("1h", "730d"),
    }
    _scan_interval_label = _scan_col2.selectbox(
        "Interval / Period",
        list(_SCAN_INTERVAL_OPTIONS.keys()),
        key="scan_interval_selectbox",
    )
    _scan_interval, _scan_period = _SCAN_INTERVAL_OPTIONS[_scan_interval_label]
    _scan_votes = _scan_col3.slider("Min Votes", 5, 8, 7, key="scan_votes_slider")

    if st.button("Scan Tickers", type="primary", key="scan_btn"):
        # Parse ticker list
        _raw = _scan_tickers_raw.replace(",", "\n")
        _scan_list = [t.strip().upper() for t in _raw.splitlines() if t.strip()]
        _scan_list = list(dict.fromkeys(_scan_list))  # deduplicate, preserve order

        if not _scan_list:
            st.warning("Enter at least one ticker.")
        else:
            _scan_results = []
            _scan_progress = st.progress(0, text="Starting scan...")
            _scan_errors = []

            for _i, _t in enumerate(_scan_list):
                _scan_progress.progress(
                    int((_i / len(_scan_list)) * 100),
                    text=f"Scanning {_t} ({_i + 1}/{len(_scan_list)}) ...",
                )
                try:
                    _s_df, _s_bull, _s_bears, _s_labels, _ = get_hmm_analysis(
                        _t, _scan_interval, _scan_period
                    )
                    _s_row = _s_df.iloc[-1]
                    _s_state = int(_s_row["State"])
                    _s_regime = _s_labels[_s_state]
                    _s_is_long = _s_state == _s_bull
                    _s_signal = "LONG" if _s_is_long else "CASH"

                    _s_votes_ready = not any(
                        pd.isna(_s_row.get(col)) for col in INDICATOR_COLS
                    )
                    if _s_votes_ready:
                        _s_vote_results = count_votes(_s_row)
                        _s_votes = sum(p for _, p in _s_vote_results)
                    else:
                        _s_votes = None

                    _s_entry = _s_is_long and _s_votes is not None and _s_votes >= _scan_votes

                    _scan_results.append({
                        "Ticker": _t,
                        "Signal": _s_signal,
                        "Entry Ready": _s_entry,
                        "Regime": _s_regime,
                        "Votes": _s_votes,
                        "Price": float(_s_row["Close"]),
                    })
                except Exception as _e:
                    _scan_errors.append((_t, str(_e)))

            _scan_progress.progress(100, text="Scan complete.")
            st.session_state["scan_results"] = _scan_results
            st.session_state["scan_errors"] = _scan_errors

    # Display results
    if "scan_results" in st.session_state and st.session_state["scan_results"]:
        _res = st.session_state["scan_results"]

        # Sort: entry-ready first, then LONG, then by votes desc
        _res_sorted = sorted(
            _res,
            key=lambda r: (not r["Entry Ready"], r["Signal"] != "LONG", -(r["Votes"] or 0)),
        )

        _long_count = sum(1 for r in _res if r["Signal"] == "LONG")
        _entry_count = sum(1 for r in _res if r["Entry Ready"])
        st.subheader(f"Results — {_entry_count} entry-ready, {_long_count} in Bull regime, {len(_res)} scanned")

        for _r in _res_sorted:
            _votes_str = f"{_r['Votes']}/8" if _r["Votes"] is not None else "N/A"

            if _r["Entry Ready"]:
                _sig_md = "<span style='color:#22c55e;font-weight:700'>LONG ✓</span>"
            elif _r["Signal"] == "LONG":
                _sig_md = "<span style='color:#f59e0b;font-weight:700'>LONG</span>"
            else:
                _sig_md = "<span style='color:#888'>CASH</span>"

            _c1, _c2, _c3, _c4, _c5 = st.columns([1.5, 1, 2, 1, 1.5])
            _c1.markdown(f"**{_r['Ticker']}**")
            _c2.markdown(_sig_md, unsafe_allow_html=True)
            _c3.caption(_r["Regime"])
            _c4.caption(f"Votes: {_votes_str}")
            _c5.caption(f"${_r['Price']:,.2f}")
            st.divider()

        if st.session_state.get("scan_errors"):
            with st.expander(f"Errors ({len(st.session_state['scan_errors'])})"):
                for _t, _e in st.session_state["scan_errors"]:
                    st.error(f"{_t}: {_e}")

# ════════════════════════════════════════════════════════════════════════════
# RESEARCH TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_research:
    st.title("Research — Universe Scanner")
    st.caption(
        "By default, Stage 1 pre-filters the universe to the top N tickers by 5-day momentum before running HMM. "
        "Enable 'Scan all tickers' to skip the filter and run HMM on the full universe (slower)."
    )

    rc1, rc2, rc3, rc4 = st.columns([3, 1, 1, 1])

    _research_universes = rc1.multiselect(
        "Universe(s)",
        options=list(UNIVERSE_LABELS.keys()),
        default=["sp500", "ndx100"],
        format_func=lambda k: UNIVERSE_LABELS[k],
        help=(
            "S&P 500 and NASDAQ 100 constituent lists are fetched from Wikipedia (cached 24h). "
            "Crypto and ETF lists are hardcoded."
        ),
    )
    _skip_stage1 = rc1.checkbox(
        "Scan all tickers (skip 5d momentum filter)",
        value=False,
        help="When checked, Stage 1 is skipped and HMM runs on every ticker in the universe. Much slower.",
    )
    _research_top_n = rc2.number_input(
        "Top N (Stage 1 filter)",
        min_value=5, max_value=500, value=30, step=5,
        disabled=_skip_stage1,
        help="After Stage 1 ranks all tickers by 5-day % change, only the top N get full HMM analysis. Disabled when 'Scan all tickers' is checked.",
    )
    _RESEARCH_INTERVAL_OPTIONS = {
        "1d / 5y    (Daily, 5 years)":   ("1d", "5y"),
        "1h / 730d  (Hourly, ~2 years)": ("1h", "730d"),
    }
    _research_interval_label = rc3.selectbox(
        "Interval / Period",
        list(_RESEARCH_INTERVAL_OPTIONS.keys()),
        key="research_interval_selectbox",
    )
    _research_interval, _research_period = _RESEARCH_INTERVAL_OPTIONS[_research_interval_label]
    _research_min_votes = rc4.slider(
        "Min Votes", 5, 8, 7, key="research_votes_slider",
        help="Minimum votes/8 to show as 'Entry Ready' (green checkmark).",
    )

    if st.button(
        "Run Research Scan", type="primary", key="research_scan_btn",
        disabled=len(_research_universes) == 0,
    ):
        st.session_state.pop("research_results", None)
        st.session_state.pop("research_errors", None)
        st.session_state["research_params"] = {
            "universe_keys": tuple(sorted(_research_universes)),
            "top_n":         int(_research_top_n),
            "interval":      _research_interval,
            "period":        _research_period,
            "min_votes":     _research_min_votes,
            "skip_stage1":   _skip_stage1,
        }

    if "research_params" in st.session_state:
        _rp = st.session_state["research_params"]
        _universe_size = len(get_universe_tickers(list(_rp["universe_keys"])))
        _skip = _rp.get("skip_stage1", False)
        if _skip:
            st.caption(
                f"Universe: {_universe_size} tickers   "
                f"Stage 1 filter: OFF (scanning all)   "
                f"Interval: {_rp['interval']} / {_rp['period']}"
            )
            _spinner_msg = f"Running HMM on all {_universe_size} tickers — this may take a while ..."
        else:
            st.caption(
                f"Universe: {_universe_size} tickers   "
                f"Stage 1 filter: top {_rp['top_n']} by 5d momentum   "
                f"Interval: {_rp['interval']} / {_rp['period']}"
            )
            _spinner_msg = (
                f"Stage 1: batch-downloading {_universe_size} tickers ...  "
                f"Stage 2: HMM on top {_rp['top_n']} ..."
            )

        with st.spinner(_spinner_msg):
            _r_results, _r_stage1, _r_errors = get_research_scan(
                universe_keys=_rp["universe_keys"],
                top_n=_rp["top_n"],
                scan_interval=_rp["interval"],
                scan_period=_rp["period"],
                min_votes=_rp["min_votes"],
                skip_stage1=_skip,
            )
        st.session_state["research_results"] = _r_results
        st.session_state["research_stage1"] = _r_stage1
        st.session_state["research_errors"] = _r_errors

    if "research_results" in st.session_state:
        _r_results = st.session_state["research_results"]
        _r_stage1  = st.session_state.get("research_stage1", {})
        _r_errors  = st.session_state.get("research_errors", [])

        _long_count  = sum(1 for r in _r_results if r["is_long"])
        _entry_count = sum(1 for r in _r_results if r["entry_ready"])
        _skip = st.session_state.get("research_params", {}).get("skip_stage1", False)
        _subheader_prefix = f"All {len(_r_results)} tickers" if _skip else f"Top {len(_r_results)} movers"
        st.subheader(f"{_subheader_prefix} — {_long_count} in Bull regime, {_entry_count} entry-ready")

        # Stage 1 debug
        if _r_stage1:
            with st.expander(f"Stage 1: top {len(_r_stage1)} by 5d momentum"):
                for _t, _chg in sorted(_r_stage1.items(), key=lambda x: -x[1]):
                    _sign = "+" if _chg >= 0 else ""
                    _col = "#22c55e" if _chg >= 0 else "#ef4444"
                    st.markdown(
                        f"**{_t}** — <span style='color:{_col}'>{_sign}{_chg:.1f}%</span>",
                        unsafe_allow_html=True,
                    )

        for _r in _r_results:
            _hc1, _hc2, _hc3, _hc4, _hc5, _hc6, _hc7 = st.columns([1.2, 1, 1.5, 0.8, 1.2, 1.5, 1])

            if _r["entry_ready"]:
                _sig_md = "<span style='color:#22c55e;font-weight:700'>LONG ✓</span>"
            elif _r["is_long"]:
                _sig_md = "<span style='color:#f59e0b;font-weight:700'>LONG</span>"
            else:
                _sig_md = "<span style='color:#888'>CASH</span>"

            _votes_str = f"{_r['votes']}/8" if _r["votes"] is not None else "N/A"
            _chg_color = "#22c55e" if _r["change_5d"] >= 0 else "#ef4444"
            _chg_sign  = "+" if _r["change_5d"] >= 0 else ""

            _hc1.markdown(f"### {_r['ticker']}")
            _hc2.markdown(_sig_md, unsafe_allow_html=True)
            _hc3.caption(_r["regime"])
            _hc4.caption(f"Votes: {_votes_str}")
            _hc5.markdown(
                f"<span style='color:{_chg_color}'>{_chg_sign}{_r['change_5d']:.1f}% (5d)</span>"
                f"<br><small style='color:#aaa'>${_r['price']:,.2f}</small>",
                unsafe_allow_html=True,
            )
            if _hc6.button("Open in Dashboard", key=f"research_open_{_r['ticker']}",
                           use_container_width=True):
                st.session_state["active_ticker"] = _r["ticker"]
                st.rerun()
            _hc7.link_button(
                "News", f"https://finance.yahoo.com/quote/{_r['ticker']}/news",
                use_container_width=True,
            )

            st.plotly_chart(
                make_mini_chart(_r["close_series"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.divider()

        if _r_errors:
            with st.expander(f"Stage 2 errors ({len(_r_errors)})"):
                for _t, _e in _r_errors:
                    st.error(f"{_t}: {_e}")

# ════════════════════════════════════════════════════════════════════════════
# OPTIMIZER TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.title(f"Parameter Optimizer — {ticker}")
    st.caption(
        "Tests every combination of the selected values and ranks by Score "
        "(Total Return ÷ Max Drawdown). Higher score = better risk-adjusted return."
    )

    st.subheader("Search Space")
    o1, o2, o3, o4 = st.columns(4)

    sl_opts = o1.multiselect(
        "Stop Loss %", [1, 2, 3, 5, 8, 10, 12, 15],
        default=[2, 3, 5, 8],
        help="Values of stop loss % to test. Each value means: exit if price drops this % from entry. "
             "Tip: BTC averages 2-3% hourly swings, so values below 3% will be hit very frequently."
    )
    tp_opts = o2.multiselect(
        "Take Profit %", [5, 10, 15, 20, 30, 40, 50],
        default=[10, 15, 20, 30],
        help="Values of take profit % to test. Each value means: exit and lock in gains if price rises this % from entry. "
             "Remember leverage multiplies this — a 20% TP at 2x leverage = 40% account gain per winning trade."
    )
    lev_opts = o3.multiselect(
        "Leverage", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        default=[1.0, 1.5, 2.0, 2.5],
        help="Multiplier values to test. 1.0 = no leverage (safest). "
             "Higher values amplify both gains and losses proportionally. "
             "With a low win rate, high leverage tends to destroy accounts over time."
    )
    votes_opts = o4.multiselect(
        "Min Votes", [5, 6, 7, 8],
        default=[5, 6, 7, 8],
        help="How many of the 8 indicators must agree to trigger an entry. "
             "Lower values = more trades but noisier. Higher = fewer but more selective entries. "
             "All 8 values will be tested so you can see the trade-off."
    )

    n_combos = len(sl_opts) * len(tp_opts) * len(lev_opts) * len(votes_opts)
    st.caption(f"{n_combos} combinations selected")

    if st.button("Run Optimizer", type="primary", disabled=n_combos == 0):
        if n_combos > 500:
            st.warning(f"{n_combos} combinations is a lot — consider narrowing the ranges.")

        progress_bar = st.progress(0, text="Running backtest combinations...")
        status_text = st.empty()

        def update_progress(idx, total):
            pct = int((idx / total) * 100)
            progress_bar.progress(pct, text=f"Running combination {idx + 1} / {total} ...")

        results_df = grid_search(
            df, bull_state, bear_crash_states,
            sl_values=sl_opts,
            tp_values=tp_opts,
            leverage_values=lev_opts,
            votes_values=votes_opts,
            cooldown_hours=cooldown,
            progress_cb=update_progress,
        )

        progress_bar.progress(100, text="Done!")
        st.session_state["optimizer_results"] = results_df

    # Show results if available
    if "optimizer_results" in st.session_state:
        results_df = st.session_state["optimizer_results"]
        best = results_df.iloc[0]

        st.subheader("Results (sorted by Score)")
        st.caption(
            "Score = Total Return % / |Max Drawdown %|. "
            "Filters out high-return but catastrophic-drawdown combos."
        )
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Best Parameters Found")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Stop Loss", f"{best['SL %']:.0f}%",
                  help="The stop loss % that produced the best score in this run.")
        b2.metric("Take Profit", f"{best['TP %']:.0f}%",
                  help="The take profit % that produced the best score in this run.")
        b3.metric("Leverage", f"{best['Leverage']}x",
                  help="The leverage multiplier that produced the best score in this run.")
        b4.metric("Min Votes", f"{best['Min Votes']:.0f}/8",
                  help="The vote threshold that produced the best score in this run.")
        b5.metric("Score", f"{best['Score']:.3f}",
                  help="Score = Total Return % ÷ |Max Drawdown %|. "
                       "A score of 1.0 means you gained as much as your worst drawdown. "
                       "Above 1.0 = gains outpaced the pain. Below 0 = overall loss.")

        b_r1, b_r2, b_r3, b_r4 = st.columns(4)
        b_r1.metric("Total Return", f"{best['Total Return %']:+.2f}%")
        b_r2.metric("Alpha", f"{best['Alpha %']:+.2f}%")
        b_r3.metric("Win Rate", f"{best['Win Rate %']:.1f}%")
        b_r4.metric("Max Drawdown", f"{best['Max Drawdown %']:.2f}%")

        if st.button("Apply Best Parameters to Dashboard", type="primary"):
            st.session_state["apply_sl"] = int(best["SL %"])
            st.session_state["apply_tp"] = int(best["TP %"])
            st.session_state["apply_lev"] = float(best["Leverage"])
            st.session_state["apply_votes"] = int(best["Min Votes"])
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# HOW TO USE TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.title("How to Use This Tool")

    st.warning(
        "This is a research and signal tool, not a trading bot. "
        "It does not place trades, manage positions, or guarantee any outcome. "
        "Always apply your own judgment and never risk money you cannot afford to lose."
    )

    st.markdown("""
### Honest caveats

- **The backtest has look-ahead bias.** The HMM was trained on all available historical data, then tested on
  the same data. In real life the model only knows what happened before today, which makes backtest
  results optimistic.
- **Poor backtest results mean no edge yet.** A negative total return or alpha means the current
  parameters did not outperform simply holding. Run the optimizer before trusting any signal.
- **Data is 1-2 hours delayed.** yfinance does not provide real-time prices. This tool is for
  swing trades held days to weeks, not intraday decisions.
- **No tool predicts the future.** The HMM detects patterns in historical behavior. Markets can
  enter regimes that have never existed before.
""")

    st.divider()
    st.header("Workflow")

    with st.expander("Step 1 — Find candidates (Research tab)", expanded=True):
        st.markdown("""
Go to the **Research** tab to discover which tickers are worth looking at.

- Select one or more universes (S&P 500, NASDAQ 100, Crypto, ETFs).
- Set **Top N** — how many of the strongest 5-day movers to run full analysis on (30 is a good start).
- Hit **Run Research Scan**. Stage 1 ranks all tickers by 5-day momentum. Stage 2 runs the full
  HMM on only the top N.

Results are sorted so **entry-ready tickers appear first** (green LONG ✓ = Bull regime + enough votes).
Click **Open in Dashboard** on any ticker to jump straight to its full analysis.

Use this when you want to find new opportunities, not just monitor ones you already know about.
""")

    with st.expander("Step 2 — Set up parameters for a ticker (Optimizer + Presets)", expanded=True):
        st.markdown("""
Once you have a candidate ticker open in the Dashboard, go to the **Parameter Optimizer** tab
and run a grid search over a wide range of values.

Look for the combination with:
- A **positive Score** (return outpaced drawdown)
- **Positive alpha** (beat buy & hold — otherwise just hold the asset)
- **At least 10+ trades** (do not trust a result with only 2-3 lucky trades)

Click **Apply Best Parameters** to load the results into the sidebar sliders.

Then click **Save settings for [ticker]** in the sidebar. Next time you open this ticker, a
"Load preset" button appears that restores all your optimized settings instantly.

Repeat this step monthly or after a major market shift in that asset.
""")

    with st.expander("Step 3 — Morning check (~5 minutes)", expanded=True):
        st.markdown("""
**Option A — Check your watchlist all at once (Scanner tab)**

Go to the **Scanner** tab. Your quick-select tickers are pre-loaded. Hit **Scan Tickers**.
Any ticker showing **LONG ✓** is in a Bull regime with enough votes — go straight to the Dashboard
to verify before acting. Everything else you can ignore for the day.

**Option B — Check a specific ticker (Dashboard tab)**

Open the dashboard, hit **Refresh Data** in the sidebar, and check three things in order:

**1. Regime (top of page)**
- Bear or Crash → stop here. Stay flat.
- Sideways → no new entries. Watch for a potential regime shift.
- Bull Run → continue.

**2. Voting Conditions panel**
- 7+ passing and regime is Bull Run → entry conditions met.
- 5-6 passing → bullish but not fully aligned. Watch, do not act yet.
- Under 5 → weak signal even in a bull regime. Stay out.

**3. Chart — quick sanity check**
- Is price below both the EMA 50 (orange) and EMA 200 (purple)? Be skeptical of any buy signal.
- Are there large red background zones nearby? That indicates recent Bear/Crash regime activity.
""")

    with st.expander("Step 4 — If signal is LONG"):
        st.markdown("""
Use the signal as *input to your judgment*, not a command.

- **Check external context.** Use the **Yahoo News** button on the dashboard to quickly scan for
  upcoming earnings, Fed announcements, or major news. The model knows nothing about those.
- **Decide your position size.** A common rule: never risk more than 1-2% of your total account
  on a single trade, regardless of what the leverage slider shows. The leverage in the backtest
  is simulated — in real trading you choose whether to use it at all.
- **Place your actual orders.** Set a stop loss and take profit as real orders with your broker at
  the levels the optimizer found. Do not rely on memory or checking back manually.
""")

    with st.expander("Step 5 — While in a trade (check once or twice a day)"):
        st.markdown("""
Refresh the dashboard and check:

- **Has the regime flipped to Bear or Crash?** → Exit the trade. This is the primary exit signal.
  Do not wait for your stop loss to be hit if the regime already turned negative.
- **Are you near your stop loss or take profit?** The dashboard shows what *would have* happened
  in the backtest, but it does not manage live positions. Your broker orders handle the actual exit.
""")

    with st.expander("Step 6 — After a trade closes"):
        st.markdown("""
- Note the exit reason in your own trade journal (stop loss, take profit, or regime flip).
- If you are getting stopped out repeatedly → run the optimizer again with wider stop loss values.
  Your SL is likely too tight for the asset's normal volatility.
- Respect the cooldown period (set in the sidebar) before looking for the next entry. Its purpose
  is to prevent immediately re-entering after a loss on emotion.
""")

    st.divider()
    st.header("What each indicator means")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
**HMM Regime**
The Hidden Markov Model groups all historical price/volume behavior into 7 distinct market states.
It automatically identifies which state has the best historical returns (Bull Run) and worst
(Bear, Crash). The regime for each hour is color-coded on the chart background.

**RSI — Relative Strength Index**
Measures how fast price has been moving. 0-100 scale.
Above 70 = overbought (may pull back). Below 30 = oversold (may bounce).
The strategy uses RSI < 90 — a loose filter that only blocks extreme blowoff tops.

**Momentum**
Compares today's price to the price 14 hours ago.
Positive = price is higher than it was 14 hours ago. The strategy requires > 1%.

**Volatility**
Rolling standard deviation of hourly returns over the last 20 candles.
High volatility = big swings happening. The strategy avoids entering when volatility > 6%
because entries during chaotic conditions are unreliable.
""")

    with col_b:
        st.markdown("""
**ADX — Average Directional Index**
Measures trend strength regardless of direction. 0-100 scale.
Below 20 = no trend (choppy). Above 25 = a real trend is forming.
The strategy requires ADX > 25 to confirm there is directional momentum.

**EMA 50 / EMA 200**
Exponential Moving Averages — smoothed versions of recent price.
Price above EMA 50 = short-term uptrend. Price above EMA 200 = long-term uptrend.
The strategy requires both to confirm the uptrend is intact at multiple timeframes.

**MACD — Moving Average Convergence Divergence**
Compares two EMAs (12-period vs 26-period) to detect momentum shifts.
MACD line above Signal line = bullish momentum is building.

**Volume vs SMA 20**
Compares current volume to the 20-period average.
Above average volume on an up move = real buying pressure behind the move.
""")

    st.divider()
    st.header("Quick reference")

    st.markdown("""
| What you see | What it means | What to do |
|---|---|---|
| Regime = Bull Run, 7+ votes | All conditions aligned | Consider entering |
| Regime = Bull Run, < 7 votes | Bullish but not confirmed | Watch, wait for more votes |
| Regime = Sideways | No clear direction | Stay flat, no new entries |
| Regime = Bear or Crash | Actively negative conditions | Exit any open position |
| Price below EMA 200 | Long-term downtrend intact | Be very cautious with longs |
| Win rate < 30% in backtest | Strategy losing more than winning | Re-run optimizer before using |
| Negative alpha | Buy & hold beats the strategy | Consider just holding instead |
""")

    st.info(
        "Tip: before using real money, paper trade for 2-4 weeks. "
        "Each morning write down what the signal is and what you would have done. "
        "After a month, check whether following the signals would have been profitable. "
        "That is the only real test of whether this works for the current market conditions."
    )
