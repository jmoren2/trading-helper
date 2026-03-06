import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Features fed to the HMM.
# Price-derived + volume features give the base signal.
# RSI, ADX, and Volatility add momentum, trend strength, and risk context
# so the model can distinguish e.g. a high-return calm trending regime from
# a high-return volatile spike regime.
HMM_FEATURES = ["Returns", "Range", "Vol_Change", "RSI", "ADX", "Volatility"]


def train_hmm(df, n_components=7):
    df = df.copy()

    if "Returns" not in df.columns:
        df["Returns"] = df["Close"].pct_change()
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Vol_Change"] = df["Volume"].pct_change()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=HMM_FEATURES, inplace=True)

    # Scale all features to zero mean / unit variance so no single feature
    # dominates (e.g. RSI ranges 0-100 vs Returns ~0.001-0.05)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[HMM_FEATURES].values)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
    )
    model.fit(X)
    df["State"] = model.predict(X)

    # Regime stats
    summary = {}
    for state in range(n_components):
        mask = df["State"] == state
        returns = df.loc[mask, "Returns"]
        summary[state] = {
            "Mean_Return": returns.mean(),
            "Volatility": returns.std(),
            "Count": int(mask.sum()),
        }

    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "State"
    summary_df.replace([np.nan, np.inf, -np.inf], None, inplace=True)

    sorted_returns = summary_df["Mean_Return"].sort_values()
    crash_state = int(sorted_returns.index[0])
    bear_state = int(sorted_returns.index[1])
    bull_state = int(sorted_returns.index[-1])
    bear_crash_states = {crash_state, bear_state}

    # Label every state
    neutral_n = 0
    state_labels = {}
    for state in range(n_components):
        if state == bull_state:
            state_labels[state] = "Bull Run"
        elif state == crash_state:
            state_labels[state] = "Crash"
        elif state == bear_state:
            state_labels[state] = "Bear"
        else:
            neutral_n += 1
            state_labels[state] = f"Sideways {neutral_n}"

    df["Regime"] = df["State"].map(state_labels)

    return df, bull_state, bear_crash_states, state_labels, summary_df
