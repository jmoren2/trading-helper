import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

# Features fed to the HMM.
# Price-derived + volume features give the base signal.
# RSI, ADX, and Volatility add momentum, trend strength, and risk context
# so the model can distinguish e.g. a high-return calm trending regime from
# a high-return volatile spike regime.
HMM_FEATURES = ["Returns", "Range", "Vol_Change", "RSI", "ADX", "Volatility"]

# Walk-forward settings. The first model is fit on the oldest slice of history;
# every bar after that is labelled by a model that has only seen earlier bars.
TRAIN_FRACTION = 0.25    # share of history reserved to fit the first model
MIN_TRAIN_BARS = 300     # floor on that window, regardless of fraction
MIN_CAUSAL_BARS = 100    # refuse to return a backtest shorter than this
REFITS = 40              # expanding-window refits spread over the causal span
FILTER_WARMUP = 250      # bars replayed before a segment to settle the belief
WARM_REFIT_ITERS = 25    # a refit nudges an existing fit, it does not redo it

# Model shape. Walk-forward fitting only ever sees a slice of history, so the
# emission model has to be cheap enough to identify from that slice. Chosen on
# out-of-sample predictive likelihood, state persistence, and stability under
# harmless perturbation - not on backtest returns. Beyond 4 states the weakest
# state's self-transition collapses and it stops being a regime at all.
N_COMPONENTS = 4
COVARIANCE_TYPE = "diag"

# States are emitted as a return rank, 0 = worst to n-1 = best, recomputed at
# every refit. That makes the regime ids stable instead of per-fit arbitrary.
# Only the bottom state is an exit trigger: at 4 states the second-worst still
# has a positive mean return, so treating it as bearish would exit good regimes.
BEAR_CRASH_STATES = {0}


def _state_labels(n_components):
    labels = {0: "Bear", n_components - 1: "Bull Run"}
    for i, state in enumerate(range(1, n_components - 1), start=1):
        labels[state] = f"Sideways {i}"
    return labels


def _prepare(df):
    """Add the two HMM-only features and drop rows the model cannot use."""
    df = df.copy()
    if "Returns" not in df.columns:
        df["Returns"] = df["Close"].pct_change()
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Vol_Change"] = df["Volume"].pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=HMM_FEATURES, inplace=True)
    return df


def _repair_covars(covars, floor=1e-3):
    """Make covariances usable as a warm-start again, in the shape hmmlearn wants.

    A previous fit can leave them fractionally asymmetric, or near-singular where
    a state collapsed; hmmlearn rejects both. Note `covars_` always hands back
    full matrices, but the setter expects each covariance_type's own shape.
    """
    covars = np.asarray(covars, dtype=float)
    covars = (covars + covars.transpose(0, 2, 1)) / 2
    w, v = np.linalg.eigh(covars)
    w = np.maximum(w, floor)
    full = v @ (w[..., None] * v.transpose(0, 2, 1))
    if COVARIANCE_TYPE == "diag":
        return np.maximum(np.diagonal(full, axis1=1, axis2=2), floor)
    return full


def _fit(X, n_components, warm_from=None):
    """Fit on X, optionally warm-starting from a previous model so EM converges
    in a few iterations. Purely a speed optimisation - warm starting does not
    guarantee state identities survive the refit, so nothing may rely on that.
    """
    if warm_from is None:
        return GaussianHMM(
            n_components=n_components,
            covariance_type=COVARIANCE_TYPE,
            n_iter=1000,
            random_state=42,
        ).fit(X)

    # hmmlearn's tol is an absolute log-likelihood delta, so it gets effectively
    # stricter as the expanding window grows. Cap the iterations instead.
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=COVARIANCE_TYPE,
        n_iter=WARM_REFIT_ITERS,
        random_state=42,
        init_params="",
        params="stmc",
    )
    model.startprob_ = warm_from.startprob_
    model.transmat_ = warm_from.transmat_
    model.means_ = warm_from.means_
    model.covars_ = _repair_covars(warm_from.covars_)
    try:
        return model.fit(X)
    except ValueError:
        # A collapsed state can leave the previous fit unusable as a starting
        # point. Falling back to a cold fit keeps the walk-forward going.
        return _fit(X, n_components)


def _rank_of_state(model):
    """Map raw hmmlearn state ids to a return rank (0 = worst, n-1 = best).

    Ranks on the fitted mean of the Returns feature, which is monotone in the
    empirical mean return because the scaler is a fixed affine transform.
    """
    returns_idx = HMM_FEATURES.index("Returns")
    order = np.argsort(model.means_[:, returns_idx])
    ranks = np.empty(model.n_components, dtype=int)
    ranks[order] = np.arange(model.n_components)
    return ranks


def _log_emissions(model, X):
    """Log P(x_t | state) per bar and state, using only public model attributes."""
    out = np.empty((len(X), model.n_components))
    for k in range(model.n_components):
        out[:, k] = multivariate_normal.logpdf(
            X, mean=model.means_[k], cov=model.covars_[k], allow_singular=True
        )
    return out


def _forward(log_e, model):
    """Filtered forward pass: the state at bar t uses bars up to t only.

    This is the whole point of the walk-forward. hmmlearn's predict() runs
    Viterbi over the full sequence, so the state it assigns to bar t depends on
    bars after t - unusable as a trading signal.
    """
    log_A = np.log(np.maximum(model.transmat_, 1e-300))
    A = np.exp(log_A)
    states = np.empty(len(log_e), dtype=int)
    log_alpha = np.log(np.maximum(model.startprob_, 1e-300)) + log_e[0]
    log_alpha -= log_alpha.max()
    states[0] = int(log_alpha.argmax())

    for t in range(1, len(log_e)):
        shift = log_alpha.max()
        log_alpha = np.log(np.exp(log_alpha - shift) @ A) + shift + log_e[t]
        log_alpha -= log_alpha.max()  # normalise; argmax is unaffected
        states[t] = int(log_alpha.argmax())

    return states


def train_hmm(df, n_components=None, train_end=None, step=None):
    """Label each bar with a market regime using walk-forward inference.

    The returned frame starts where the first out-of-sample bar does - every
    earlier bar was spent fitting the initial model and is not tradeable.

    `train_end` and `step` pin the walk-forward schedule in bars; both default to
    a fraction of the series length, which is fine for a study fixed up front but
    means the schedule shifts if the series does.
    """
    df = _prepare(df)
    n = len(df)
    if n_components is None:
        n_components = N_COMPONENTS

    if train_end is None:
        train_end = max(MIN_TRAIN_BARS, int(n * TRAIN_FRACTION))
    if n - train_end < MIN_CAUSAL_BARS:
        raise ValueError(
            f"Not enough history: {n} usable bars, need at least "
            f"{train_end + MIN_CAUSAL_BARS} for a walk-forward backtest."
        )

    # The scaler sees only the initial training window, both to avoid lookahead
    # and to keep feature scale fixed so warm-started models stay comparable.
    raw = df[HMM_FEATURES].values
    scaler = StandardScaler().fit(raw[:train_end])
    X = scaler.transform(raw)

    model = _fit(X[:train_end], n_components)
    ranks = _rank_of_state(model)

    if step is None:
        step = max(1, (n - train_end) // REFITS)
    bounds = list(range(train_end, n, step)) + [n]

    states = np.empty(n - train_end, dtype=int)
    for start, stop in zip(bounds, bounds[1:]):
        if start > train_end:
            model = _fit(X[:start], n_components, warm_from=model)
            ranks = _rank_of_state(model)
        # Restart the filter under the current model instead of carrying a belief
        # over from the previous one - a refit may permute or merge states. The
        # replayed warm-up bars are all earlier than `start`, so still causal.
        warmup = max(0, start - FILTER_WARMUP)
        seg = _forward(_log_emissions(model, X[warmup:stop]), model)
        states[start - train_end:stop - train_end] = ranks[seg[start - warmup:]]

    out = df.iloc[train_end:].copy()
    out["State"] = states

    state_labels = _state_labels(n_components)
    out["Regime"] = out["State"].map(state_labels)

    bull_state = n_components - 1
    bear_crash_states = set(BEAR_CRASH_STATES)

    grouped = out.groupby("State")["Returns"]
    summary_df = pd.DataFrame({
        "Mean_Return": grouped.mean(),
        "Volatility": grouped.std(),
        "Count": grouped.size(),
    }).reindex(range(n_components))
    summary_df["Count"] = summary_df["Count"].fillna(0).astype(int)

    # Persistence: P(same regime next bar), measured on the emitted regime sequence
    # rather than read off the final model's transition matrix. The two disagree -
    # the matrix describes the latent chain under one fit, while what the UI shows
    # is the filtered argmax across ~REFITS models - and only this version matches
    # the run lengths a user can actually see on the chart. Mean run is 1 / (1 - p).
    cur, nxt = states[:-1], states[1:]
    persistence = np.full(n_components, np.nan)
    for state in range(n_components):
        seen = cur == state
        if seen.sum() >= 2:
            persistence[state] = (nxt[seen] == state).mean()
    summary_df["Persistence"] = persistence
    with np.errstate(divide="ignore", invalid="ignore"):
        summary_df["Expected_Bars"] = np.minimum(
            1.0 / np.maximum(1.0 - persistence, 1e-3), 999.0
        )
    summary_df.index.name = "State"
    summary_df.replace([np.nan, np.inf, -np.inf], None, inplace=True)

    return out, bull_state, bear_crash_states, state_labels, summary_df
