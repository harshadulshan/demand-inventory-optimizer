"""
forecasting.py
The forecasting engine: demand-pattern classification, eight forecasting
methods (from the benchmark every method must beat, through the
intermittent-demand specialists, to a global gradient-boosting model), and
an honest rolling-origin backtest.

Design principles
-----------------
* **No lookahead, anywhere.** Every forecast made at origin o (the last
  observed day) uses only data up to and including day o: method states
  are filtered forward in time, ML models are trained only on targets
  <= their refit date, weekday profiles and closure rules are learned from
  the past. `tests/verify_engine.py` checks this directly by scrambling all
  data after an origin and confirming the forecasts don't change.
* **Closed days are not zero-demand days.** Local methods skip days the
  store was closed (see calendar_utils.py) and forecast zero on days it is
  known in advance to be closed.
* **Forecasts are judged on what drives stock.** Besides daily accuracy
  (MASE, RMSSE), each method is scored on its error over the inventory
  protection period (review period + lead time), because that error - not
  daily MAPE - is what determines safety stock. Per-product method
  selection uses that criterion.

Methods
-------
Seasonal naive (weekly) - the benchmark (Hyndman & Athanasopoulos, FPP3).
Moving average (28 days).
SES - simple exponential smoothing, alpha chosen per product by in-sample
    one-step squared error (Brown 1959; Hyndman et al. 2008).
Croston-SBA - Croston (1972) with the Syntetos-Boylan (2005) bias correction,
    for intermittent demand.
TSB - Teunter, Syntetos & Babai (2011): smooths the probability of demand
    every period, so it reacts to products dying out (obsolescence).
Theta - Assimakopoulos & Nikolopoulos (2000), in its SES-with-drift form
    (Hyndman & Billah 2003), applied to weekday-adjusted demand.
Global ML - one scikit-learn HistGradientBoostingRegressor with Poisson loss,
    trained across all products at once on lag, rolling, weekday, holiday and
    last-year features (the approach that dominated the M5 competition;
    Makridakis, Spiliotis & Assimakopoulos 2022).
Ensemble - equal-weight average of Theta, TSB and Global ML (forecast
    combination is a robust winner across the M competitions).

Accuracy measures: MASE (Hyndman & Koehler 2006, weekly-seasonal scaling),
RMSSE (as used in M5), WAPE and bias.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from calendar_utils import StoreCalendar, build_store_calendar

SEASONAL_NAIVE = "Seasonal naive"
MOVING_AVERAGE = "Moving average (28d)"
SES = "SES"
CROSTON_SBA = "Croston-SBA"
TSB = "TSB"
THETA = "Theta"
ML = "Global ML (gradient boosting)"
ENSEMBLE = "Ensemble"
AUTO = "Auto (best per product)"

ALL_METHODS = [SEASONAL_NAIVE, MOVING_AVERAGE, SES, CROSTON_SBA, TSB, THETA, ML, ENSEMBLE]
LOCAL_METHODS = [SEASONAL_NAIVE, MOVING_AVERAGE, SES, CROSTON_SBA, TSB, THETA]

H_MAX = 63          # forecast horizon kept for every origin (covers review + lead time up to 56 days + 6)
ORIGIN_STEP = 7     # forecast origins are weekly (a weekly planning cycle)

ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6])
DEFAULT_ALPHA_IDX = 4         # 0.2 - used for products with too little history to tune alpha
SES_WARMUP, SES_BURN = 7, 14  # running-mean warm-up, then SSE accumulation starts after BURN obs
CROSTON_ALPHA = 0.1
TSB_ALPHA, TSB_BETA = 0.1, 0.05
THETA_WINDOW = 364            # trailing window for weekday profiles
THETA_LAMBDA = 5.0            # pseudo-units shrinking weekday indices toward 1 for slow sellers

# Syntetos-Boylan-Croston (2005) cut-offs for demand classification
ADI_CUTOFF, CV2_CUTOFF = 1.32, 0.49


# ---------------------------------------------------------------------------
# Demand classification
# ---------------------------------------------------------------------------

def life_start(Y: np.ndarray) -> np.ndarray:
    """Index of each product's first sale (len(T) if it never sold)."""
    has = Y > 0
    first = np.argmax(has, axis=1)
    first[~has.any(axis=1)] = Y.shape[1]
    return first


def classify_demand(Y: np.ndarray, valid: np.ndarray, value: np.ndarray, end: Optional[int] = None,
                    window: int = 364, start: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Per-product demand profile over the last `window` days before `end`.

    * ADI  - average inter-demand interval (trading days per day with a sale)
    * CV^2 - squared coefficient of variation of the non-zero demand sizes
    * pattern - Syntetos-Boylan-Croston quadrant: smooth / erratic / intermittent / lumpy
    * ABC  - by sales value: A = the items making up the first 80%, B = next 15%, C = last 5%
    * XYZ  - by coefficient of variation of *weekly* totals: X < 0.5, Y < 1.0, Z >= 1.0
    """
    n, T = Y.shape
    end = T if end is None else end
    lo = max(0, end - window)
    seg = Y[:, lo:end]
    start = life_start(Y) if start is None else start
    days = np.arange(lo, end)
    rows = []
    for i in range(n):
        y = seg[i][valid[lo:end] & (days >= start[i])]
        nz = y[y > 0]
        n_open = len(y)
        if len(nz) == 0:
            adi, cv2, pattern = np.inf, np.nan, "no recent demand"
        else:
            adi = n_open / len(nz)
            cv2 = float((nz.std(ddof=1) / nz.mean()) ** 2) if len(nz) > 1 else 0.0
            if adi < ADI_CUTOFF:
                pattern = "smooth" if cv2 < CV2_CUTOFF else "erratic"
            else:
                pattern = "intermittent" if cv2 < CV2_CUTOFF else "lumpy"
        weeks = seg[i][: (seg.shape[1] // 7) * 7].reshape(-1, 7).sum(axis=1)
        cvw = float(weeks.std(ddof=1) / weeks.mean()) if weeks.mean() > 0 and len(weeks) > 1 else np.inf
        rows.append(dict(adi=adi, cv2=cv2, pattern=pattern, cv_weekly=cvw, units=float(seg[i].sum()),
                         mean_daily=float(seg[i].mean()), zero_share=float((y == 0).mean()) if n_open else 1.0))
    df = pd.DataFrame(rows)
    df["value"] = df["units"] * np.where(np.isfinite(value), value, 1.0)
    order = np.argsort(-df["value"].to_numpy())
    total = df["value"].sum()
    cum_before = np.zeros(n)
    if total > 0:
        cum_before[order] = np.concatenate([[0.0], np.cumsum(df["value"].to_numpy()[order])[:-1]]) / total
    df["abc"] = np.where(cum_before < 0.80, "A", np.where(cum_before < 0.95, "B", "C"))
    df.loc[df["value"] <= 0, "abc"] = "C"
    df["xyz"] = np.where(df["cv_weekly"] < 0.5, "X", np.where(df["cv_weekly"] < 1.0, "Y", "Z"))
    return df


# ---------------------------------------------------------------------------
# Local methods: filtered states at every day (one forward pass serves all origins)
# ---------------------------------------------------------------------------

def _ses_states(X: np.ndarray, upd: np.ndarray, warmup: int = SES_WARMUP, burn: int = SES_BURN):
    """Simple exponential smoothing for every alpha in ALPHAS, every product, every day.

    X:   (n, T) series;  upd: (n, T) bool - observation usable on that day
         (store open and product already launched).
    The first `warmup` usable observations initialise the level as their
    running mean; after that the standard recursion
        level <- level + alpha * (x - level)
    applies. One-step squared errors are summed from observation `burn` on,
    and used to pick alpha at any origin. Returns levels (T, n, A),
    cumulative SSE (T, n, A) and usable-observation counts (T, n).
    """
    n, T = X.shape
    A = len(ALPHAS)
    level = np.zeros((n, A))
    sse = np.zeros((n, A))
    count = np.zeros(n)
    L = np.empty((T, n, A))
    S = np.empty((T, n, A))
    C = np.empty((T, n))
    a = ALPHAS[None, :]
    for t in range(T):
        x = X[:, t]
        v = upd[:, t]
        warm = v & (count < warmup)
        act = v & (count >= warmup)
        err = x[:, None] - level
        sse += np.where((act & (count >= burn))[:, None], err * err, 0.0)
        level = np.where(act[:, None], level + a * err, level)
        level = np.where(warm[:, None], level + (x[:, None] - level) / (count[:, None] + 1.0), level)
        count = count + v
        L[t], S[t], C[t] = level, sse, count
    return L, S, C


def _pick_ses(L, S, C, o_idx: np.ndarray):
    """Level and alpha at each origin, alpha chosen by lowest cumulative one-step SSE."""
    Lo, So, Co = L[o_idx], S[o_idx], C[o_idx]          # (n_o, n, A), (n_o, n)
    a_idx = np.argmin(So, axis=2)
    young = Co < SES_BURN + 14
    a_idx = np.where(young, DEFAULT_ALPHA_IDX, a_idx)
    lvl = np.take_along_axis(Lo, a_idx[..., None], axis=2)[..., 0]
    return lvl, ALPHAS[a_idx], Co


def _croston_states(Y: np.ndarray, upd: np.ndarray, alpha: float = CROSTON_ALPHA):
    """Croston's method: smooth demand sizes (z) and inter-demand intervals (p)
    separately, updating only when a sale occurs. Intervals are counted in
    trading days. Initialisation: running means of the first 3 sizes and the
    first 3 observed intervals, then exponential smoothing. Returns z, p (T, n)."""
    n, T = Y.shape
    z = np.zeros(n)
    p = np.ones(n)
    q = np.zeros(n)
    n_dem = np.zeros(n)
    n_int = np.zeros(n)
    Z = np.empty((T, n))
    P = np.empty((T, n))
    for t in range(T):
        v = upd[:, t]
        y = Y[:, t]
        q = q + v
        d = v & (y > 0)
        n_dem = n_dem + d
        z = np.where(d & (n_dem <= 3), z + (y - z) / np.maximum(n_dem, 1), np.where(d, z + alpha * (y - z), z))
        has_int = d & (n_dem >= 2)
        n_int = n_int + has_int
        p = np.where(has_int & (n_int <= 3), p + (q - p) / np.maximum(n_int, 1),
                     np.where(has_int, p + alpha * (q - p), p))
        q = np.where(d, 0.0, q)
        Z[t], P[t] = z, p
    return Z, P


def _tsb_states(Y: np.ndarray, upd: np.ndarray, alpha: float = TSB_ALPHA, beta: float = TSB_BETA):
    """TSB: demand probability smoothed every trading day (beta), demand size
    smoothed on sale days (alpha). Forecast = probability x size.
    Initialisation: running means over the first 14 trading days / first 3 sales."""
    n, T = Y.shape
    prob = np.zeros(n)
    z = np.zeros(n)
    n_open = np.zeros(n)
    n_dem = np.zeros(n)
    PR = np.empty((T, n))
    Z = np.empty((T, n))
    for t in range(T):
        v = upd[:, t]
        y = Y[:, t]
        d = (v & (y > 0)).astype(float)
        n_open = n_open + v
        prob = np.where(v & (n_open <= 14), prob + (d - prob) / np.maximum(n_open, 1),
                        np.where(v, prob + beta * (d - prob), prob))
        dm = d > 0
        n_dem = n_dem + dm
        z = np.where(dm & (n_dem <= 3), z + (y - z) / np.maximum(n_dem, 1), np.where(dm, z + alpha * (y - z), z))
        PR[t], Z[t] = prob, z
    return PR, Z


def _cum(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Cumulative sum with a leading zero along `axis` (so window sums are differences)."""
    shape = list(a.shape)
    shape[axis] = 1
    return np.concatenate([np.zeros(shape), np.cumsum(a, axis=axis)], axis=axis)


@dataclass
class LocalStates:
    """Everything the local methods need, computed in one forward pass."""
    upd: np.ndarray                 # (n, T) usable observations
    start: np.ndarray               # (n,) life start
    ses: tuple                      # (L, S, C)
    croston: tuple                  # (Z, P)
    tsb: tuple                      # (PR, Z)
    ma_sum: np.ndarray              # (n, T+1) cumsum of usable y
    ma_cnt: np.ndarray              # (n, T+1) cumsum of usable flags
    theta_idx: np.ndarray           # (7, n, T) weekday index at each day (data <= day)
    theta_ses: tuple                # SES states on weekday-adjusted series
    theta_reg: tuple                # cumulative regression sums (n_v, St, Sx, Stt, Stx), each (T, n)
    last_same_dow: np.ndarray       # (T,) most recent open day <= t with t's weekday (-1 if none)


def compute_local_states(Y: np.ndarray, valid: np.ndarray, dow: np.ndarray) -> LocalStates:
    """Run every local method forward through the whole history once.

    valid: (T,) bool, store traded that day (closed days are skipped, not
    treated as zero demand). dow: (T,) weekday of each observed day.
    """
    n, T = Y.shape
    start = life_start(Y)
    tt = np.arange(T)
    alive = tt[None, :] >= start[:, None]
    upd = alive & valid[None, :]

    ses = _ses_states(Y, upd)
    croston = _croston_states(Y, upd)
    tsb = _tsb_states(Y, upd)
    yu = np.where(upd, Y, 0.0)
    ma_sum, ma_cnt = _cum(yu, 1), _cum(upd.astype(float), 1)

    # Weekday indices from a trailing window of usable days, shrunk toward 1:
    #   idx_w = (observed_w + lambda) / (expected_w + lambda),   expected_w = n_w * overall mean
    # then renormalised so the trading-day-weighted mean index is 1.
    Sw = np.zeros((7, n, T))
    Nw = np.zeros((7, n, T))
    for w in range(7):
        m = (dow == w)[None, :] & upd
        cs = _cum(np.where(m, Y, 0.0), 1)
        cn = _cum(m.astype(float), 1)
        lo = np.maximum(tt + 1 - THETA_WINDOW, 0)
        Sw[w] = cs[:, tt + 1] - cs[:, lo]
        Nw[w] = cn[:, tt + 1] - cn[:, lo]
    S_all = Sw.sum(axis=0)
    N_all = Nw.sum(axis=0)
    mean_all = np.divide(S_all, N_all, out=np.zeros_like(S_all), where=N_all > 0)
    raw_idx = (Sw + THETA_LAMBDA) / (Nw * mean_all[None] + THETA_LAMBDA)
    norm = np.divide((Nw * raw_idx).sum(axis=0), N_all, out=np.ones_like(S_all), where=N_all > 0)
    theta_idx = raw_idx / np.where(norm > 0, norm, 1.0)[None]

    # Weekday-adjust each observation with the index known *before* that day.
    idx_prev = np.ones((n, T))
    for w in range(7):
        cols = np.where(dow == w)[0]
        cols_prev = cols[cols > 0]
        idx_prev[:, cols_prev] = theta_idx[w][:, cols_prev - 1]
    x_upd = upd & (idx_prev > 0.05)
    X = np.where(x_upd, Y / np.maximum(idx_prev, 1e-9), 0.0)
    theta_ses = _ses_states(X, x_upd)
    tf = tt[None, :].astype(float)
    xu = x_upd.astype(float)
    reg = tuple(np.cumsum(a, axis=1).T for a in (xu, xu * tf, X * xu, xu * tf * tf, X * xu * tf))

    last_same = np.full(T, -1)
    for w in range(7):
        last = -1
        for t in np.where(dow == w)[0]:
            if valid[t]:
                last = t
            last_same[t] = last
    return LocalStates(upd=upd, start=start, ses=ses, croston=croston, tsb=tsb, ma_sum=ma_sum, ma_cnt=ma_cnt,
                       theta_idx=theta_idx, theta_ses=theta_ses, theta_reg=reg, last_same_dow=last_same)


def local_forecasts(Y: np.ndarray, st: LocalStates, o_idx: np.ndarray, H: int, open_full: np.ndarray,
                    dow_full: np.ndarray, methods: list[str]) -> dict[str, np.ndarray]:
    """Forecasts (n, len(o_idx), H) for each requested local method at the given origins.

    open_full / dow_full cover observed days plus the horizon (T + H), and
    open_full is the *predicted* open factor (closures known in advance).
    """
    n, T = Y.shape
    o_idx = np.asarray(o_idx)
    n_o = len(o_idx)
    hh = np.arange(1, H + 1)
    tgt = o_idx[:, None] + hh[None, :]                     # (n_o, H)
    of = open_full[tgt][None, :, :]                        # (1, n_o, H)
    alive = (o_idx[None, :] >= st.start[:, None])[:, :, None]
    out: dict[str, np.ndarray] = {}

    # moving average over trading days in the last 28 calendar days
    lo = np.maximum(o_idx + 1 - 28, 0)
    ssum = st.ma_sum[:, o_idx + 1] - st.ma_sum[:, lo]
    scnt = st.ma_cnt[:, o_idx + 1] - st.ma_cnt[:, lo]
    ma = np.divide(ssum, scnt, out=np.zeros_like(ssum), where=scnt > 0)   # (n, n_o)
    if MOVING_AVERAGE in methods or SEASONAL_NAIVE in methods:
        out[MOVING_AVERAGE] = ma[:, :, None] * of * alive

    if SEASONAL_NAIVE in methods:
        ref0 = tgt - 7 * np.ceil(hh / 7).astype(int)[None, :]          # same weekday, <= origin
        ref = st.last_same_dow[np.clip(ref0, 0, T - 1)]
        ok = (ref0 >= 0) & (ref >= 0)
        vals = Y[:, np.clip(ref, 0, T - 1)]                             # (n, n_o, H)
        ok_life = ok[None] & (ref[None] >= st.start[:, None, None])
        sn = np.where(ok_life, vals, ma[:, :, None])
        out[SEASONAL_NAIVE] = sn * of * alive

    if SES in methods:
        lvl, _, _ = _pick_ses(*st.ses, o_idx)                           # (n_o, n)
        out[SES] = lvl.T[:, :, None] * of * alive

    if CROSTON_SBA in methods:
        Z, P = st.croston
        f = np.divide(Z[o_idx], P[o_idx], out=np.zeros_like(Z[o_idx]), where=P[o_idx] > 0) * (1 - CROSTON_ALPHA / 2)
        out[CROSTON_SBA] = f.T[:, :, None] * of * alive

    if TSB in methods:
        PR, Z = st.tsb
        out[TSB] = (PR[o_idx] * Z[o_idx]).T[:, :, None] * of * alive

    if THETA in methods:
        lvl, alpha, cnt = _pick_ses(*st.theta_ses, o_idx)               # (n_o, n)
        nv, St, Sx, Stt, Stx = (a[o_idx] for a in st.theta_reg)         # (n_o, n)
        den = nv * Stt - St * St
        b = np.divide(nv * Stx - St * Sx, den, out=np.zeros_like(den), where=(den > 0) & (nv >= 28))
        # SES-with-drift forecast (Hyndman & Billah 2003): l_n + b/2 * [(h-1) + 1/a - (1-a)^n / a]
        drift = (b / 2)[:, :, None] * ((hh[None, None, :] - 1) + 1 / alpha[:, :, None]
                                         - (1 - alpha[:, :, None]) ** cnt[:, :, None] / alpha[:, :, None])
        xhat = lvl[:, :, None] + drift                                  # (n_o, n, H)
        tgt_dow = dow_full[tgt]                                         # (n_o, H)
        idx_o = st.theta_idx[:, :, o_idx]                               # (7, n, n_o)
        season = idx_o[tgt_dow, :, np.arange(n_o)[:, None]]             # (n_o, H, n)
        f = np.clip(xhat * np.transpose(season, (0, 2, 1)), 0, None)    # (n_o, n, H)
        out[THETA] = np.transpose(f, (1, 0, 2)) * of * alive

    if MOVING_AVERAGE not in methods:
        out.pop(MOVING_AVERAGE, None)
    return out


# ---------------------------------------------------------------------------
# Global ML model (gradient boosting across all products)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "horizon", "weekday", "day_of_month", "month", "predicted_open", "is_holiday", "days_to_holiday",
    "days_since_holiday", "log_level", "last_day_rel", "mean_7d_rel", "mean_28d_rel", "mean_91d_rel",
    "mean_364d_rel", "std_28d_rel", "zero_share_28d", "days_since_sale", "product_age",
    "same_weekday_last_rel", "same_weekday_mean_4w_rel", "weekday_index", "last_year_target_week_rel",
    "last_year_origin_28d_rel", "last_year_ratio", "lifetime_mean_rel",
]
CATEGORICAL = ["weekday", "month"]
RATIO_CAP = 60.0   # winsorise demand/level ratios (one-off bulk orders) when training


class FeatureBuilder:
    """Builds ML features for arbitrary (product, origin, horizon) triples.

    Every feature for origin o uses observations <= o only; calendar
    features describe the *target* day and are known in advance.

    The model is scale-free: it predicts demand *relative to each product's
    recent level* ("base", a blend of its 28- and 91-day averages), and its
    level features are ratios to that base. Tree models cannot extrapolate
    beyond the values seen in training, so a model fit on raw unit counts
    under-forecasts as soon as a product's level rises above its history
    (e.g. a gift wholesaler's autumn surge). Learning multiplicative patterns
    - weekday shape, holiday lift, momentum, last year's seasonal ramp - and
    multiplying back by the current level removes that failure mode. With
    Poisson loss this is fitted exactly as a Poisson GLM with an exposure
    offset: target y/base, sample weight base.
    """

    def __init__(self, Y: np.ndarray, cal: StoreCalendar, st: LocalStates):
        self.Y = Y
        n, T = Y.shape
        self.n, self.T = n, T
        self.start = st.start
        self.theta_idx = st.theta_idx
        tt = np.arange(T)
        cs = _cum(Y, 1)
        cs2 = _cum(Y * Y, 1)
        csz = _cum((Y == 0).astype(float), 1)
        self.cs = cs
        alive = tt[None, :] >= self.start[:, None]

        def wmean(c, k):
            """Mean over the last k days, counting only days since the product's launch."""
            lo = np.maximum(tt[None, :] + 1 - k, np.minimum(self.start, T)[:, None])
            lo = np.minimum(lo, tt[None, :])
            num = c[:, tt + 1] - np.take_along_axis(c, lo, axis=1)
            return np.where(alive, num / (tt[None, :] + 1 - lo), np.nan)

        self.m7, self.m28, self.m91 = wmean(cs, 7), wmean(cs, 28), wmean(cs, 91)
        self.m364 = np.where(tt[None, :] - self.start[:, None] >= 181, wmean(cs, 364), np.nan)
        self.sd28 = np.sqrt(np.clip(wmean(cs2, 28) - self.m28 ** 2, 0, None))
        self.z28 = wmean(csz, 28)
        self.life_mean = wmean(cs, 10 ** 6)
        # Level the target is expressed relative to. The floor (a share of the
        # product's lifetime average) keeps dormant products - a Christmas line
        # in August - from getting a near-zero base and an absurd ratio target.
        recent = np.nan_to_num(0.5 * self.m28 + 0.5 * self.m91, nan=0.0)
        self.base = np.maximum(recent, 0.3 * np.nan_to_num(self.life_mean, nan=0.0)) + 0.05
        self.run_max = np.maximum.accumulate(Y, axis=1)        # highest day so far (prediction guard)
        last_sale = np.maximum.accumulate(np.where(Y > 0, tt[None, :], -1), axis=1)
        self.dsl = np.where(last_sale >= 0, tt[None, :] - last_sale, np.nan)
        self.age = np.where(alive, tt[None, :] - self.start[:, None], np.nan)

        # mean of the last 4 same-weekday values (<= t), per weekday
        dow_obs = cal.dates[:T].dayofweek.to_numpy()
        self.dowm4 = np.full((7, n, T), np.nan)
        for w in range(7):
            cols = np.where(dow_obs == w)[0]
            if len(cols) == 0:
                continue
            c = _cum(Y[:, cols], 1)
            k = np.arange(1, len(cols) + 1)
            lo = np.maximum(k - 4, 0)
            roll = (c[:, k] - c[:, lo]) / (k - lo)[None, :]
            pos = np.searchsorted(cols, tt, side="right") - 1       # last weekday-w column <= t
            okp = pos >= 0
            self.dowm4[w][:, okp] = roll[:, pos[okp]]

        full = cal.dates
        self.dow_f = full.dayofweek.to_numpy()
        self.dom_f = full.day.to_numpy()
        self.mon_f = full.month.to_numpy()
        self.open_f = cal.open_factor()
        self.hol_f = cal.is_holiday.astype(float)
        self.toh_f = cal.days_to_holiday
        self.sinh_f = cal.days_since_holiday

    def _lymean(self, i, center, half):
        lo, hi = center - half, center + half
        ok = lo >= 0
        lo_c, hi_c = np.clip(lo, 0, self.T - 1), np.clip(hi, 0, self.T - 1)
        s = self.cs[i, hi_c + 1] - self.cs[i, lo_c]
        return np.where(ok, s / (hi_c - lo_c + 1), np.nan)

    def build(self, i: np.ndarray, o: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feature matrix (N, n_features) and the level `base` (N,) the target is relative to."""
        t = o + h
        dow_t = self.dow_f[t]
        Y = self.Y
        base = self.base[i, o]
        sref = t - 7 * np.ceil(h / 7).astype(int)
        sdl = np.where(sref >= 0, Y[i, np.clip(sref, 0, self.T - 1)], np.nan)
        ly_t = self._lymean(i, t - 364, 3)
        ly_o = self._lymean(i, o - 364 - 14, 14)
        cols = [
            h, dow_t, self.dom_f[t], self.mon_f[t], self.open_f[t], self.hol_f[t], self.toh_f[t], self.sinh_f[t],
            np.log(base), Y[i, o] / base, self.m7[i, o] / base, self.m28[i, o] / base, self.m91[i, o] / base,
            self.m364[i, o] / base, self.sd28[i, o] / base, self.z28[i, o], self.dsl[i, o], self.age[i, o],
            sdl / base, self.dowm4[dow_t, i, o] / base, self.theta_idx[dow_t, i, o], ly_t / base, ly_o / base,
            (ly_t + 0.1) / (ly_o + 0.1), self.life_mean[i, o] / base,
        ]
        return np.column_stack([np.asarray(c, dtype=np.float32) for c in cols]), base


def _sample_training_rows(start: np.ndarray, train_end: int, H: int, max_rows: int, rng, o_min: int = 28):
    """Random (product, origin, horizon) triples whose target day is <= train_end."""
    n = len(start)
    if train_end - o_min < 2:
        return None
    draws = int(max_rows * 1.6) + 1000
    i = rng.integers(0, n, draws)
    o = rng.integers(o_min, train_end, draws)
    h = rng.integers(1, H + 1, draws)
    ok = (o + h <= train_end) & (o >= start[i])
    i, o, h = i[ok][:max_rows], o[ok][:max_rows], h[ok][:max_rows]
    return i, o, h


def fit_global_model(fb: FeatureBuilder, train_end: int, H: int, max_rows: int, seed: int = 0):
    """Train one Poisson gradient-boosting model on all products, targets <= train_end."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed + train_end)
    rows = _sample_training_rows(fb.start, train_end, H, max_rows, rng)
    if rows is None or len(rows[0]) < 500:
        return None, {"train_end": train_end, "rows": 0}
    i, o, h = rows
    X, base = fb.build(i, o, h)
    y = fb.Y[i, o + h]
    # Features that need a year of history (last-year ones) are all-missing for
    # early refits; give the model a constant column instead, which it ignores.
    X[:, np.isnan(X).all(axis=0)] = 0.0
    cat = np.array([name in CATEGORICAL for name in FEATURE_NAMES])
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.1, max_iter=300, max_leaf_nodes=48, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=cat, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(X, np.minimum(y / base, RATIO_CAP), sample_weight=base)
    info = {"train_end": int(train_end), "rows": int(len(y)), "iterations": int(model.n_iter_),
            "seconds": round(time.perf_counter() - t0, 2)}
    return model, info


def predict_global(model, fb: FeatureBuilder, o_idx: np.ndarray, H: int) -> np.ndarray:
    """ML forecasts (n, len(o_idx), H) at the given origins."""
    n = fb.n
    o_idx = np.asarray(o_idx)
    I, O, Hh = np.meshgrid(np.arange(n), o_idx, np.arange(1, H + 1), indexing="ij")
    X, base = fb.build(I.ravel(), O.ravel(), Hh.ravel())
    with np.errstate(over="ignore"):
        ratio = np.minimum(model.predict(X), RATIO_CAP)
    pred = (ratio * base).reshape(n, len(o_idx), H)
    # Guard rail: never forecast a single day above 2x the product's best day so far.
    cap = 2.0 * np.maximum(fb.run_max[np.arange(n)[:, None], o_idx[None, :]], 1.0)[:, :, None]
    alive = (o_idx[None, :] >= fb.start[:, None])[:, :, None]
    return np.clip(np.minimum(pred, cap), 0, None) * fb.open_f[O + Hh] * alive


# ---------------------------------------------------------------------------
# Rolling-origin backtest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    methods: tuple[str, ...] = tuple(ALL_METHODS)
    val_weeks: int = 26
    test_weeks: int = 26
    refit_weeks: int = 13
    calendar: Optional[str] = None
    H: int = H_MAX
    ml_max_rows: int = 80_000
    seed: int = 0


@dataclass
class Backtest:
    config: BacktestConfig
    methods: list[str]
    origins: np.ndarray            # (n_o,) origin day indices (last observed day)
    val_start: int                 # first validation day = origins[0] + 1
    test_start: int                # first test day
    H: int
    F: dict[str, np.ndarray]       # method -> (n, n_o, H) daily forecasts
    final: dict[str, np.ndarray]   # method -> (n, H) forecasts from the last observed day
    final_dates: pd.DatetimeIndex
    open_bt: np.ndarray            # (T+H,) predicted open factor used in the backtest
    open_final: np.ndarray         # (T+H,) predicted open factor for the final forecast
    valid: np.ndarray              # (T,) store traded that day
    start: np.ndarray              # (n,) product life start
    scale_mase: np.ndarray         # (n,)
    scale_rmsse: np.ndarray        # (n,)
    calendar_final: StoreCalendar
    ml_log: list[dict] = field(default_factory=list)
    importance: Optional[pd.DataFrame] = None
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_test_origin(self) -> np.ndarray:
        return self.origins >= self.test_start - 1


def suggest_windows(n_days: int) -> tuple[int, int, int]:
    """(validation weeks, test weeks, max weeks each) sized to the history length.

    Keeps at least ~16 weeks of training before the first origin."""
    usable = max(0, n_days - 112)
    max_each = max(4, min(26, usable // 14))
    default = int(min(26, max(4, (n_days // 4) // 7)))
    default = min(default, max_each)
    return default, default, max_each


def _scales(Y: np.ndarray, start: np.ndarray, end: int):
    """Scaling for MASE (in-sample MAE of the weekly seasonal naive) and RMSSE
    (in-sample MSE of the naive), computed over each product's life up to the
    day before the test window, so it never sees test data. Floors keep a
    product launched shortly before the test window from producing absurd
    ratios off a near-zero scale."""
    n = Y.shape[0]
    mase = np.ones(n)
    rmsse = np.ones(n)
    for i in range(n):
        s = int(min(start[i], end))
        y = Y[i, s:end + 1]
        lvl = y.mean() if len(y) else 0.0
        if len(y) > 8:
            mase[i] = max(np.abs(y[7:] - y[:-7]).mean(), 0.1 * lvl, 1e-3)
        else:
            mase[i] = max(lvl, 1e-3)
        if len(y) > 2:
            rmsse[i] = max(((y[1:] - y[:-1]) ** 2).mean(), (0.1 * lvl) ** 2, 1e-6)
        else:
            rmsse[i] = max(lvl ** 2, 1e-6)
    return mase, rmsse


def run_backtest(Y: np.ndarray, dates: pd.DatetimeIndex, store_total: np.ndarray, config: BacktestConfig,
                 progress: Optional[Callable[[float, str], None]] = None) -> Backtest:
    """Forecast from every weekly origin across the validation + test windows, plus the final origin."""
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    def report(frac, msg):
        if progress:
            progress(frac, msg)

    n, T = Y.shape
    H = config.H
    val_days, test_days = 7 * config.val_weeks, 7 * config.test_weeks
    test_start = T - test_days
    o0 = test_start - 1 - val_days
    if o0 < 84:
        raise ValueError(f"Not enough history: the first forecast origin would be day {o0}; need at least 84 days "
                         f"of training. Shorten the validation/test windows.")
    origins = np.arange(o0, T - 1, ORIGIN_STEP)
    methods = [m for m in ALL_METHODS if m in config.methods]

    cal_bt = build_store_calendar(dates, store_total, config.calendar, H, rule_cutoff=o0 + 1)
    cal_fin = build_store_calendar(dates, store_total, config.calendar, H, rule_cutoff=T)
    valid = ~cal_fin.closed_observed
    dow_full = cal_fin.dates.dayofweek.to_numpy()
    open_bt, open_fin = cal_bt.open_factor(), cal_fin.open_factor()
    warnings = []
    from calendar_utils import years_without_holidays
    gaps = years_without_holidays(config.calendar, dates[0], cal_fin.dates[-1])
    if gaps:
        warnings.append(f"The holiday calendar has no data for {', '.join(map(str, gaps))} - holiday effects "
                        f"in those years are invisible to the models (update the `holidays` package).")

    report(0.02, "Running local methods forward through the history...")
    t0 = time.perf_counter()
    st = compute_local_states(Y, valid, dow_full[:T])
    timings["local_states"] = time.perf_counter() - t0
    F = local_forecasts(Y, st, origins, H, open_bt, dow_full, LOCAL_METHODS)
    final = local_forecasts(Y, st, np.array([T - 1]), H, open_fin, dow_full, LOCAL_METHODS)
    final = {k: v[:, 0, :] for k, v in final.items()}
    timings["local_forecasts"] = time.perf_counter() - t0 - timings["local_states"]

    ml_log: list[dict] = []
    importance = None
    if ML in methods:
        t0 = time.perf_counter()
        fb_bt = FeatureBuilder(Y, cal_bt, st)
        refit_every = max(1, config.refit_weeks) * 7
        refit_points = list(range(o0, T - 1, refit_every))
        ml_F = np.zeros((n, len(origins), H))
        first_model = None
        for j, r in enumerate(refit_points):
            report(0.1 + 0.75 * j / max(len(refit_points), 1),
                   f"Training the global ML model ({j + 1}/{len(refit_points) + 1}) on data up to {dates[r]:%d %b %Y}...")
            model, info = fit_global_model(fb_bt, r, H, config.ml_max_rows, config.seed)
            ml_log.append(info)
            sel = (origins >= r) & (origins < r + refit_every)
            if model is None:
                ml_F[:, sel, :] = F[SES][:, sel, :]
                continue
            if first_model is None:
                first_model = (model, r)
            ml_F[:, sel, :] = predict_global(model, fb_bt, origins[sel], H)
        F[ML] = ml_F
        report(0.88, "Training the final ML model on all data...")
        fb_fin = FeatureBuilder(Y, cal_fin, st)
        model_fin, info = fit_global_model(fb_fin, T - 1, H, config.ml_max_rows, config.seed)
        info["final"] = True
        ml_log.append(info)
        final[ML] = (predict_global(model_fin, fb_fin, np.array([T - 1]), H)[:, 0, :]
                     if model_fin is not None else final[SES])
        if first_model is not None:
            report(0.95, "Measuring feature importance on unseen validation data...")
            importance = _permutation_importance(first_model[0], fb_bt, origins, first_model[1], test_start, H,
                                                 config.seed)
        timings["ml"] = time.perf_counter() - t0

    if ENSEMBLE in methods:
        members = [m for m in (THETA, TSB, ML) if m in F] if ML in F else [THETA, TSB, SES]
        F[ENSEMBLE] = np.mean([F[m] for m in members], axis=0)
        final[ENSEMBLE] = np.mean([final[m] for m in members], axis=0)

    F = {m: F[m] for m in methods}
    final = {m: final[m] for m in methods}
    mase, rmsse = _scales(Y, st.start, test_start - 1)
    timings["total"] = time.perf_counter() - t_all
    report(1.0, "Done.")
    final_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    return Backtest(config=config, methods=methods, origins=origins, val_start=o0 + 1, test_start=test_start, H=H,
                    F=F, final=final, final_dates=final_dates, open_bt=open_bt, open_final=open_fin, valid=valid,
                    start=st.start, scale_mase=mase, scale_rmsse=rmsse, calendar_final=cal_fin, ml_log=ml_log,
                    importance=importance, timings=timings, warnings=warnings)


def _permutation_importance(model, fb: FeatureBuilder, origins, train_end, test_start, H, seed, max_rows=12_000):
    """Permutation importance of the first backtest model, measured on validation
    rows it never saw (targets after its training cut-off, before the test window)."""
    from sklearn.inspection import permutation_importance

    rng = np.random.default_rng(seed + 7)
    o_val = origins[(origins >= train_end) & (origins < test_start - 1)]
    if len(o_val) == 0:
        return None
    draws = max_rows * 2
    i = rng.integers(0, fb.n, draws)
    o = rng.choice(o_val, draws)
    h = rng.integers(1, H + 1, draws)
    ok = (o + h <= test_start - 1) & (o >= fb.start[i])
    i, o, h = i[ok][:max_rows], o[ok][:max_rows], h[ok][:max_rows]
    if len(i) < 200:
        return None
    X, base = fb.build(i, o, h)
    y = fb.Y[i, o + h]
    res = permutation_importance(model, X, np.minimum(y / base, RATIO_CAP), scoring="neg_mean_poisson_deviance", n_repeats=3,
                                 random_state=seed, n_jobs=1, sample_weight=base)
    return (pd.DataFrame({"feature": FEATURE_NAMES, "importance": res.importances_mean, "std": res.importances_std})
              .sort_values("importance", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Accuracy, protection-period errors, model selection
# ---------------------------------------------------------------------------

def actuals_matrix(Y: np.ndarray, origins: np.ndarray, H: int) -> np.ndarray:
    """(n, n_o, H) actual demand for each origin/horizon (NaN beyond the data)."""
    n, T = Y.shape
    tgt = origins[:, None] + np.arange(1, H + 1)[None, :]
    ok = tgt < T
    A = Y[:, np.clip(tgt, 0, T - 1)]
    return np.where(ok[None], A, np.nan)


def window_sums(F: np.ndarray, P: int, max_offset: int = 6) -> np.ndarray:
    """Sum daily values over a P-day window starting j days after the origin,
    for j = 0..max_offset. F: (n, n_o, H) -> (n, n_o, max_offset+1)."""
    n, n_o, H = F.shape
    if P + max_offset > H:
        raise ValueError(f"Window {P} + offset {max_offset} exceeds horizon {H}")
    c = _cum(F, 2)
    return np.stack([c[:, :, j + P] - c[:, :, j] for j in range(max_offset + 1)], axis=2)


def accuracy_by_product(bt: Backtest, Y: np.ndarray, P: int, origin_mask: np.ndarray,
                        forecasts: Optional[dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Long table: product x method -> MAE, RMSE, MASE, RMSSE, bias and protection-period error.

    Daily errors are taken over horizons 1..P; only origins in `origin_mask`
    whose P-day window is fully observed are used.
    """
    forecasts = bt.F if forecasts is None else forecasts
    n, T = Y.shape
    A = actuals_matrix(Y, bt.origins, bt.H)
    use = origin_mask & (bt.origins + P <= T - 1)
    rows = []
    if not use.any():
        return pd.DataFrame()
    Ad = A[:, use, :P]
    Aw = Ad.sum(axis=2)
    for m, Fm in forecasts.items():
        Fd = Fm[:, use, :P]
        e = Fd - Ad
        ew = Fd.sum(axis=2) - Aw
        mae = np.nanmean(np.abs(e), axis=(1, 2))
        mse = np.nanmean(e ** 2, axis=(1, 2))
        for i in range(n):
            rows.append(dict(
                product=i, method=m, mae=mae[i], rmse=np.sqrt(mse[i]), mase=mae[i] / bt.scale_mase[i],
                rmsse=np.sqrt(mse[i] / bt.scale_rmsse[i]), sum_err=np.nansum(e[i]), sum_abs_err=np.nansum(np.abs(e[i])),
                sum_actual=np.nansum(Ad[i]), window_rmse=np.sqrt(np.nanmean(ew[i] ** 2)),
                window_mae=np.nanmean(np.abs(ew[i])), window_bias=np.nanmean(ew[i]), window_mean=np.nanmean(Aw[i]),
            ))
    return pd.DataFrame(rows)


def summarize_accuracy(acc: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-level leaderboard from accuracy_by_product."""
    if acc.empty:
        return acc
    g = acc.groupby("method")
    out = pd.DataFrame({
        "MASE (mean)": g["mase"].mean(),
        "MASE (median)": g["mase"].median(),
        "RMSSE (mean)": g["rmsse"].mean(),
        "WAPE": g["sum_abs_err"].sum() / g["sum_actual"].sum(),
        "Bias": g["sum_err"].sum() / g["sum_actual"].sum(),
        "Protection-period error": g["window_mae"].sum() / g["window_mean"].sum(),
    })
    # Forecast value added (Gilliland 2010) on the measure that sizes safety stock. Daily WAPE is
    # not used here: for intermittent items it rewards forecasting zero (Kolassa 2016).
    if SEASONAL_NAIVE in out.index:
        base = out.loc[SEASONAL_NAIVE, "Protection-period error"]
        out["Value added vs naive"] = (base - out["Protection-period error"]) / base if base > 0 else np.nan
    return out.sort_values("Protection-period error")


def select_methods(bt: Backtest, Y: np.ndarray, P: int, origin_mask: np.ndarray,
                   candidates: Optional[list[str]] = None) -> np.ndarray:
    """Per product, the method with the lowest mean squared protection-period error
    (offset 0) over the given origins - the error that drives safety stock."""
    n, T = Y.shape
    candidates = [m for m in (candidates or bt.methods) if m in bt.F]
    use = origin_mask & (bt.origins + P <= T - 1)
    if not use.any():
        return np.array([candidates[0]] * n, dtype=object)
    A = actuals_matrix(Y, bt.origins[use], P).sum(axis=2)
    scores = np.stack([np.nanmean((bt.F[m][:, use, :P].sum(axis=2) - A) ** 2, axis=1) for m in candidates], axis=1)
    scores = np.where(np.isfinite(scores), scores, np.inf)
    return np.array(candidates, dtype=object)[np.argmin(scores, axis=1)]


def empirical_intervals(Y: np.ndarray, bt: Backtest, F: np.ndarray, final: np.ndarray,
                        levels: tuple[float, float] = (0.1, 0.9),
                        buckets=((1, 7), (8, 14), (15, 28), (29, 63))) -> tuple[np.ndarray, np.ndarray]:
    """Daily prediction band for the final forecast, from the backtest's own errors
    (empirical prediction intervals; Williams & Goodman 1971). Errors are
    standardised by sqrt(forecast + 1), pooled per horizon bucket, so the band
    widens and narrows with the forecast level and grows with the horizon."""
    n, H = final.shape
    A = actuals_matrix(Y, bt.origins, bt.H)
    Z = (A - F) / np.sqrt(np.maximum(F, 0) + 1)
    lo, hi = np.zeros((n, H)), np.zeros((n, H))
    open_f = bt.open_final[Y.shape[1]:Y.shape[1] + H]
    for a, b in buckets:
        if a > H:
            break
        b = min(b, H)
        z = Z[:, :, a - 1:b].reshape(n, -1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                qlo = np.nan_to_num(np.nanquantile(z, levels[0], axis=1))
                qhi = np.nan_to_num(np.nanquantile(z, levels[1], axis=1))
        f = np.maximum(final[:, a - 1:b], 0)
        s = np.sqrt(f + 1)
        lo[:, a - 1:b] = np.clip(f + qlo[:, None] * s, 0, None)
        hi[:, a - 1:b] = np.clip(f + qhi[:, None] * s, 0, None)
    return lo * open_f, hi * open_f


def compose_auto(bt: Backtest, choice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forecast arrays (backtest (n, n_o, H), final (n, H)) using each product's chosen method."""
    n = len(choice)
    F = np.empty_like(next(iter(bt.F.values())))
    fin = np.empty((n, bt.H))
    for i, m in enumerate(choice):
        F[i] = bt.F[m][i]
        fin[i] = bt.final[m][i]
    return F, fin
