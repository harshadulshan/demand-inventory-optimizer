"""
inventory.py
Turns forecasts into stock decisions - and proves, by simulating them on
held-out history, what those decisions would actually have done.

The chain
---------
1. **Protection period.** A periodic-review order-up-to policy (R, S) must
   hold enough stock to cover demand until the *next* order arrives: R + L
   days (review period + lead time). A continuous-review reorder-point
   policy (s, Q) only has to cover the lead time L.

2. **Lead-time demand distribution, calibrated on forecast errors.** The
   standard textbook mistake is to size safety stock from the variability
   of *demand*; the right quantity is the variability of the *forecast
   error* over the protection period (Silver, Pyke & Thomas 2017). Here the
   distribution of demand over the next P days is
       mean     = P-day forecast, corrected by its recent relative bias
       variance = phi x (forecast + 1)       (over-dispersed Poisson)
   with bias and dispersion phi estimated from past P-day windows that had
   fully happened by the planning date, weighted by recency (half-life one
   quarter), and shrunk toward the portfolio median for products with few
   windows. It is modelled as a negative binomial (count data, right-
   skewed - right for slow movers) or, distribution-free, by weighted
   split-conformal prediction (Barber, Candes, Ramdas & Tibshirani 2023).
   `calibration()` checks the result: of all windows where the model said
   "95% chance demand stays below S", demand did stay below S in ~95%.

3. **Service level from economics, not habit.** The cost-optimal chance of
   not running out in a cycle (cycle service level, CSL) is a newsvendor
   critical ratio (Silver, Pyke & Thomas 2017, ch. 6-7), per product:
       (R, S), lost sales:  CSL* = B / (B + h*R)
       (R, S), backorders:  CSL* = 1 - h*R / B
       (s, Q), lost sales:  CSL* = 1 - Q*h_yr / (Q*h_yr + D_yr*B)
       (s, Q), backorders:  CSL* = 1 - Q*h_yr / (D_yr*B)
   B = shortage cost per unit (lost margin x penalty multiplier), h =
   holding cost per unit per day, D_yr = annual demand, Q = EOQ
   (Harris 1913). High-margin cheap items earn ~99%; low-margin bulky
   ones much less.

4. **Simulation on held-out data.** Each policy is replayed day by day over
   the test window against what customers actually bought, with real lead
   times, pack sizes and lost sales or backorders - and compared with the
   rule of thumb most small shops use ("keep N days of average sales").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from forecasting import Backtest, actuals_matrix, window_sums

PRIOR_WEIGHT = 4.0      # pseudo-observations for shrinking per-product error statistics toward the portfolio
LEVEL_OFFSET = 1.0      # c in the error variance phi * (forecast + c), so a zero forecast still carries some risk
MAX_HISTORY = 52        # use at most the last 52 weekly error windows
HALF_LIFE_WEEKS = 13.0   # error windows lose half their weight every quarter (adapts to regime changes)
LEVEL_BANDWIDTH = 1.0    # localisation: a past window whose forecast level differs by a factor e (2.7x) gets weight 0.6


@dataclass(frozen=True)
class PolicyConfig:
    policy: str = "RS"            # "RS" periodic review order-up-to, "sQ" continuous review reorder point
    review_days: int = 7          # R (RS only)
    lead_days: int = 3            # L
    service_mode: str = "cost"    # "cost": per-product critical ratio; "target": same CSL for all
    target_csl: float = 0.95
    lost_sales: bool = True       # True: unmet demand is lost; False: customers wait (backorders)
    holding_rate: float = 0.25    # annual holding cost as a fraction of unit cost
    order_cost: float = 0.0       # fixed cost per order line
    penalty_mult: float = 1.0     # shortage cost per unit = penalty_mult x unit margin
    default_margin: float = 0.25  # gross margin assumed when unit cost is unknown
    dist: str = "nb"              # "nb" negative binomial, "empirical" normalised conformal
    baseline_buffer: float = 0.20 # rule of thumb: cover (R+L) days of 28-day average sales + this buffer
    csl_min: float = 0.50
    csl_max: float = 0.995

    @property
    def protection_days(self) -> int:
        return self.review_days + self.lead_days if self.policy == "RS" else self.lead_days

    @property
    def baseline_review(self) -> int:
        return self.review_days if self.policy == "RS" else 7


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------

def unit_economics(price: np.ndarray, cost: np.ndarray, default_margin: float):
    """Fill in missing prices/costs from a default gross margin.
    Returns (price, cost, margin, known) arrays; `known` flags real cost data."""
    price = np.asarray(price, dtype=float).copy()
    cost = np.asarray(cost, dtype=float).copy()
    m = float(np.clip(default_margin, 0.01, 0.95))
    known = np.isfinite(cost) & (cost > 0)
    p_ok = np.isfinite(price) & (price > 0)
    cost = np.where(~known & p_ok, price * (1 - m), cost)
    price = np.where(~p_ok & known, cost / (1 - m), price)
    both = ~np.isfinite(price) | ~np.isfinite(cost)
    price = np.where(both, 1.0, price)          # no money data at all: work in "price units"
    cost = np.where(both, 1.0 - m, cost)
    margin = np.maximum(price - cost, 0.01 * price)
    return price, cost, margin, known


def holding_cost_per_day(cost: np.ndarray, cfg: PolicyConfig) -> np.ndarray:
    return cost * cfg.holding_rate / 365.0


def eoq(daily_demand: np.ndarray, cost: np.ndarray, cfg: PolicyConfig, pack: np.ndarray) -> np.ndarray:
    """Economic order quantity (Harris 1913), rounded up to the pack size; at least one pack."""
    annual = 365.0 * np.maximum(daily_demand, 0)
    h_yr = np.maximum(cost * cfg.holding_rate, 1e-9)
    q = np.sqrt(2 * annual * max(cfg.order_cost, 0.0) / h_yr)
    q = np.maximum(q, 1.0)
    return np.maximum(np.ceil(q / pack) * pack, pack)


def service_levels(cfg: PolicyConfig, price, cost, margin, daily_demand, pack) -> np.ndarray:
    """Cycle service level per product: uniform target, or the cost-optimal critical ratio."""
    n = len(price)
    if cfg.service_mode != "cost":
        return np.full(n, float(np.clip(cfg.target_csl, cfg.csl_min, cfg.csl_max)))
    B = np.maximum(cfg.penalty_mult * margin, 1e-9)
    h = holding_cost_per_day(cost, cfg)
    if cfg.policy == "RS":
        hR = h * cfg.review_days
        csl = B / (B + hR) if cfg.lost_sales else 1 - hR / B
    else:
        Q = eoq(daily_demand, cost, cfg, pack)
        h_yr = cost * cfg.holding_rate
        D_yr = np.maximum(365.0 * daily_demand, 1e-9)
        csl = 1 - Q * h_yr / (Q * h_yr + D_yr * B) if cfg.lost_sales else 1 - Q * h_yr / (D_yr * B)
    return np.clip(csl, cfg.csl_min, cfg.csl_max)


# ---------------------------------------------------------------------------
# Lead-time demand distributions
# ---------------------------------------------------------------------------

def nb_ppf(q, mu, var) -> np.ndarray:
    """Quantile of a negative binomial with the given mean and variance (Poisson
    when the variance does not exceed the mean). Vectorised; zero when mu ~ 0."""
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    q = np.broadcast_to(np.asarray(q, dtype=float), mu.shape)
    out = np.zeros(mu.shape)
    pos = mu > 1e-9
    over = pos & (var > mu * 1.000001)
    poi = pos & ~over
    if poi.any():
        out[poi] = stats.poisson.ppf(q[poi], mu[poi])
    if over.any():
        r = mu[over] ** 2 / (var[over] - mu[over])
        out[over] = stats.nbinom.ppf(q[over], r, r / (r + mu[over]))
    return np.nan_to_num(out, nan=0.0, posinf=0.0)


def nb_pmf(x, mu: float, var: float) -> np.ndarray:
    """Probability mass of the same negative binomial / Poisson at integer points x (scalar mu, var)."""
    x = np.asarray(x)
    if mu <= 1e-9:
        return (x == 0).astype(float)
    if var <= mu * 1.000001:
        return stats.poisson.pmf(x, mu)
    r = mu * mu / (var - mu)
    return stats.nbinom.pmf(x, r, r / (r + mu))


def nb_sf(x, mu, var) -> np.ndarray:
    """P(D > x) for the same negative binomial / Poisson."""
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    x = np.broadcast_to(np.asarray(x, dtype=float), mu.shape)
    out = np.where(x < 0, 1.0, 0.0)
    pos = mu > 1e-9
    over = pos & (var > mu * 1.000001)
    poi = pos & ~over
    if poi.any():
        out[poi] = stats.poisson.sf(np.floor(x[poi]), mu[poi])
    if over.any():
        r = mu[over] ** 2 / (var[over] - mu[over])
        out[over] = stats.nbinom.sf(np.floor(x[over]), r, r / (r + mu[over]))
    return out


@dataclass
class LeadTimeModel:
    """Calibrated distribution of demand over a W-day window starting j days after an origin.

    Built from one forecast array (the method chosen for planning). Error
    statistics for planning origin k come only from earlier origins whose
    W-day window had completely happened by origin k.

    Error model (heteroscedastic, quasi-Poisson): for a window forecast f,
        actual = f + bias * (f + c) + noise,   Var(noise) = phi * (f + c)
    i.e. the bias scales with the forecast level and the variance grows in
    proportion to it, as in an over-dispersed Poisson GLM. A constant-
    variance model would carry a Christmas cake's large December errors into
    July and keep a safety stock of cakes all year; here the buffer shrinks
    with the forecast. (A fully multiplicative model, Var ~ f^2, was tested
    and rejected: on lumpy wholesale lines a single huge order after a quiet
    week made it explode.) The per-product bias and dispersion phi are
    shrunk toward the portfolio median when a product has few error windows.
    """
    W: int
    origins: np.ndarray
    FS: np.ndarray          # (n, n_o, 7) forecast window sums
    E: np.ndarray           # (n, n_o, 7) errors actual - forecast (NaN if not yet observable)
    final_FS: np.ndarray    # (n,) window sum of the final forecast (offset 0)
    T: int

    @classmethod
    def build(cls, Y: np.ndarray, bt: Backtest, F: np.ndarray, final: np.ndarray, W: int) -> "LeadTimeModel":
        FS = window_sums(F, W)
        A = actuals_matrix(Y, bt.origins, bt.H)
        AS = window_sums(np.nan_to_num(A, nan=0.0), W)
        last_day = bt.origins[:, None] + np.arange(7)[None, :] + W     # a window is usable once its last day is observed
        AS = np.where((last_day <= Y.shape[1] - 1)[None], AS, np.nan)
        return cls(W=W, origins=bt.origins, FS=FS, E=AS - FS, final_FS=final[:, :W].sum(axis=1), T=Y.shape[1])

    def _history(self, k: Optional[int], j: int):
        """(errors, past forecasts), each (n, m), usable when planning at origin index k (None = final)."""
        if k is None:
            idx = np.where(self.origins + self.W <= self.T - 1)[0][-MAX_HISTORY:]
            return self.E[:, idx, 0], self.FS[:, idx, 0]
        hi = k - int(np.ceil((j + self.W) / 7.0))
        if hi < 0:
            empty = np.empty((self.E.shape[0], 0))
            return empty, empty
        lo = max(0, hi - MAX_HISTORY + 1)
        return self.E[:, lo:hi + 1, j], self.FS[:, lo:hi + 1, j]

    def params(self, k: Optional[int], j: int = 0, half_life: float = HALF_LIFE_WEEKS):
        """Mean and variance of window demand (plus the inputs of the empirical quantile).

        Past error windows are weighted by recency, w = 0.5^(age / half_life)
        (age in weeks) - Brown's (1959) exponentially smoothed error tracking,
        so a shift in the forecast's bias or accuracy (a seasonal surge, a
        promotion period) feeds through within weeks rather than being
        averaged against a year of calm history.

        Returns (mu, var, n_eff, z, w, scale_now): n_eff is the effective
        number of windows (sum w)^2 / sum w^2, z the standardised residuals,
        w their weights and scale_now = sqrt(f_now + c)."""
        fc = self.final_FS if k is None else self.FS[:, k, j]
        e, f = self._history(k, j)
        n, m = e.shape
        fin = np.isfinite(e)
        age = (m - 1 - np.arange(m))[None, :]
        w = np.where(fin, 0.5 ** (age / half_life), 0.0)
        if LEVEL_BANDWIDTH and m > 0:
            # Localised weighting (cf. localized conformal prediction, Guan 2023): windows whose
            # forecast level resembled today's count more, so a seasonal line's off-season risk
            # is judged by off-season errors, not by last April's peak.
            gap = (np.log(np.where(fin, np.maximum(f, 0) + LEVEL_OFFSET, 1.0))
                   - np.log(np.maximum(fc, 0) + LEVEL_OFFSET)[:, None]) / LEVEL_BANDWIDTH
            w = w * np.exp(-0.5 * gap ** 2)
        sw = w.sum(axis=1)
        n_eff = np.where(sw > 0, sw ** 2 / np.maximum((w ** 2).sum(axis=1), 1e-12), 0.0)
        e0 = np.where(fin, e, 0.0)
        fp = np.where(fin, np.maximum(f, 0) + LEVEL_OFFSET, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel_bias = np.where(sw > 0, (w * e0).sum(axis=1) / np.maximum((w * fp).sum(axis=1), 1e-9), 0.0)
            resid = np.where(fin, e0 - rel_bias[:, None] * fp, 0.0)
        z = np.where(fin, resid / np.sqrt(np.where(fin, fp, 1.0)), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.where(n_eff > 1, (w * np.nan_to_num(z) ** 2).sum(axis=1) / np.maximum(sw, 1e-12)
                           * n_eff / np.maximum(n_eff - 1, 1e-9), np.nan)
        pool = (n_eff >= 6) & np.isfinite(phi)
        phi_prior = float(np.median(phi[pool])) if pool.sum() >= 3 else 1.5
        bias_prior = float(np.median(rel_bias[pool])) if pool.sum() >= 3 else 0.0
        a = n_eff / (n_eff + PRIOR_WEIGHT)
        bias_s = a * rel_bias + (1 - a) * bias_prior
        dof = np.maximum(n_eff - 1, 0)
        phi_s = (dof * np.nan_to_num(phi, nan=0.0) + PRIOR_WEIGHT * phi_prior) / (dof + PRIOR_WEIGHT)
        base = np.maximum(fc, 0) + LEVEL_OFFSET
        mu = np.maximum(fc + bias_s * base, 0.0)
        var = phi_s * base
        return mu, var, n_eff, z, w, np.sqrt(base)

    def quantile(self, q, k: Optional[int], j: int = 0, dist: str = "nb") -> np.ndarray:
        mu, var, n_eff, z, w, scale = self.params(k, j)
        level = nb_ppf(q, mu, var)
        if dist == "empirical" and z.shape[1] > 0:
            # Weighted (non-exchangeable) split-conformal quantile, Barber, Candes, Ramdas &
            # Tibshirani (2023): the smallest standardised residual whose cumulative weight
            # reaches q * (sum of weights + 1), the +1 being the weight of the new window.
            order = np.argsort(np.where(np.isfinite(z), z, np.inf), axis=1)
            zs = np.take_along_axis(np.where(np.isfinite(z), z, np.inf), order, axis=1)
            ws = np.take_along_axis(w, order, axis=1)
            cw = np.cumsum(ws, axis=1)
            qa = np.broadcast_to(np.asarray(q, dtype=float), mu.shape)
            need = qa * (w.sum(axis=1) + 1.0)
            pos = np.argmax(cw >= need[:, None], axis=1)
            reach = cw[:, -1] >= need
            pick = zs[np.arange(len(mu)), pos]
            ok = reach & (n_eff >= 6) & np.isfinite(pick)
            emp = np.ceil(np.clip(mu + np.where(ok, pick, 0.0) * scale, 0, None))
            level = np.where(ok, emp, level)
        return level


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    name: str
    on_hand: np.ndarray       # (n, D) end-of-day stock
    demand: np.ndarray        # (n, D)
    sold: np.ndarray          # (n, D) served from stock that day
    short: np.ndarray         # (n, D) unmet that day (lost or backordered)
    orders: np.ndarray        # (n, D) quantity ordered (morning of day d)
    level: np.ndarray         # (n, D) S or s in force each day (NaN when not reviewed)
    warmup: int
    kpi: dict = field(default_factory=dict)
    per_product: Optional[pd.DataFrame] = None


def simulate(demand: np.ndarray, level: np.ndarray, review: np.ndarray, L: int, policy: str,
             lost_sales: bool, init: np.ndarray, pack: np.ndarray, Q: Optional[np.ndarray] = None,
             name: str = "policy", warmup: int = 0) -> SimResult:
    """Replay a policy day by day.

    Each morning: receive deliveries due, review stock (if a review day),
    order; deliveries take L days (an order placed on day d is usable on
    day d+L). Then the day's demand is served from stock; the rest is lost
    (lost_sales) or backordered and filled from later deliveries first.
    RS: order up to level S on review days. sQ: whenever the inventory
    position is at or below s, order enough multiples of Q to rise above s.
    """
    n, D = demand.shape
    arrivals = np.zeros((n, D + L + 1))
    on = init.astype(float).copy()
    pipe = np.zeros(n)
    backlog = np.zeros(n)
    rec_on = np.zeros((n, D))
    rec_sold = np.zeros((n, D))
    rec_short = np.zeros((n, D))
    rec_ord = np.zeros((n, D))
    for d in range(D):
        arr = arrivals[:, d]
        on += arr
        pipe -= arr
        if not lost_sales:
            fill = np.minimum(on, backlog)
            on -= fill
            backlog -= fill
        ip = on + pipe - backlog
        if policy == "RS":
            if review[d]:
                q = np.maximum(level[:, d] - ip, 0.0)
                q = np.where(q > 0, np.ceil(q / pack - 1e-9) * pack, 0.0)
            else:
                q = np.zeros(n)
        else:
            s = level[:, d]
            need = ip <= s
            mult = np.floor((s - ip) / Q) + 1
            q = np.where(need, mult * Q, 0.0)
        if L == 0:
            on += q
        else:
            arrivals[:, d + L] += q
            pipe += q
        rec_ord[:, d] = q
        dem = demand[:, d]
        sold = np.minimum(on, dem)
        short = dem - sold
        on -= sold
        if not lost_sales:
            backlog += short
        rec_on[:, d], rec_sold[:, d], rec_short[:, d] = on, sold, short
    return SimResult(name=name, on_hand=rec_on, demand=demand, sold=rec_sold, short=rec_short, orders=rec_ord,
                     level=level, warmup=warmup)


def score(sim: SimResult, price, cost, margin, cfg: PolicyConfig, review: np.ndarray,
          csl_target: Optional[np.ndarray] = None) -> SimResult:
    """Attach KPIs (after the warm-up period) to a simulation result."""
    w = sim.warmup
    dem, sold, short, on, orders = (a[:, w:] for a in (sim.demand, sim.sold, sim.short, sim.on_hand, sim.orders))
    days = max(dem.shape[1], 1)
    h = holding_cost_per_day(cost, cfg)
    B = cfg.penalty_mult * margin
    holding = (on * h[:, None]).sum(axis=1)
    n_orders = (orders > 0).sum(axis=1)
    ordering = n_orders * cfg.order_cost
    shortage = (short * B[:, None]).sum(axis=1)
    inv_value = (on * cost[:, None]).mean(axis=1)
    dem_tot, sold_tot = dem.sum(axis=1), sold.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = np.where(dem_tot > 0, sold_tot / dem_tot, np.nan)
        days_cover = np.where(sold_tot > 0, on.mean(axis=1) / (sold_tot / days), np.nan)
    so_days = (short > 0).sum(axis=1)
    dem_days = (dem > 0).sum(axis=1)
    # achieved cycle service level (RS): no shortfall between this order's arrival and the next one's
    achieved = np.full(dem.shape[0], np.nan)
    if cfg.policy == "RS":
        R, L = cfg.review_days, cfg.lead_days
        rdays = [d for d in np.where(review)[0] if d >= w and d + L + R <= sim.demand.shape[1]]
        if rdays:
            ok = np.stack([(sim.short[:, d + L:d + L + R] <= 0).all(axis=1) for d in rdays], axis=1)
            achieved = ok.mean(axis=1)
    pp = pd.DataFrame({
        "fill_rate": fill, "stockout_days": so_days, "demand_days": dem_days, "avg_inventory_value": inv_value,
        "avg_inventory_units": on.mean(axis=1), "days_of_cover": days_cover, "orders": n_orders,
        "holding_cost": holding, "ordering_cost": ordering, "shortage_cost": shortage,
        "total_cost": holding + ordering + shortage, "lost_units": short.sum(axis=1),
        "lost_revenue": (short * price[:, None]).sum(axis=1), "units_demanded": dem_tot,
        "achieved_csl": achieved,
    })
    if csl_target is not None:
        pp["target_csl"] = csl_target
    tot_dem = dem_tot.sum()
    cogs = (sold * cost[:, None]).sum()
    avg_inv = inv_value.sum()
    sim.kpi = {
        "fill_rate": sold_tot.sum() / tot_dem if tot_dem > 0 else np.nan,
        "value_fill_rate": (sold * price[:, None]).sum() / max((dem * price[:, None]).sum(), 1e-9),
        "stockout_day_share": so_days.sum() / max(dem_days.sum(), 1),
        "avg_inventory_value": avg_inv,
        "inventory_turns": cogs * 365.0 / days / avg_inv if avg_inv > 0 else np.nan,
        "orders": int(n_orders.sum()),
        "holding_cost": holding.sum(), "ordering_cost": ordering.sum(), "shortage_cost": shortage.sum(),
        "total_cost": (holding + ordering + shortage).sum(),
        "lost_revenue": (short * price[:, None]).sum(),
        "days": days,
        "achieved_csl": float(np.nanmean(achieved)) if np.isfinite(achieved).any() else np.nan,
    }
    sim.per_product = pp
    return sim


# ---------------------------------------------------------------------------
# Policy backtest over the test window
# ---------------------------------------------------------------------------

@dataclass
class PolicyRun:
    optimized: SimResult
    baseline: SimResult
    csl: np.ndarray                 # (n,) target cycle service level used
    Q: Optional[np.ndarray]
    days: pd.DatetimeIndex          # simulated dates
    review: np.ndarray              # (D,) bool review days (optimized)


def _test_layout(bt: Backtest, T: int):
    D = T - bt.test_start
    k0 = int(np.where(bt.origins == bt.test_start - 1)[0][0])
    d = np.arange(D)
    return D, k0, d // 7, d % 7


def plan_levels(ltm: LeadTimeModel, bt: Backtest, T: int, q: np.ndarray, dist: str,
                need: Optional[np.ndarray] = None) -> np.ndarray:
    """Order-up-to (or reorder) level for every product on every test day that needs one, (n, D).

    Day d of the test window is planned from the latest weekly origin before
    it (index k0 + d//7) at offset d%7 - i.e. the forecast is at most six
    days old, as with a weekly planning run. Days not in `need` are NaN.
    """
    D, k0, m, j = _test_layout(bt, T)
    n = ltm.FS.shape[0]
    out = np.full((n, D), np.nan)
    need = np.ones(D, dtype=bool) if need is None else need
    cache: dict[tuple[int, int], np.ndarray] = {}
    for d in np.where(need)[0]:
        key = (k0 + m[d], j[d])
        if key not in cache:
            cache[key] = ltm.quantile(q, key[0], key[1], dist)
        out[:, d] = cache[key]
    return out


def baseline_levels(Y: np.ndarray, bt: Backtest, T: int, P: int, buffer: float) -> np.ndarray:
    """Rule of thumb: order up to (P days x average daily sales of the last 28 days) x (1 + buffer)."""
    D = T - bt.test_start
    cs = np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(Y, axis=1)], axis=1)
    out = np.zeros((Y.shape[0], D))
    for d in range(D):
        t = bt.test_start + d - 1
        lo = max(0, t - 27)
        avg = (cs[:, t + 1] - cs[:, lo]) / (t + 1 - lo)
        out[:, d] = np.ceil(avg * P * (1 + buffer))
    return out


def run_policy(Y: np.ndarray, bt: Backtest, F: np.ndarray, final: np.ndarray, cfg: PolicyConfig,
               price, cost, margin, pack: Optional[np.ndarray] = None,
               csl_override: Optional[np.ndarray] = None, ltm: Optional[LeadTimeModel] = None,
               baseline_buffer: Optional[float] = None, with_baseline: bool = True) -> PolicyRun:
    """Simulate the optimised policy and the rule-of-thumb baseline over the test window."""
    n, T = Y.shape
    pack = np.ones(n) if pack is None else np.maximum(np.asarray(pack, dtype=float), 1)
    D, k0, _, _ = _test_layout(bt, T)
    days = pd.RangeIndex(D)
    P = cfg.protection_days
    ltm = ltm or LeadTimeModel.build(Y, bt, F, final, P)
    demand = Y[:, bt.test_start:]
    # demand rate for EOQ / economics: forecast at the first test origin
    daily_rate = F[:, k0, :28].mean(axis=1)
    Q = eoq(daily_rate, cost, cfg, pack) if cfg.policy == "sQ" else None
    csl = csl_override if csl_override is not None else service_levels(cfg, price, cost, margin, daily_rate, pack)
    warmup = min(P, D // 4)
    if cfg.policy == "RS":
        review = (np.arange(D) % cfg.review_days) == 0
        level = plan_levels(ltm, bt, T, csl, cfg.dist, need=review)
        init = np.nan_to_num(level[:, 0], nan=0.0)
    else:
        review = np.ones(D, dtype=bool)
        level = plan_levels(ltm, bt, T, csl, cfg.dist)
        init = level[:, 0] + Q
    opt = simulate(demand, level, review, cfg.lead_days, cfg.policy, cfg.lost_sales, init, pack, Q,
                   name="Optimised policy", warmup=warmup)
    opt = score(opt, price, cost, margin, cfg, review, csl)
    if with_baseline:
        base = run_baseline(Y, bt, cfg, price, cost, margin, pack,
                            cfg.baseline_buffer if baseline_buffer is None else baseline_buffer)
    else:
        base = opt
    return PolicyRun(optimized=opt, baseline=base, csl=csl, Q=Q, days=days, review=review)


def run_baseline(Y: np.ndarray, bt: Backtest, cfg: PolicyConfig, price, cost, margin, pack, buffer: float) -> SimResult:
    """Simulate the rule of thumb over the test window (periodic review, same lead time)."""
    n, T = Y.shape
    pack = np.ones(n) if pack is None else np.maximum(np.asarray(pack, dtype=float), 1)
    D = T - bt.test_start
    Rb = cfg.baseline_review
    blev = baseline_levels(Y, bt, T, Rb + cfg.lead_days, buffer)
    breview = (np.arange(D) % Rb) == 0
    blev = np.where(breview[None, :], blev, np.nan)
    warmup = min(cfg.protection_days, D // 4)
    base = simulate(Y[:, bt.test_start:], blev, breview, cfg.lead_days, "RS", cfg.lost_sales,
                    np.nan_to_num(blev[:, 0]), pack, None, name="Rule of thumb", warmup=warmup)
    bcfg = PolicyConfig(**{**cfg.__dict__, "policy": "RS", "review_days": Rb})
    return score(base, price, cost, margin, bcfg, breview)


def frontier(Y, bt, F, final, cfg: PolicyConfig, price, cost, margin, pack=None,
             levels=(0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995),
             buffers=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)) -> pd.DataFrame:
    """Service-vs-stock trade-off: the policy at a range of uniform service levels,
    and the rule of thumb at a range of buffers."""
    n = Y.shape[0]
    ltm = LeadTimeModel.build(Y, bt, F, final, cfg.protection_days)
    rows = []
    for q in levels:
        run = run_policy(Y, bt, F, final, cfg, price, cost, margin, pack, csl_override=np.full(n, q), ltm=ltm,
                         with_baseline=False)
        k = run.optimized.kpi
        rows.append(dict(kind="Optimised policy", setting=f"CSL {q:.1%}", x=q, fill_rate=k["fill_rate"],
                         avg_inventory_value=k["avg_inventory_value"], total_cost=k["total_cost"],
                         achieved_csl=k["achieved_csl"]))
    for b in buffers:
        k = run_baseline(Y, bt, cfg, price, cost, margin, pack, b).kpi
        rows.append(dict(kind="Rule of thumb", setting=f"+{b:.0%} buffer", x=b, fill_rate=k["fill_rate"],
                         avg_inventory_value=k["avg_inventory_value"], total_cost=k["total_cost"],
                         achieved_csl=k["achieved_csl"]))
    return pd.DataFrame(rows)


def calibration(Y: np.ndarray, bt: Backtest, F: np.ndarray, final: np.ndarray, W: int, dist: str,
                levels=(0.5, 0.8, 0.9, 0.95, 0.98)) -> pd.DataFrame:
    """Probabilistic calibration on the test window: for each target level q, the share
    of W-day windows whose actual demand was <= the predicted q-quantile. A
    well-calibrated model sits on the diagonal."""
    ltm = LeadTimeModel.build(Y, bt, F, final, W)
    T = Y.shape[1]
    ks = [k for k, o in enumerate(bt.origins) if o >= bt.test_start - 1 and o + W <= T - 1]
    A = actuals_matrix(Y, bt.origins, bt.H)
    rows = []
    for q in levels:
        hits, total = 0, 0
        for k in ks:
            qq = ltm.quantile(np.full(Y.shape[0], q), k, 0, dist)
            act = A[:, k, :W].sum(axis=1)
            live = ltm.FS[:, k, 0] + np.abs(act) > 0       # skip products with no demand and no forecast
            hits += int((act[live] <= qq[live]).sum())
            total += int(live.sum())
        rows.append(dict(target=q, achieved=hits / total if total else np.nan, windows=total))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# What to order today
# ---------------------------------------------------------------------------

def purchase_plan(Y: np.ndarray, bt: Backtest, F: np.ndarray, final: np.ndarray, cfg: PolicyConfig,
                  price, cost, margin, on_hand, on_order, pack) -> pd.DataFrame:
    """Recommended order per product as of the last day of data."""
    n, T = Y.shape
    pack = np.maximum(np.asarray(pack, dtype=float), 1)
    on_hand = np.maximum(np.asarray(on_hand, dtype=float), 0)
    on_order = np.maximum(np.asarray(on_order, dtype=float), 0)
    P, L = cfg.protection_days, cfg.lead_days
    ltm_P = LeadTimeModel.build(Y, bt, F, final, P)
    ltm_L = ltm_P if P == L else LeadTimeModel.build(Y, bt, F, final, L)
    daily_rate = final[:, :28].mean(axis=1)
    csl = service_levels(cfg, price, cost, margin, daily_rate, pack)
    mu_P, var_P, cnt, _, _, _ = ltm_P.params(None)
    level = ltm_P.quantile(csl, None, 0, cfg.dist)
    ip = on_hand + on_order
    if cfg.policy == "RS":
        need = np.maximum(level - ip, 0)
        qty = np.where(need > 0, np.ceil(need / pack - 1e-9) * pack, 0)
        Q = None
    else:
        Q = eoq(daily_rate, cost, cfg, pack)
        qty = np.where(ip <= level, (np.floor((level - ip) / Q) + 1) * Q, 0)
    mu_L, var_L, _, _, _, _ = ltm_L.params(None)
    risk_now = nb_sf(ip, mu_L, var_L)               # chance of running out before this order can arrive
    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(daily_rate > 0, (ip + qty) / daily_rate, np.inf)
    recent = Y[:, max(0, T - 90):].sum(axis=1)
    status = np.where(recent == 0, "No sales in 90 days", np.where(risk_now > 0.5, "Urgent - stockout likely",
                      np.where(qty > 0, "Order", "OK")))
    df = pd.DataFrame({
        "forecast_7d": final[:, :7].sum(axis=1), "forecast_P": mu_P, "safety_stock": np.maximum(level - mu_P, 0),
        "target_csl": csl, "level": level, "on_hand": on_hand, "on_order": on_order, "position": ip,
        "order_qty": qty, "pack": pack, "order_value": qty * cost, "days_cover_after": cover,
        "risk_before_delivery": risk_now, "status": status, "calib_windows": cnt,
    })
    if Q is not None:
        df["Q"] = Q
    return df
