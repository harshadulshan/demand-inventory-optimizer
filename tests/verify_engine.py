"""
verify_engine.py - verification suite for the forecasting and inventory engines.

Run either way:
    python tests/verify_engine.py      (plain Python, prints a report)
    python -m pytest tests             (if pytest is installed)

What is checked
---------------
Forecasting
  * SES matches an independent scalar loop AND statsmodels' SimpleExpSmoothing.
  * Croston / SBA / TSB match straightforward reference loops; SBA = Croston x (1 - a/2).
  * Theta: regression slope matches numpy's polyfit; the drift term matches
    statsmodels' ThetaModel exactly (Hyndman & Billah 2003 form).
  * NO LOOKAHEAD: scrambling every observation after an origin leaves every
    method's forecast at that origin unchanged - including the ML model.
  * Seasonal naive has MASE 0 on a perfectly weekly-periodic series.
  * Demand classification puts textbook series in the right SBC quadrant.
Inventory
  * Negative binomial helper reproduces the requested mean and variance.
  * Simulation conserves stock (start + received = sold + end + in transit).
  * The critical-ratio service level matches the cost-minimising order-up-to
    level found by brute-force simulation (newsvendor check).
  * With a correct forecast, the calibrated lead-time distribution hits its
    target coverage, and simulated cycle service level is close to target.
Data
  * Day-first date detection, POS column guessing, returns netting,
    UCI-style cleaning (cancellations, postage, zero prices), spike capping,
    store-closure detection and learned holiday closures.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calendar_utils as cu  # noqa: E402
import data_sources as ds  # noqa: E402
import forecasting as fc  # noqa: E402
import inventory as inv  # noqa: E402


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def _ref_ses(y, alpha, warmup=fc.SES_WARMUP):
    lvl, out = 0.0, []
    for t, x in enumerate(y):
        if t < warmup:
            lvl = lvl + (x - lvl) / (t + 1)
        else:
            lvl = lvl + alpha * (x - lvl)
        out.append(lvl)
    return np.array(out)


def test_ses_matches_reference_and_statsmodels():
    rng = np.random.default_rng(0)
    y = rng.poisson(8, 300).astype(float)
    L, S, C = fc._ses_states(y[None, :], np.ones((1, 300), dtype=bool))
    for ai, a in enumerate(fc.ALPHAS):
        assert np.allclose(L[:, 0, ai], _ref_ses(y, a)), f"SES differs from reference loop at alpha={a}"
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    except ImportError:
        return
    a = 0.2
    ai = int(np.where(np.isclose(fc.ALPHAS, a))[0][0])
    w = fc.SES_WARMUP
    res = SimpleExpSmoothing(y[w:], initialization_method="known", initial_level=y[:w].mean()).fit(
        smoothing_level=a, optimized=False)
    assert np.allclose(L[w:, 0, ai], np.asarray(res.level)), "SES differs from statsmodels"


def _ref_croston(y, alpha):
    z, p, q, nd, ni = 0.0, 1.0, 0, 0, 0
    Z, P = [], []
    for x in y:
        q += 1
        if x > 0:
            nd += 1
            z = z + (x - z) / nd if nd <= 3 else z + alpha * (x - z)
            if nd >= 2:
                ni += 1
                p = p + (q - p) / ni if ni <= 3 else p + alpha * (q - p)
            q = 0
        Z.append(z)
        P.append(p)
    return np.array(Z), np.array(P)


def _ref_tsb(y, alpha, beta):
    pr, z, no, nd = 0.0, 0.0, 0, 0
    PR, Z = [], []
    for x in y:
        no += 1
        dflag = 1.0 if x > 0 else 0.0
        pr = pr + (dflag - pr) / no if no <= 14 else pr + beta * (dflag - pr)
        if x > 0:
            nd += 1
            z = z + (x - z) / nd if nd <= 3 else z + alpha * (x - z)
        PR.append(pr)
        Z.append(z)
    return np.array(PR), np.array(Z)


def test_croston_sba_tsb_match_reference():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.25) * (1 + rng.poisson(2, 400))
    y = y.astype(float)
    upd = np.ones((1, 400), dtype=bool)
    Z, P = fc._croston_states(y[None, :], upd)
    rz, rp = _ref_croston(y, fc.CROSTON_ALPHA)
    assert np.allclose(Z[:, 0], rz) and np.allclose(P[:, 0], rp), "Croston differs from reference loop"
    PR, Zt = fc._tsb_states(y[None, :], upd)
    rpr, rzt = _ref_tsb(y, fc.TSB_ALPHA, fc.TSB_BETA)
    assert np.allclose(PR[:, 0], rpr) and np.allclose(Zt[:, 0], rzt), "TSB differs from reference loop"
    # SBA correction as published (Syntetos & Boylan 2005)
    dates = pd.date_range("2024-01-01", periods=400)
    st = fc.compute_local_states(y[None, :], np.ones(400, dtype=bool), dates.dayofweek.to_numpy())
    of = np.ones(400 + 10)
    dow = pd.date_range("2024-01-01", periods=410).dayofweek.to_numpy()
    out = fc.local_forecasts(y[None, :], st, np.array([399]), 5, of, dow, [fc.CROSTON_SBA])
    assert np.isclose(out[fc.CROSTON_SBA][0, 0, 0], rz[-1] / rp[-1] * (1 - fc.CROSTON_ALPHA / 2))


def test_theta_matches_statsmodels():
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
    except ImportError:
        return
    rng = np.random.default_rng(2)
    y = 50 + 0.3 * np.arange(200) + rng.normal(0, 5, 200)
    res = ThetaModel(y, period=1, deseasonalize=False, use_test=False).fit()
    a, b0 = float(res.params["alpha"]), float(res.params["b0"])
    # 1) slope from cumulative regression sums == OLS slope
    t = np.arange(200.0)
    n, St, Sx, Stt, Stx = 200, t.sum(), y.sum(), (t * t).sum(), (t * y).sum()
    b = (n * Stx - St * Sx) / (n * Stt - St * St)
    assert np.isclose(b, np.polyfit(t, y, 1)[0]) and np.isclose(b, b0, rtol=1e-6), "Theta slope mismatch"
    # 2) drift term == statsmodels' trend component x (theta-1)/theta
    hh = np.arange(1, 11)
    ours = b / 2 * ((hh - 1) + 1 / a - (1 - a) ** len(y) / a)
    comp = res.forecast_components(10)
    assert np.allclose(ours, 0.5 * np.asarray(comp["trend"]), rtol=1e-6), "Theta drift differs from statsmodels"


def test_no_lookahead_all_methods():
    """Scramble everything after origin o; forecasts made at o must not change."""
    d = ds.generate_demo_data(end_date="2026-06-30", n_days=500, seed=3)
    o = 420
    Y1 = d.Y
    Y2 = d.Y.copy()
    rng = np.random.default_rng(9)
    Y2[:, o + 1:] = rng.poisson(30, Y2[:, o + 1:].shape)
    H = 21
    outs = []
    for Y in (Y1, Y2):
        cal = cu.build_store_calendar(d.dates, Y.sum(axis=0), "Sri Lanka", H, rule_cutoff=o + 1)
        valid = ~cal.closed_observed
        valid[o + 1:] = True            # closure status of future days is unknown at o anyway
        dow = cal.dates.dayofweek.to_numpy()
        st = fc.compute_local_states(Y, valid, dow[:len(d.dates)])
        f = fc.local_forecasts(Y, st, np.array([o]), H, cal.open_factor(), dow, fc.LOCAL_METHODS)
        fb = fc.FeatureBuilder(Y, cal, st)
        model, _ = fc.fit_global_model(fb, o, H, 20_000, seed=0)
        f[fc.ML] = fc.predict_global(model, fb, np.array([o]), H)
        outs.append(f)
    for m in outs[0]:
        assert np.allclose(outs[0][m], outs[1][m]), f"{m}: forecast at origin changed when future data changed"


def test_mase_zero_for_periodic_series():
    pattern = np.array([5, 7, 6, 8, 12, 20, 15], dtype=float)
    Y = np.tile(pattern, 60)[None, :]
    dates = pd.date_range("2024-01-01", periods=Y.shape[1])
    bt = fc.run_backtest(Y + 0.0, dates, Y[0], fc.BacktestConfig(methods=(fc.SEASONAL_NAIVE,), val_weeks=8,
                                                                 test_weeks=8, calendar=None))
    acc = fc.accuracy_by_product(bt, Y, 7, bt.is_test_origin)
    assert np.isclose(acc.loc[acc.method == fc.SEASONAL_NAIVE, "mae"].iloc[0], 0.0)


def test_demand_classification():
    rng = np.random.default_rng(4)
    T = 364
    smooth = rng.poisson(10, T)
    erratic = rng.negative_binomial(0.6, 0.6 / (0.6 + 10), T) + 1
    intermittent = (rng.random(T) < 0.2) * 1
    lumpy = (rng.random(T) < 0.15) * (1 + rng.negative_binomial(0.4, 0.4 / (0.4 + 20), T))
    Y = np.vstack([smooth, erratic, intermittent, lumpy]).astype(float)
    cls = fc.classify_demand(Y, np.ones(T, dtype=bool), np.ones(4))
    assert cls["pattern"].tolist() == ["smooth", "erratic", "intermittent", "lumpy"], cls["pattern"].tolist()


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def test_nb_helper_moments():
    from scipy import stats
    mu, var = np.array([10.0, 3.0, 50.0]), np.array([30.0, 3.0, 400.0])
    for m, v in zip(mu, var):
        r = m * m / (v - m) if v > m else None
        if r is None:
            assert np.isclose(inv.nb_ppf(0.5, np.array([m]), np.array([v]))[0], stats.poisson.ppf(0.5, m))
        else:
            mean, variance = stats.nbinom.stats(r, r / (r + m), moments="mv")
            assert np.isclose(mean, m) and np.isclose(variance, v)
    q = inv.nb_ppf(np.array([0.5, 0.9, 0.99]), np.full(3, 10.0), np.full(3, 30.0))
    assert q[0] <= q[1] <= q[2]


def test_simulation_conserves_stock():
    rng = np.random.default_rng(5)
    n, D, L = 3, 120, 4
    demand = rng.poisson([[3], [10], [0.5]], (n, D)).astype(float)
    level = np.full((n, D), 40.0)
    review = (np.arange(D) % 7) == 0
    for lost in (True, False):
        sim = inv.simulate(demand, level, review, L, "RS", lost, np.full(n, 20.0), np.ones(n))
        ordered = sim.orders.sum(axis=1)
        in_transit = sim.orders[:, D - L:].sum(axis=1)
        received = ordered - in_transit
        if lost:
            assert np.allclose(20 + received, sim.sold.sum(axis=1) + sim.on_hand[:, -1])
            assert np.allclose(sim.sold + sim.short, demand)
        else:
            # every unit demanded is eventually either shipped from stock or still owed
            shipped = 20 + received - sim.on_hand[:, -1]
            owed = demand.sum(axis=1) - shipped
            assert (owed >= -1e-9).all()


def test_critical_ratio_matches_brute_force():
    """For iid Poisson demand, the (R,S) lost-sales critical ratio should pick
    (almost) the same S as brute-force cost minimisation by simulation."""
    from scipy import stats
    rng = np.random.default_rng(6)
    lam, R, L, D = 6.0, 7, 3, 20_000
    price, cost = 100.0, 60.0
    cfg = inv.PolicyConfig(review_days=R, lead_days=L, holding_rate=4.0)   # high holding cost -> interior optimum
    margin = price - cost
    h = cost * cfg.holding_rate / 365
    csl = inv.service_levels(cfg, np.array([price]), np.array([cost]), np.array([margin]), np.array([lam]), np.ones(1))[0]
    S_cr = stats.poisson.ppf(csl, lam * (R + L))
    demand = rng.poisson(lam, (1, D)).astype(float)
    review = (np.arange(D) % R) == 0
    costs = {}
    for S in range(int(S_cr) - 8, int(S_cr) + 9):
        sim = inv.simulate(demand, np.full((1, D), float(S)), review, L, "RS", True, np.array([S]), np.ones(1))
        costs[S] = (sim.on_hand.sum() * h + sim.short.sum() * margin) / D
    S_best = min(costs, key=costs.get)
    assert abs(S_best - S_cr) <= 2, f"critical ratio S={S_cr}, brute-force optimum S={S_best}"
    # and the cost curve is flat near the optimum, so the CR choice costs < 1% extra
    assert costs[int(S_cr)] <= costs[S_best] * 1.01


def _iid_backtest(Y, origins, H, test_start):
    """Minimal stand-in for a Backtest: forecasts equal to the true mean."""
    return SimpleNamespace(origins=origins, H=H, test_start=test_start)


def test_calibration_and_achieved_service():
    rng = np.random.default_rng(7)
    n, T, H = 40, 900, 30
    mus = rng.uniform(0.3, 30, n)
    k = 3.0
    Y = rng.negative_binomial(k, k / (k + mus[:, None]), (n, T)).astype(float)
    origins = np.arange(200, T - 1, 7)
    test_start = origins[60] + 1
    bt = _iid_backtest(Y, origins, H, test_start)
    F = np.broadcast_to(mus[:, None, None], (n, len(origins), H)).copy()
    fin = np.broadcast_to(mus[:, None], (n, H)).copy()
    W = 10
    ltm = inv.LeadTimeModel.build(Y, bt, F, fin, W)
    A = fc.actuals_matrix(Y, origins, H)
    for q in (0.8, 0.9, 0.95):
        hits, tot = 0, 0
        for kk in range(60, len(origins)):
            if origins[kk] + W > T - 1:
                continue
            qq = ltm.quantile(np.full(n, q), kk, 0, "nb")
            act = A[:, kk, :W].sum(axis=1)
            hits += (act <= qq).sum()
            tot += n
        assert abs(hits / tot - q) < 0.035, f"coverage {hits / tot:.3f} vs target {q}"
    # simulated cycle service level near target for a (R,S) policy
    cfg = inv.PolicyConfig(review_days=7, lead_days=3, service_mode="target", target_csl=0.9)
    price, cost, margin, _ = inv.unit_economics(np.full(n, 10.0), np.full(n, 7.0), 0.3)
    run = inv.run_policy(Y, bt, F, fin, cfg, price, cost, margin, with_baseline=False)
    ach = run.optimized.kpi["achieved_csl"]
    assert 0.85 <= ach <= 0.97, f"achieved CSL {ach:.3f} for target 0.90"


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def test_dayfirst_and_column_guessing():
    assert ds.guess_dayfirst(pd.Series(["03/04/2025", "25/12/2025"])) is True
    assert ds.guess_dayfirst(pd.Series(["12/25/2025", "01/02/2025"])) is False
    m = ds.guess_columns(["Bill Date", "Item Code", "Item Name", "Qty", "Unit Price", "Unit Cost", "Cashier"])
    assert m == {"date": "Bill Date", "sku": "Item Code", "quantity": "Qty", "price": "Unit Price",
                 "cost": "Unit Cost", "description": "Item Name"}, m


def test_upload_parsing_with_returns():
    rows = []
    for d in pd.date_range("2025-01-01", periods=200):
        rows.append((d.strftime("%d/%m/%Y"), "A1", "Apple", 5, 100.0, 70.0))
        rows.append((d.strftime("%d/%m/%Y"), "B2", "Bread", 3, 50.0, None))
    rows.append(("15/03/2025", "A1", "Apple", -2, 100.0, 70.0))      # a return
    rows.append(("16/03/2025", "A1", "Apple", -9, 100.0, 70.0))      # return bigger than the day's sales
    df = pd.DataFrame(rows, columns=["Date", "SKU", "Name", "Qty", "Price", "Cost"])
    m = ds.guess_columns(df.columns.tolist())
    sd = ds.sales_data_from_table(df, m, dayfirst=True, top_n=10)
    i = sd.skus.index("A1")
    assert sd.Y[i, sd.dates.get_loc(pd.Timestamp("2025-03-15"))] == 3      # 5 sold - 2 returned
    assert sd.Y[i, sd.dates.get_loc(pd.Timestamp("2025-03-16"))] == 0      # clipped at zero
    assert np.isclose(sd.price[i], 100.0) and np.isclose(sd.cost[i], 70.0)
    assert np.isnan(sd.cost[sd.skus.index("B2")])
    assert sd.dates[0] == pd.Timestamp("2025-01-01") and sd.n_days == 200


def test_uci_cleaning_rules():
    raw = pd.DataFrame({
        "Invoice": ["1", "1", "1", "C2", "3", "4", "A5", "6"],
        "StockCode": ["85123A", "POST", "22423", "85123A", "22423", "22423", "B", "85123a"],
        "Description": ["HEART", "POSTAGE", "CAKESTAND", "HEART", "CAKESTAND", "CAKESTAND", "Adjust", "HEART"],
        "Quantity": [10, 1, 4, 3, 2, 5, 1, 6],
        "InvoiceDate": pd.to_datetime(["2010-12-01 09:00", "2010-12-01 09:00", "2010-12-01 09:00",
                                       "2010-12-01 15:00", "2010-12-02 10:00", "2010-12-02 11:00",
                                       "2010-12-02 12:00", "2010-12-03 10:00"]),
        "Price": [2.5, 18.0, 12.75, 2.5, 12.75, 0.0, 100.0, 2.5],
        "Customer ID": [1, 1, 1, 1, 2, 3, None, 4],
        "Country": ["UK"] * 8,
    })
    daily, products, log = ds.clean_retail_transactions(raw)
    got = {(r.date.strftime("%m-%d"), r.sku): r.units for r in daily.itertuples()}
    assert got == {("12-01", "85123A"): 7, ("12-01", "22423"): 4, ("12-02", "22423"): 2, ("12-03", "85123A"): 6}, got
    assert "POST" not in set(daily["sku"]) and "B" not in set(daily["sku"])


def test_spike_cleansing():
    rng = np.random.default_rng(8)
    y = rng.poisson(20, 300).astype(float)
    y[100] = 5000
    Yc, n_cap, removed = ds.cleanse_spikes(y[None, :])
    assert n_cap == 1 and Yc[0, 100] < 200 and removed > 4500
    assert np.array_equal(np.delete(Yc[0], 100), np.delete(y, 100))


def test_store_closures_and_learned_holidays():
    dates = pd.date_range("2010-01-01", "2011-12-31")
    rng = np.random.default_rng(10)
    total = rng.poisson(200, len(dates)).astype(float)
    total[dates.dayofweek == 5] = 0                                # never opens on Saturdays
    hol = cu.holiday_table("United Kingdom (England)", dates[0], dates[-1])
    closes_on = {"Christmas Day", "Boxing Day", "Good Friday"}
    for d, nm in zip(hol["date"], hol["name"]):
        if nm in closes_on:
            total[dates.get_loc(d)] = 0
    closed, wd = cu.detect_closures(dates, total)
    assert wd == (5,)
    cal = cu.build_store_calendar(dates, total, "United Kingdom (England)", horizon=30)
    assert closes_on <= cal.closure_holidays and "Easter Monday" not in cal.closure_holidays
    fut = cal.predicted_closed()[len(dates):]
    fut_dates = cal.dates[len(dates):]
    assert all(fut[i] == (fut_dates[i].dayofweek == 5) for i in range(len(fut_dates))
               if fut_dates[i] not in set(pd.to_datetime(["2012-01-02"])))


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    print(f"Running {len(tests)} verification checks...\n")
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  PASS  {name}  ({time.perf_counter() - t0:.1f}s)")
        except Exception:  # noqa: BLE001 - report every failure
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
