"""
Demand Forecasting & Inventory Optimizer
-----------------------------------------
Forecasts every product's daily demand with eight methods - from the
seasonal-naive benchmark, through the intermittent-demand specialists
(Croston-SBA, TSB), to a global gradient-boosting model trained across all
products at once - then turns those forecasts into stock decisions:
calibrated order-up-to levels, a service level set by each product's own
economics, and a purchase list for today. Every claim is backtested: the
policy is replayed day by day against held-out sales and compared with the
rule of thumb most shops actually use.

Why this problem: small and mid-sized retailers - supermarkets, pharmacies,
hardware stores, wholesalers - mostly reorder by habit ("same as last
month", "keep two weeks of stock"). The result is stockouts on fast movers
and dead stock tying up cash on slow movers, at the same time. Enterprise
demand-planning suites solve this with exactly this kind of machinery, at a
price no small business can pay.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calendar_utils as cu
import data_sources as ds
import forecasting as fc
import inventory as inv

st.set_page_config(page_title="Demand & Inventory Optimizer", layout="wide", page_icon="\U0001F4E6")


# ---------------------------------------------------------------------------
# Colour system (validated categorical palette; one colour per entity, fixed)
# ---------------------------------------------------------------------------

def _dark_mode() -> bool:
    try:
        return str(st.context.theme.type or "light") == "dark"
    except Exception:  # older Streamlit without st.context.theme
        return False


DARK = _dark_mode()
if DARK:
    SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
    MUTED, GRID, AXIS, RING, INK2 = "#898781", "#2c2c2a", "#383835", "#0e1117", "#c3c2b7"
    SEQ = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
else:
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    MUTED, GRID, AXIS, RING, INK2 = "#898781", "#e1e0d9", "#c3c2b7", "#ffffff", "#52514e"
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
CONTEXT = "rgba(137,135,129,0.45)"      # de-emphasised context marks (actual demand)
CRITICAL = "#d03b3b"                    # status colour - stockout risk only
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

METHOD_COLORS = {
    fc.ENSEMBLE: SERIES[0], fc.ML: SERIES[1], fc.THETA: SERIES[2], fc.TSB: SERIES[3],
    fc.CROSTON_SBA: SERIES[4], fc.SES: SERIES[5], fc.MOVING_AVERAGE: SERIES[6], fc.AUTO: SERIES[7],
    fc.SEASONAL_NAIVE: MUTED,
}
POLICY = {"opt": ("Optimised policy", SERIES[0]), "rule": ("Rule of thumb", SERIES[1])}
ABC_COLORS = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}
CURRENCY_SYMBOL = {"GBP": "£", "USD": "$", "EUR": "€", "INR": "₹"}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def style(fig: go.Figure, height: int = 360, legend: bool = True, hover: str = "closest") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=40 if legend else 12, b=8), hovermode=hover,
        font=dict(family=FONT), showlegend=legend, bargap=0.3,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, traceorder="normal"),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key, config={"displaylogo": False})


def money(x: float, cur: str) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    sym = CURRENCY_SYMBOL.get((cur or "").upper())
    ax = abs(x)
    if ax >= 1e9:
        s = f"{x / 1e9:,.2f}B"
    elif ax >= 1e6:
        s = f"{x / 1e6:,.2f}M"
    elif ax >= 1e4:
        s = f"{x / 1e3:,.1f}K"
    else:
        s = f"{x:,.0f}"
    return f"{sym}{s}" if sym else f"{cur} {s}".strip()


def num(x: float) -> str:
    """Compact number without currency (1.16M, 377.6K, 9,870)."""
    return money(x, "").strip()


def progress_column(**kw):
    try:
        return st.column_config.ProgressColumn(color=SERIES[0], **kw)
    except TypeError:  # older Streamlit without the colour option
        return st.column_config.ProgressColumn(**kw)


def pct(x: float, d: int = 1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_demo(end: str) -> ds.SalesData:
    return ds.generate_demo_data(end_date=end)


@st.cache_data(show_spinner=False)
def cached_uci(top_n: int, cleanse: bool, stamp: float) -> ds.SalesData:
    return ds.load_uci_online_retail(top_n=top_n, cleanse=cleanse)


@st.cache_data(show_spinner=False)
def cached_table(data: bytes, name: str) -> pd.DataFrame:
    return ds.read_uploaded_table(data, name)


@st.cache_data(show_spinner=False)
def cached_upload(data: bytes, name: str, mapping: tuple, dayfirst: bool, top_n: int, cleanse: bool,
                  currency: str) -> ds.SalesData:
    table = ds.read_uploaded_table(data, name)
    return ds.sales_data_from_table(table, dict(mapping), dayfirst, top_n, cleanse, currency, "Sri Lanka", name)


# ---------------------------------------------------------------------------
# Sidebar - data
# ---------------------------------------------------------------------------

SRC_DEMO = "Demo shop (synthetic, Colombo)"
SRC_UCI = "Real retailer: UCI Online Retail II"
SRC_UP = "Upload your own sales file"

st.title("\U0001F4E6 Demand Forecasting & Inventory Optimizer")
st.caption(
    "Forecast every product with eight methods - benchmarks, intermittent-demand specialists and a global "
    "gradient-boosting model - turn the forecasts into calibrated stock levels, and prove the policy by "
    "replaying it against held-out sales."
)

st.sidebar.title("\U0001F5C2️ Data")
source = st.sidebar.radio("Sales history", [SRC_DEMO, SRC_UCI, SRC_UP], index=0)

data: ds.SalesData | None = None
try:
    if source == SRC_DEMO:
        end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = cached_demo(end)
    elif source == SRC_UCI:
        if not ds.uci_cache_ready():
            st.sidebar.info("The real dataset is downloaded once from the UCI Machine Learning Repository "
                            "(about 45 MB) and cached in the app's data_cache folder.")
            if st.sidebar.button("⬇️ Download and prepare the dataset", type="primary"):
                bar = st.sidebar.progress(0.0, text="Connecting to UCI...")
                try:
                    if not ds.UCI_ZIP_PATH.exists():
                        ds.download_uci_online_retail(progress=lambda f, m: bar.progress(0.6 * f, text=m))
                    ds.build_uci_cache(progress=lambda f, m: bar.progress(min(0.6 + 0.4 * f, 1.0), text=m))
                except ds.DataError as e:
                    st.sidebar.error(str(e))
                    st.stop()
                st.rerun()
            st.subheader("Real data: UCI Online Retail II")
            st.markdown(
                "1,067,371 real transactions from a UK-based online gift wholesaler, December 2009 - December "
                "2011, published by the [UCI Machine Learning Repository](%s) under CC BY 4.0. Click "
                "**Download and prepare the dataset** in the sidebar. The first preparation reads a 1-million-row "
                "Excel file and takes a minute or two; after that it loads instantly." % ds.UCI_DATASET_PAGE)
            st.stop()
        top_n = st.sidebar.slider("Products to plan (top by revenue)", 10, 150, 40, 5)
        cleanse = st.sidebar.checkbox(
            "Treat one-off bulk orders as special orders", True,
            help="Caps a product's extreme single days (above Q3 + 10×IQR of its selling days) before "
                 "forecasting. Wholesalers usually fulfil such orders separately rather than from stock.")
        data = cached_uci(top_n, cleanse, ds.UCI_DAILY_CACHE.stat().st_mtime)
    else:
        up = st.sidebar.file_uploader("Sales file (CSV or Excel)", type=["csv", "txt", "xlsx", "xls"])
        if up is None:
            st.subheader("Plan with your own sales data")
            st.markdown(
                "Upload any export from your POS, ERP or accounting system - transaction lines or daily "
                "totals - with at least these columns:\n\n"
                "| Needed | Examples of column names it recognises |\n|---|---|\n"
                "| **Date** | Date, Bill Date, Invoice Date, Order Date |\n"
                "| **Product code** | SKU, Item Code, Product Code, Barcode, StockCode |\n"
                "| **Quantity sold** | Qty, Quantity, Units, Qty Sold |\n"
                "| Unit selling price *(optional)* | Price, Unit Price, Selling Price, MRP |\n"
                "| Unit cost *(optional)* | Cost, Unit Cost, Cost Price, Purchase Price |\n"
                "| Product name *(optional)* | Description, Item Name, Product Name |\n\n"
                "Returns can be negative quantities. Day-first dates (dd/mm/yyyy) are detected automatically. "
                "At least ~5 months of history is required; a year or more is much better. A sample file is "
                "included in the project's `sample_data` folder (`sample_pos_export.csv`).")
            st.stop()
        raw = up.getvalue()
        table = cached_table(raw, up.name)
        guess = ds.guess_columns([str(c) for c in table.columns])
        cols = ["-"] + [str(c) for c in table.columns]
        with st.sidebar.expander("Column mapping", expanded=True):
            mapping: dict[str, str | None] = {}
            for role, label in [("date", "Date *"), ("sku", "Product code *"), ("quantity", "Quantity sold *"),
                                ("price", "Unit selling price"), ("cost", "Unit cost"), ("description", "Product name")]:
                g = guess.get(role)
                choice = st.selectbox(label, cols, index=cols.index(g) if g in cols else 0, key=f"map_{role}")
                mapping[role] = None if choice == "-" else choice
            dayfirst = st.checkbox("Dates are day-first (dd/mm/yyyy)",
                                   value=ds.guess_dayfirst(table[mapping["date"]]) if mapping["date"] else True)
        currency = st.sidebar.text_input("Currency", "LKR")
        top_n = st.sidebar.slider("Products to plan (top by sales)", 5, 200, 50, 5)
        cleanse = st.sidebar.checkbox("Treat one-off bulk orders as special orders", False)
        data = cached_upload(raw, up.name, tuple(sorted(mapping.items(), key=lambda kv: kv[0])), dayfirst,
                             top_n, cleanse, currency)
except ds.DataError as e:
    st.error(str(e))
    st.stop()

n, T = data.Y.shape
v_def, t_def, w_max = fc.suggest_windows(T)

# ---------------------------------------------------------------------------
# Sidebar - planning settings
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Planning")
policy_label = st.sidebar.radio(
    "Ordering policy", ["Fixed order cycle (R, S)", "Reorder point (s, Q)"],
    help="(R, S): every R days, order enough to bring stock up to S - typical when a distributor visits weekly.\n\n"
         "(s, Q): check stock daily; when it falls to the reorder point s, order a quantity Q (EOQ).")
policy = "RS" if policy_label.startswith("Fixed") else "sQ"
c1, c2 = st.sidebar.columns(2)
R = int(c1.number_input("Order every (days)", 1, 28, 7, disabled=policy != "RS"))
L = int(c2.number_input("Lead time (days)", 1, 28, 3, help="Days from placing an order to having the goods on the shelf."))
svc = st.sidebar.radio(
    "Service level", ["Margin-based per product", "Same target for every product"],
    help="Margin-based: each product's chance of not running out is set by its own economics - lost margin vs. "
         "the cost of holding one more unit (a newsvendor critical ratio).")
target = st.sidebar.slider("Target chance of no stockout per cycle", 0.80, 0.995, 0.95, 0.005, format="%.3f",
                           disabled=svc.startswith("Margin"))
lost = st.sidebar.radio("When a product is out of stock...",
                        ["Customer buys elsewhere (lost sale)", "Customer waits (backorder)"]).startswith("Customer buys")

with st.sidebar.expander("\U0001F4B0 Costs", expanded=False):
    holding = st.slider("Holding cost (% of unit cost per year)", 5, 60, 25,
                        help="Capital, storage, spoilage and shrinkage. 20-30% is typical for retail.") / 100
    order_cost = st.number_input(f"Cost per order line ({data.currency})", 0.0, 1e7,
                                 {"LKR": 250.0, "GBP": 2.0}.get(data.currency.upper(), 0.0), step=10.0)
    penalty = st.slider("Stockout penalty (× lost margin)", 1.0, 5.0, 1.0, 0.5,
                        help="1 = a stockout only costs the margin on the lost sale. Higher values add the cost "
                             "of an annoyed customer who may not come back.")
    margin_default = st.slider("Gross margin to assume where unit cost is unknown (%)", 5, 80, 30) / 100

with st.sidebar.expander("\U0001F52E Forecasting", expanded=False):
    use_ml = st.checkbox("Include the global ML model", True,
                         help="Gradient boosting trained across all products. Most accurate on calendar effects; "
                              "makes the first run for new settings take 15-60 seconds.")
    cal_names = list(cu.HOLIDAY_CALENDARS)
    cal_default = data.default_calendar if data.default_calendar in cal_names else "None"
    calendar = st.selectbox("Public-holiday calendar", cal_names, index=cal_names.index(cal_default))
    plan_options = [fc.ENSEMBLE, fc.AUTO] + [m for m in fc.ALL_METHODS if m != fc.ENSEMBLE and (use_ml or m != fc.ML)]
    plan_method = st.selectbox("Forecast used for planning", plan_options, index=0,
                               help="Ensemble (average of Theta, TSB and the ML model) is the most robust choice in "
                                    "this app's backtests on both datasets. Auto picks, per product, the method "
                                    "with the smallest protection-period error on the validation window.")
    dist_label = st.radio("Lead-time demand model", ["Negative binomial", "Empirical errors (conformal)"],
                          help="How the uncertainty around the forecast is modelled. Both are calibrated on the "
                               "backtest's own forecast errors.")
    val_w = st.slider("Validation window (weeks)", 4, w_max, v_def)
    test_w = st.slider("Test window (weeks)", 4, w_max, t_def)
    refit_w = st.slider("Retrain the ML model every (weeks)", 4, 26, 13)

with st.sidebar.expander("\U0001F4CF Rule of thumb to compare against", expanded=False):
    buffer = st.slider("Keep (cycle + lead time) days of average sales plus a buffer of (%)", 0, 200, 20, 5) / 100
    st.caption("Average = last 28 days. This is how most small shops set stock levels.")

pcfg = inv.PolicyConfig(
    policy=policy, review_days=R, lead_days=L, service_mode="cost" if svc.startswith("Margin") else "target",
    target_csl=float(target), lost_sales=lost, holding_rate=holding, order_cost=float(order_cost),
    penalty_mult=float(penalty), default_margin=margin_default,
    dist="nb" if dist_label.startswith("Negative") else "empirical", baseline_buffer=buffer,
)
P = pcfg.protection_days
price, cost, margin, cost_known = inv.unit_economics(data.price, data.cost, margin_default)

# ---------------------------------------------------------------------------
# Heavy computation (cached per configuration)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _backtest_store() -> dict:
    """Process-wide store of finished backtests (shared across reruns and browser tabs).

    Managed by hand rather than with @st.cache_resource on the backtest itself,
    because the backtest reports live progress to an element created outside
    it - which Streamlit's cache cannot replay on later hits."""
    return {}


def run_backtest_cached(key: tuple, data_: ds.SalesData, progress, keep: int = 6) -> fc.Backtest:
    store = _backtest_store()
    if key in store:
        store[key] = store.pop(key)            # mark as most recently used
        return store[key]
    fp_, methods_, val_weeks, test_weeks, refit_weeks, cal = key
    cfg = fc.BacktestConfig(methods=methods_, val_weeks=val_weeks, test_weeks=test_weeks,
                            refit_weeks=refit_weeks, calendar=cal)
    result = fc.run_backtest(data_.Y, data_.dates, data_.store_total, cfg, progress=progress)
    store[key] = result
    while len(store) > keep:
        store.pop(next(iter(store)))
    return result


@st.cache_resource(show_spinner=False, max_entries=12)
def auto_selection(key: tuple, _bt: fc.Backtest, _Y: np.ndarray, P_days: int):
    ch_val = fc.select_methods(_bt, _Y, P_days, ~_bt.is_test_origin)
    ch_all = fc.select_methods(_bt, _Y, P_days, np.ones(len(_bt.origins), dtype=bool))
    F_auto, _ = fc.compose_auto(_bt, ch_val)
    _, fin_auto = fc.compose_auto(_bt, ch_all)
    return ch_val, ch_all, F_auto, fin_auto


@st.cache_resource(show_spinner=False, max_entries=24)
def policy_cached(key: tuple, _Y, _bt, _F, _fin, _pcfg, _price, _cost, _margin):
    return inv.run_policy(_Y, _bt, _F, _fin, _pcfg, _price, _cost, _margin)


@st.cache_resource(show_spinner=False, max_entries=24)
def frontier_cached(key: tuple, _Y, _bt, _F, _fin, _pcfg, _price, _cost, _margin):
    return inv.frontier(_Y, _bt, _F, _fin, _pcfg, _price, _cost, _margin)


@st.cache_resource(show_spinner=False, max_entries=12)
def accuracy_cached(key: tuple, _bt, _Y, _F_auto, P_days: int):
    acc = fc.accuracy_by_product(_bt, _Y, P_days, _bt.is_test_origin)
    acc_auto = fc.accuracy_by_product(_bt, _Y, P_days, _bt.is_test_origin, forecasts={fc.AUTO: _F_auto})
    acc = pd.concat([acc, acc_auto], ignore_index=True)
    return acc, fc.summarize_accuracy(acc)


@st.cache_resource(show_spinner=False, max_entries=12)
def calibration_cached(key: tuple, _Y, _bt, _F, _fin, W: int):
    return {d: inv.calibration(_Y, _bt, _F, _fin, W, d) for d in ("nb", "empirical")}


fp = data.fingerprint()
methods = tuple(m for m in fc.ALL_METHODS if use_ml or m != fc.ML)
bt_key = (fp, methods, int(val_w), int(test_w), int(refit_w), None if calendar == "None" else calendar)
progress_slot = st.empty()


def _progress(frac: float, msg: str) -> None:
    progress_slot.progress(float(np.clip(frac, 0.0, 1.0)), text=f"First run with these settings - {msg}")


try:
    bt = run_backtest_cached(bt_key, data, _progress)
except ValueError as e:
    st.error(f"{e}")
    st.stop()
progress_slot.empty()

ch_val, ch_all, F_auto, fin_auto = auto_selection((bt_key, P), bt, data.Y, P)


def forecasts_for(method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == fc.AUTO:
        return F_auto, fin_auto
    return bt.F[method], bt.final[method]


F_plan, fin_plan = forecasts_for(plan_method)
pol_key = (bt_key, plan_method, pcfg)
run = policy_cached(pol_key, data.Y, bt, F_plan, fin_plan, pcfg, price, cost, margin)
acc, leaderboard = accuracy_cached((bt_key, P), bt, data.Y, F_auto, P)
cls_now = fc.classify_demand(data.Y, bt.valid, price, start=bt.start)
cls_pre = fc.classify_demand(data.Y, bt.valid, price, end=bt.test_start, start=bt.start)
test_dates = data.dates[bt.test_start:]
o_k, b_k = run.optimized.kpi, run.baseline.kpi
cur = data.currency

for w in bt.warnings:
    st.warning(w)
if data.is_synthetic:
    st.info("You are looking at the **synthetic demo shop** - a generated Colombo mini-supermarket (42 products, "
            "2 years). Switch to the real UCI dataset or upload your own sales in the sidebar.", icon="\U0001F9EA")

# KPI row --------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
lb_plan = leaderboard.loc[plan_method] if plan_method in leaderboard.index else None
k1.metric("Forecast error", pct(lb_plan["Protection-period error"]) if lb_plan is not None else "–",
          f"{-lb_plan['Value added vs naive'] * 100:+.0f}% vs naive" if lb_plan is not None else None,
          delta_color="inverse", border=True,
          help=f"Mean absolute error of the forecast total over the {P}-day protection period, as a share of actual "
               "demand (test window) - the error that sizes safety stock. Delta: relative to the seasonal-naive benchmark.")
k2.metric("Fill rate", pct(o_k["fill_rate"]), f"{(o_k['fill_rate'] - b_k['fill_rate']) * 100:+.1f} pts vs rule",
          border=True, help="Share of customer demand served straight from stock.")
k3.metric("Stockout days", pct(o_k["stockout_day_share"]),
          f"{(o_k['stockout_day_share'] - b_k['stockout_day_share']) * 100:+.1f} pts vs rule",
          delta_color="inverse", border=True, help="Share of product-days with demand on which some demand went unserved.")
k4.metric(f"Avg stock ({cur})", num(o_k["avg_inventory_value"]),
          f"{(o_k['avg_inventory_value'] / max(b_k['avg_inventory_value'], 1e-9) - 1) * 100:+.0f}% vs rule",
          delta_color="off", border=True, help="Average value (at cost) of stock on hand across all products.")
k5.metric(f"Total cost ({cur})", num(o_k["total_cost"]),
          f"{(o_k['total_cost'] / max(b_k['total_cost'], 1e-9) - 1) * 100:+.0f}% vs rule",
          delta_color="inverse", border=True,
          help="Holding + ordering + stockout (lost margin × penalty) costs over the test window.")
st.caption(
    f"Simulated on **{test_w} weeks of held-out sales** ({test_dates[0]:%d %b %Y} - {test_dates[-1]:%d %b %Y}): "
    f"both policies faced exactly the same customers, a {L}-day lead time and the same review cycle. "
    f"Planning forecast: **{plan_method}**. {n} products, {T} days of history from {data.source}."
)

tab_port, tab_fc, tab_acc, tab_inv, tab_po, tab_about = st.tabs([
    "\U0001F4CA Portfolio", "\U0001F4C8 Forecasts", "\U0001F3AF Accuracy", "\U0001F3ED Inventory simulation",
    "\U0001F6D2 Purchase plan", "ℹ️ Methodology",
])

# ---------------------------------------------------------------------------
# Tab 1 - portfolio
# ---------------------------------------------------------------------------
with tab_port:
    st.subheader("What kind of demand does each product have?")
    st.caption("A product's demand pattern decides which forecasting family suits it (Syntetos, Boylan & Croston "
               "2005): how often it sells (ADI) and how much the quantities vary when it does (CV²). ABC ranks "
               "products by sales value; XYZ by how steady their weekly sales are.")
    dfp = cls_now.assign(sku=data.skus, name=data.names)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        live = np.isfinite(dfp["adi"]) & np.isfinite(dfp["cv2"])
        fig = go.Figure()
        for abc in ["A", "B", "C"]:
            d = dfp[live & (dfp["abc"] == abc)]
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["adi"], y=d["cv2"], mode="markers", name=f"{abc} items",
                marker=dict(size=11, color=ABC_COLORS[abc], line=dict(width=2, color=RING)),
                customdata=np.stack([d["sku"], d["name"], d["pattern"], d["mean_daily"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>%{customdata[2]} · ADI %{x:.2f} · "
                              "CV² %{y:.2f}<br>%{customdata[3]:.2f} units/day<extra></extra>"))
        x_max = float(max(dfp.loc[live, "adi"].max() * 1.25 if live.any() else 3, 3.0))
        y_max = float(max(dfp.loc[live, "cv2"].max() * 1.18 if live.any() else 1, 1.0))
        fig.add_vline(x=fc.ADI_CUTOFF, line_width=1, line_color=AXIS)
        fig.add_hline(y=fc.CV2_CUTOFF, line_width=1, line_color=AXIS)
        for text, xa, ya, xan, yan in [("Smooth", 0.01, 0.01, "left", "bottom"), ("Erratic", 0.01, 0.99, "left", "top"),
                                       ("Intermittent", 0.99, 0.01, "right", "bottom"), ("Lumpy", 0.99, 0.99, "right", "top")]:
            fig.add_annotation(text=text, xref="x domain", yref="y domain", x=xa, y=ya, xanchor=xan, yanchor=yan,
                               showarrow=False, font=dict(color=MUTED, size=13))
        fig.update_xaxes(type="log", range=[np.log10(0.95), np.log10(x_max)],
                         title="ADI: trading days per day with a sale (log scale)")
        fig.update_yaxes(range=[-0.1 * y_max, y_max], title="CV² of quantities when it sells")
        show(style(fig, 430), key="pattern_map")
        n_dead = int((~live).sum())
        if n_dead:
            st.caption(f"{n_dead} product(s) with no sales in the last year are not shown.")
    with c2:
        mat = pd.crosstab(dfp["abc"], dfp["xyz"]).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        share = (dfp.groupby(["abc", "xyz"])["value"].sum().unstack()
                 .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"]).fillna(0) / max(dfp["value"].sum(), 1e-9))
        text = [[f"{mat.iloc[r, c]} products<br>{share.iloc[r, c]:.0%} of sales" for c in range(3)] for r in range(3)]
        fig = go.Figure(go.Heatmap(
            z=mat.values, x=["X · steady", "Y · variable", "Z · erratic"],
            y=["A · top 80% of sales", "B · next 15%", "C · last 5%"], text=text, texttemplate="%{text}",
            colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)], showscale=False, xgap=2, ygap=2,
            hovertemplate="%{y} / %{x}: %{z} products<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.markdown("**ABC-XYZ matrix**")
        show(style(fig, 330, legend=False), key="abc_xyz")
        st.caption("AX items (valuable, steady) deserve the tightest planning; CZ items (cheap, erratic) are "
                   "candidates for order-on-demand.")

    st.markdown("**Store-wide daily sales**")
    tot = data.Y.sum(axis=0)
    ma = pd.Series(tot).rolling(28, min_periods=7).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.dates, y=tot, mode="lines", name="Units per day", line=dict(color=CONTEXT, width=1)))
    fig.add_trace(go.Scatter(x=data.dates, y=ma, mode="lines", name="28-day average", line=dict(color=SERIES[0], width=2)))
    closed = ~bt.valid
    if closed.any():
        fig.add_trace(go.Scatter(x=data.dates[closed], y=np.zeros(int(closed.sum())), mode="markers",
                                 name="Store closed (detected)", marker=dict(color=MUTED, size=6, symbol="line-ns-open")))
    fig.add_vrect(x0=data.dates[bt.val_start], x1=data.dates[bt.test_start - 1], fillcolor=rgba(SERIES[2], 0.06),
                  line_width=0, annotation_text="validation", annotation_position="top left",
                  annotation_font_color=MUTED)
    fig.add_vrect(x0=data.dates[bt.test_start], x1=data.dates[-1], fillcolor=rgba(SERIES[0], 0.06), line_width=0,
                  annotation_text="test (held out)", annotation_position="top left", annotation_font_color=MUTED)
    fig.update_yaxes(title="Units")
    show(style(fig, 300, hover="x unified"), key="store_sales")
    cal_f = bt.calendar_final
    notes = []
    if cal_f.closed_weekdays:
        notes.append("never trades on " + ", ".join(pd.Timestamp(2024, 1, 1 + w).day_name() for w in cal_f.closed_weekdays))
    if cal_f.closure_holidays:
        notes.append("closes on " + ", ".join(sorted(cal_f.closure_holidays)))
    if notes:
        st.caption("Learned from the history: the store " + "; ".join(notes) + ". Those days are excluded from model "
                   "updates and forecast as zero in advance.")

    st.markdown("**All products**")
    tbl = pd.DataFrame({
        "Code": data.skus, "Product": data.names, "Pattern": dfp["pattern"], "ABC": dfp["abc"], "XYZ": dfp["xyz"],
        "Avg units/day": dfp["mean_daily"], "Days with a sale": 1 - dfp["zero_share"],
        "ADI": dfp["adi"].replace(np.inf, np.nan), "CV²": dfp["cv2"],
        "Share of sales value": dfp["value"] / max(dfp["value"].sum(), 1e-9),
    })
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "Avg units/day": st.column_config.NumberColumn(format="%.2f"),
        "Days with a sale": st.column_config.NumberColumn(format="percent"),
        "ADI": st.column_config.NumberColumn(format="%.2f"),
        "CV²": st.column_config.NumberColumn(format="%.2f"),
        "Share of sales value": progress_column(format="percent", min_value=0.0,
                                                max_value=float(max(tbl["Share of sales value"].max(), 1e-9))),
    })

# ---------------------------------------------------------------------------
# Tab 2 - forecasts
# ---------------------------------------------------------------------------
ltm_P = inv.LeadTimeModel.build(data.Y, bt, F_plan, fin_plan, P)
daily_rate = fin_plan[:, :28].mean(axis=1)
csl_now = inv.service_levels(pcfg, price, cost, margin, daily_rate, np.ones(n))
mu_P, var_P, neff_P, _, _, _ = ltm_P.params(None)
level_now = ltm_P.quantile(csl_now, None, 0, pcfg.dist)

with tab_fc:
    c1, c2 = st.columns([2, 3])
    with c1:
        i = st.selectbox("Product", range(n), key="fc_product",
                         format_func=lambda k: f"{data.skus[k]} - {data.names[k]}  ({cls_now['pattern'][k]}, {cls_now['abc'][k]})")
    with c2:
        options = list(dict.fromkeys([plan_method, fc.ENSEMBLE, fc.AUTO] + bt.methods))
        chosen = st.multiselect("Methods to compare (up to 4)", options, default=[plan_method, fc.SEASONAL_NAIVE]
                                if plan_method != fc.SEASONAL_NAIVE else [plan_method], max_selections=4)
    hist = min(T, 112)
    fut = bt.final_dates
    lo, hi = fc.empirical_intervals(data.Y, bt, F_plan, fin_plan)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.dates[-hist:], y=data.Y[i, -hist:], name="Actual sales", marker_color=CONTEXT,
                         hovertemplate="%{y:.0f} units<extra>actual</extra>"))
    fig.add_trace(go.Scatter(x=fut, y=hi[i], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=lo[i], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(METHOD_COLORS[plan_method], 0.14), name="80% range (planning forecast)",
                             hoverinfo="skip"))
    for m in chosen:
        _, fin_m = forecasts_for(m)
        fig.add_trace(go.Scatter(x=fut, y=fin_m[i], mode="lines", name=m, line=dict(width=2, color=METHOD_COLORS[m]),
                                 hovertemplate="%{y:.1f}<extra>" + m + "</extra>"))
    fig.update_yaxes(title="Units per day", rangemode="tozero")
    show(style(fig, 400, hover="x unified"), key="fc_chart")
    if plan_method == fc.AUTO:
        st.caption(f"Auto chose **{ch_all[i]}** for this product (lowest protection-period error over the backtest).")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**Demand over the next {P} days and the stock level that covers it**")
        mu_i, var_i, S_i = float(mu_P[i]), float(var_P[i]), float(level_now[i])
        upper = int(max(inv.nb_ppf(0.999, np.array([mu_i]), np.array([var_i]))[0], S_i) * 1.15) + 3
        xs = np.arange(0, upper + 1)
        pmf = inv.nb_pmf(xs, mu_i, var_i)
        tail = xs > S_i
        fig = go.Figure()
        fig.add_trace(go.Bar(x=xs[~tail], y=pmf[~tail], name="Covered by stock", marker_color=rgba(SERIES[0], 0.75),
                             hovertemplate="%{x} units: %{y:.2%}<extra></extra>"))
        fig.add_trace(go.Bar(x=xs[tail], y=pmf[tail], name="Would run out", marker_color=CRITICAL,
                             hovertemplate="%{x} units: %{y:.2%}<extra>stockout</extra>"))
        fig.add_vline(x=mu_i, line_width=1, line_color=MUTED)
        fig.add_vline(x=S_i, line_width=2, line_color=INK2)
        fig.add_annotation(x=mu_i, y=1, yref="paper", yanchor="top", xanchor="right", xshift=-4, showarrow=False,
                           text=f"expected {mu_i:,.0f}", font=dict(color=MUTED))
        fig.add_annotation(x=S_i, y=1, yref="paper", yanchor="top", xanchor="left", xshift=4, showarrow=False,
                           text=f"{'order-up-to' if policy == 'RS' else 'reorder point'} {S_i:,.0f}", font=dict(color=INK2))
        fig.update_xaxes(title=f"Units demanded over {P} days")
        fig.update_yaxes(title="Probability", tickformat=".1%")
        show(style(fig, 330), key="ltd_chart")
        risk = float(inv.nb_sf(np.array([S_i]), np.array([mu_i]), np.array([var_i]))[0])
        st.caption(
            f"Expected demand **{mu_i:,.0f}** units; holding **{S_i:,.0f}** "
            f"(safety stock {max(S_i - mu_i, 0):,.0f}) leaves a **{risk:.1%}** chance of running out before the next "
            f"delivery. Target for this product: {csl_now[i]:.1%} no-stockout"
            + (" (set by its margin and holding cost)." if pcfg.service_mode == "cost" else ".")
            + f" Calibrated on {neff_P[i]:.0f} effective past {P}-day windows.")
    with c2:
        st.markdown("**Track record on the held-out test window**")
        ks = [k for k, o in enumerate(bt.origins) if o >= bt.test_start - 1 and o + P <= T - 1]
        if ks:
            starts = data.dates[bt.origins[ks] + 1]
            act = np.array([data.Y[i, bt.origins[k] + 1:bt.origins[k] + 1 + P].sum() for k in ks])
            fsum = ltm_P.FS[i, ks, 0]
            day_idx = bt.origins[ks] + 1 - bt.test_start
            lvl = run.optimized.level[i, day_idx]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=starts, y=act, name=f"Actual {P}-day demand", marker_color=CONTEXT,
                                 hovertemplate="%{y:,.0f}<extra>actual</extra>"))
            fig.add_trace(go.Scatter(x=starts, y=fsum, mode="lines+markers", name="Forecast", line=dict(color=SERIES[0], width=2),
                                     marker=dict(size=8, line=dict(width=2, color=RING)),
                                     hovertemplate="%{y:,.0f}<extra>forecast</extra>"))
            fig.add_trace(go.Scatter(x=starts, y=lvl, mode="markers", name="Stock level the policy aimed for",
                                     marker=dict(size=10, symbol="line-ew-open", color=INK2, line=dict(width=3, color=INK2)),
                                     hovertemplate="%{y:,.0f}<extra>target level</extra>"))
            fig.update_yaxes(title="Units", rangemode="tozero")
            show(style(fig, 330, hover="x unified"), key="track_chart")
            st.caption("Each bar is what customers actually bought over the protection period starting that week; the "
                       "line is what was forecast beforehand, and the dash is the stock level the policy targeted.")
        else:
            st.info("The test window is too short for this protection period.")

    if bt.importance is not None:
        with st.expander("What drives the global ML model? (permutation importance on unseen validation data)"):
            imp = bt.importance.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"].str.replace("_", " "), orientation="h",
                                   marker_color=SERIES[1], error_x=dict(type="data", array=imp["std"], color=MUTED,
                                                                        thickness=1, width=0),
                                   hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
            fig.update_xaxes(title="Increase in Poisson deviance when the feature is shuffled")
            show(style(fig, 430, legend=False), key="imp_chart")
            st.caption("Features ending in 'rel' are relative to the product's recent level - the model learns "
                       "multiplicative patterns (weekday shape, holiday lift, momentum) that transfer across products.")

# ---------------------------------------------------------------------------
# Tab 3 - accuracy
# ---------------------------------------------------------------------------
with tab_acc:
    st.subheader("Which forecasting method is most accurate - on data it has never seen?")
    st.caption(f"Rolling-origin backtest: every week of the {test_w}-week test window, each method forecast the next "
               f"{fc.H_MAX} days using only data available that day. Scores below cover days 1-{P} ahead "
               f"(the protection period). MASE below 1 = better than the seasonal-naive benchmark.")
    lb = leaderboard.copy()
    lb.index.name = "Method"
    fmt = {"MASE (mean)": "{:.3f}", "MASE (median)": "{:.3f}", "RMSSE (mean)": "{:.3f}", "WAPE": "{:.1%}",
           "Bias": "{:+.1%}", "Protection-period error": "{:.1%}", "Value added vs naive": "{:+.1%}"}
    hl = "background-color: rgba(42,120,214,0.15); font-weight: 600;"
    st.dataframe(lb.style.format(fmt, na_rep="–").apply(
        lambda row: [hl if row.name == plan_method else "" for _ in row], axis=1), width="stretch")
    st.caption("**Protection-period error** - mean absolute error of the forecast *total* over the next "
               f"{P} days, as a share of actual demand - is what sizes safety stock. **Bias**: positive = over-forecast. "
               "Highlighted: the method used for planning.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Protection-period error by method**")
        s = leaderboard["Protection-period error"].sort_values(ascending=False)
        colors = [METHOD_COLORS.get(m, MUTED) if m == plan_method else MUTED for m in s.index]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                               text=[f"{v:.1%}" for v in s.values], textposition="outside",
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", range=[0, float(s.max()) * 1.18])
        show(style(fig, 360, legend=False), key="ppe_chart")
    with c2:
        st.markdown("**Which method wins for which demand pattern?** (mean MASE, test window)")
        a2 = acc.merge(pd.DataFrame({"product": range(n), "pattern": cls_pre["pattern"]}), on="product")
        piv = a2.pivot_table(index="method", columns="pattern", values="mase", aggfunc="mean")
        piv = piv.reindex(index=[m for m in leaderboard.index if m in piv.index],
                          columns=[p for p in ["smooth", "erratic", "intermittent", "lumpy"] if p in piv.columns])
        counts = pd.Series(cls_pre["pattern"]).value_counts()
        best = piv.idxmin(axis=0)
        txt = [[f"{piv.iloc[r, c]:.2f}" + (" ★" if piv.index[r] == best.iloc[c] else "") for c in range(piv.shape[1])]
               for r in range(piv.shape[0])]
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=[f"{p} ({counts.get(p, 0)})" for p in piv.columns], y=piv.index, text=txt,
            texttemplate="%{text}", colorscale=[[k / (len(SEQ) - 1), c] for k, c in enumerate(SEQ)], showscale=False,
            xgap=2, ygap=2, hovertemplate="%{y} on %{x}: MASE %{z:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(style(fig, 360, legend=False), key="pattern_heat")
        st.caption("Darker = larger error. ★ = most accurate method for that pattern (number of products in brackets).")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**Is the uncertainty honest?** - calibration of the {P}-day demand distribution")
        cal = calibration_cached((bt_key, plan_method, P), data.Y, bt, F_plan, fin_plan, P)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0.45, 1.0], y=[0.45, 1.0], mode="lines", name="Perfect calibration",
                                 line=dict(color=MUTED, width=1)))
        for d, label, col in [("nb", "Negative binomial", SERIES[0]), ("empirical", "Empirical (conformal)", SERIES[1])]:
            c = cal[d]
            fig.add_trace(go.Scatter(x=c["target"], y=c["achieved"], mode="lines+markers", name=label,
                                     line=dict(color=col, width=2), marker=dict(size=9, line=dict(width=2, color=RING)),
                                     hovertemplate="target %{x:.0%} → achieved %{y:.1%}<extra>" + label + "</extra>"))
        fig.update_xaxes(title="Promised chance that demand stays below the level", tickformat=".0%", range=[0.45, 1.0])
        fig.update_yaxes(title="How often it actually did (test window)", tickformat=".0%", range=[0.3, 1.0])
        show(style(fig, 360), key="calib_chart")
        st.caption("Points on the diagonal mean the stated risk is the real risk. Below the line = the model is "
                   "over-confident (typical in a sudden demand surge it has never seen).")
    with c2:
        st.markdown("**Auto mode: which method each product got** (chosen on the validation window)")
        vc = pd.Series(ch_val).value_counts()
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h",
                               marker_color=[METHOD_COLORS.get(m, MUTED) for m in vc.index],
                               text=vc.values, textposition="outside", hovertemplate="%{y}: %{x} products<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Products", range=[0, float(vc.max()) * 1.2])
        show(style(fig, 360, legend=False), key="auto_chart")

# ---------------------------------------------------------------------------
# Tab 4 - inventory simulation
# ---------------------------------------------------------------------------
with tab_inv:
    st.subheader("Would this policy have beaten the rule of thumb?")
    pol_desc = (f"every {R} days, order up to S (covers {P} days)" if policy == "RS"
                else f"reorder point s (covers the {L}-day lead time), order quantity Q = EOQ")
    st.caption(f"Day-by-day replay of the test window. Optimised: {pol_desc}, levels from the calibrated forecast "
               f"distribution, {'margin-based service level per product' if pcfg.service_mode == 'cost' else f'{target:.1%} target'}. "
               f"Rule of thumb: every {pcfg.baseline_review} days, order up to {pcfg.baseline_review + L} days of the last "
               f"28 days' average sales + {buffer:.0%}. {'Lost sales' if lost else 'Backorders'}; same lead time for both.")

    def _diff(a: float, b: float, kind: str) -> str:
        if not (np.isfinite(a) and np.isfinite(b)):
            return "–"
        if kind == "pts":
            return f"{(a - b) * 100:+.1f} pts"
        if kind == "num":
            return f"{a - b:+,.1f}" if abs(a - b) < 100 else f"{a - b:+,.0f}"
        if not b:
            return "–"
        rel = (a / b - 1) * 100
        return "0%" if abs(rel) < 0.5 else f"{rel:+.0f}%"

    spec = [
        ("Fill rate", "fill_rate", pct, "pts"),
        ("Stockout days", "stockout_day_share", pct, "pts"),
        ("No-stockout cycles", "achieved_csl", pct, "pts"),
        (f"Avg stock ({cur})", "avg_inventory_value", num, "rel"),
        ("Stock turns / year", "inventory_turns", lambda v: f"{v:.1f}", "num"),
        ("Order lines", "orders", lambda v: f"{v:,.0f}", "rel"),
        (f"Holding cost ({cur})", "holding_cost", num, "rel"),
        (f"Ordering cost ({cur})", "ordering_cost", num, "rel"),
        (f"Stockout cost ({cur})", "shortage_cost", num, "rel"),
        (f"Total cost ({cur})", "total_cost", num, "rel"),
        (f"Revenue lost ({cur})", "lost_revenue", num, "rel"),
    ]
    rows = [(label, f(o_k[k]), f(b_k[k]), _diff(float(o_k[k]), float(b_k[k]), kind)) for label, k, f, kind in spec]
    c1, c2 = st.columns([5, 7], gap="large")
    with c1:
        st.markdown("**Results over the test window**")
        st.dataframe(pd.DataFrame(rows, columns=["", "Optimised", "Rule of thumb", "Difference"]), hide_index=True,
                     width="stretch", height=422)
        st.caption("Stockout cost = lost margin × penalty. No-stockout cycles: share of order cycles with no "
                   "unmet demand between a delivery and the next one.")
    with c2:
        st.markdown("**Stock invested vs. service delivered** (each point = one full simulation)")
        fr = frontier_cached((bt_key, plan_method, pcfg), data.Y, bt, F_plan, fin_plan, pcfg, price, cost, margin)
        fo = fr[fr["kind"] == "Optimised policy"]
        fb = fr[fr["kind"] == "Rule of thumb"].sort_values("x").reset_index(drop=True)
        # show the rule of thumb up to the first buffer that reaches the optimised curve's best fill rate;
        # beyond that it only adds stock and squeezes the interesting part of the chart
        reach = np.where(fb["fill_rate"].to_numpy() >= fo["fill_rate"].max())[0]
        fb = fb.iloc[: (reach[0] + 1 if len(reach) else len(fb))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fo["avg_inventory_value"], y=fo["fill_rate"], mode="lines+markers",
                                 name="Optimised, service target 50-99.5%", line=dict(color=SERIES[0], width=2),
                                 marker=dict(size=9, line=dict(width=2, color=RING)), customdata=fo["setting"],
                                 hovertemplate="%{customdata}<br>stock %{x:,.0f}<br>fill %{y:.1%}<extra>optimised</extra>"))
        fig.add_trace(go.Scatter(x=fb["avg_inventory_value"], y=fb["fill_rate"], mode="lines+markers",
                                 name=f"Rule of thumb, buffer 0-{fb['x'].max():.0%}", line=dict(color=SERIES[1], width=2),
                                 marker=dict(size=9, line=dict(width=2, color=RING)), customdata=fb["setting"],
                                 hovertemplate="%{customdata}<br>stock %{x:,.0f}<br>fill %{y:.1%}<extra>rule of thumb</extra>"))
        for kpi, (label, col), pos in [(o_k, POLICY["opt"], "top left"), (b_k, POLICY["rule"], "bottom right")]:
            fig.add_trace(go.Scatter(x=[kpi["avg_inventory_value"]], y=[kpi["fill_rate"]], mode="markers+text",
                                     marker=dict(size=15, color=col, symbol="diamond", line=dict(width=2, color=RING)),
                                     text=[f"your {label.lower()}"], textposition=pos, showlegend=False,
                                     textfont=dict(color=INK2),
                                     hovertemplate="current settings<br>stock %{x:,.0f}<br>fill %{y:.1%}<extra>" + label + "</extra>"))
        fig.update_xaxes(title=f"Average stock held ({cur}, at cost)", rangemode="tozero")
        fig.update_yaxes(title="Fill rate", tickformat=".0%")
        show(style(fig, 400), key="frontier")
        st.caption("Up and to the left is better: more demand served per unit of money tied up in stock. "
                   "The rule of thumb can only move along its curve by guessing a buffer; the optimised policy sets "
                   "each product's buffer from its own forecast uncertainty and economics.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**Where the money goes** ({cur}, test window)")
        comps = ["Holding", "Ordering", "Stockouts", "Total"]
        keys = ["holding_cost", "ordering_cost", "shortage_cost", "total_cost"]
        fig = go.Figure()
        for kpi, (label, col) in [(o_k, POLICY["opt"]), (b_k, POLICY["rule"])]:
            vals = [kpi[k] for k in keys]
            fig.add_trace(go.Bar(x=comps, y=vals, name=label, marker_color=col, text=[num(v) for v in vals],
                                 textposition="outside", cliponaxis=False,
                                 hovertemplate="%{x}: %{y:,.0f}<extra>" + label + "</extra>"))
        fig.update_layout(barmode="group", bargap=0.45, bargroupgap=0.12)
        fig.update_yaxes(rangemode="tozero", range=[0, max(o_k["total_cost"], b_k["total_cost"]) * 1.18])
        show(style(fig, 340), key="cost_bars")
    with c2:
        st.markdown("**Fill rate by ABC class**")
        abc = cls_pre["abc"].to_numpy()
        classes = [("A", "A · top 80% of sales"), ("B", "B · next 15%"), ("C", "C · last 5%")]
        vals = {}
        for sim, key in [(run.optimized, "opt"), (run.baseline, "rule")]:
            w0 = sim.warmup
            vals[key] = []
            for c, _ in classes:
                msk = abc == c
                dem = sim.demand[msk, w0:].sum()
                vals[key].append(sim.sold[msk, w0:].sum() / dem if dem > 0 else np.nan)
        ylab = [lab for _, lab in classes]
        fig = go.Figure()
        for r, lab in enumerate(ylab):        # connector between the two policies (dumbbell)
            fig.add_trace(go.Scatter(x=[vals["rule"][r], vals["opt"][r]], y=[lab, lab], mode="lines",
                                     line=dict(color=GRID if not DARK else AXIS, width=4), showlegend=False,
                                     hoverinfo="skip"))
        for key in ["rule", "opt"]:
            label, col = POLICY[key]
            fig.add_trace(go.Scatter(x=vals[key], y=ylab, mode="markers+text", name=label,
                                     marker=dict(size=14, color=col, line=dict(width=2, color=RING)),
                                     text=[pct(v) for v in vals[key]],
                                     textposition="top center" if key == "opt" else "bottom center",
                                     textfont=dict(color=INK2), hovertemplate="%{y}: %{x:.1%}<extra>" + label + "</extra>"))
        allv = [v for v in vals["opt"] + vals["rule"] if np.isfinite(v)]
        lo_x = min(allv) if allv else 0.9
        fig.update_xaxes(tickformat=".0%", range=[max(0.0, lo_x - 0.04), 1.005], showgrid=True, gridcolor=GRID)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(style(fig, 340), key="abc_fill")

    st.markdown("**Look inside one product**")
    j = st.selectbox("Product", range(n), key="sim_product",
                     format_func=lambda k: f"{data.skus[k]} - {data.names[k]}")
    opt_s, base_s = run.optimized, run.baseline
    fig = go.Figure()
    fig.add_trace(go.Bar(x=test_dates, y=opt_s.demand[j], name="Customer demand", marker_color=CONTEXT,
                         hovertemplate="%{y:.0f}<extra>demand</extra>"))
    for sim, key in [(opt_s, "opt"), (base_s, "rule")]:
        label, col = POLICY[key]
        fig.add_trace(go.Scatter(x=test_dates, y=sim.on_hand[j], mode="lines", name=f"Stock - {label.lower()}",
                                 line=dict(color=col, width=2, shape="hv"),
                                 hovertemplate="%{y:.0f}<extra>" + label + "</extra>"))
        so = sim.short[j] > 0
        if so.any():
            fig.add_trace(go.Scatter(x=test_dates[so], y=np.zeros(int(so.sum())), mode="markers",
                                     name=f"Stockout - {label.lower()}",
                                     marker=dict(color=col, size=10, symbol="x-thin", line=dict(width=2, color=col)),
                                     hovertemplate="short %{customdata:.0f}<extra>" + label + "</extra>",
                                     customdata=sim.short[j][so]))
    fig.update_yaxes(title="Units", rangemode="tozero")
    show(style(fig, 360, hover="x unified"), key="sim_chart")
    pp_o, pp_b = opt_s.per_product, base_s.per_product
    st.caption(f"{data.names[j]}: fill rate {pct(pp_o['fill_rate'][j])} (optimised) vs {pct(pp_b['fill_rate'][j])} "
               f"(rule); average stock {money(pp_o['avg_inventory_value'][j], cur)} vs {money(pp_b['avg_inventory_value'][j], cur)}; "
               f"target no-stockout {run.csl[j]:.1%}, achieved {pct(pp_o['achieved_csl'][j])}.")

    with st.expander("Results for every product"):
        res = pd.DataFrame({
            "Code": data.skus, "Product": data.names, "Pattern": cls_pre["pattern"], "Target CSL": run.csl,
            "Achieved CSL": pp_o["achieved_csl"], "Fill (opt)": pp_o["fill_rate"], "Fill (rule)": pp_b["fill_rate"],
            "Avg stock (opt)": pp_o["avg_inventory_value"], "Avg stock (rule)": pp_b["avg_inventory_value"],
            "Total cost (opt)": pp_o["total_cost"], "Total cost (rule)": pp_b["total_cost"],
        })
        res["Cost saving"] = res["Total cost (rule)"] - res["Total cost (opt)"]
        st.dataframe(res, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in ["Target CSL", "Achieved CSL", "Fill (opt)", "Fill (rule)"]
        } | {c: st.column_config.NumberColumn(format="%.0f") for c in
             ["Avg stock (opt)", "Avg stock (rule)", "Total cost (opt)", "Total cost (rule)", "Cost saving"]})
        st.download_button("Download per-product results (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "simulation_results.csv", "text/csv")

    with st.expander("How much does forecast quality matter?"):
        st.caption("The same policy machinery fed by different forecasts. Better forecasts → smaller errors → smaller "
                   "safety stocks for the same service: the money value of forecasting, measured directly.")
        if st.checkbox("Run the comparison (about 10 more simulations)", key="fq_run"):
            fig = go.Figure()
            for m in list(dict.fromkeys([fc.SEASONAL_NAIVE, fc.MOVING_AVERAGE, plan_method])):
                Fm, finm = forecasts_for(m)
                frm = frontier_cached((bt_key, m, pcfg), data.Y, bt, Fm, finm, pcfg, price, cost, margin)
                frm = frm[frm["kind"] == "Optimised policy"]
                fig.add_trace(go.Scatter(x=frm["avg_inventory_value"], y=frm["fill_rate"], mode="lines+markers", name=m,
                                         line=dict(color=METHOD_COLORS[m], width=2),
                                         marker=dict(size=8, line=dict(width=2, color=RING)), customdata=frm["setting"],
                                         hovertemplate="%{customdata}<br>stock %{x:,.0f}<br>fill %{y:.1%}<extra>" + m + "</extra>"))
            fig.update_xaxes(title=f"Average stock held ({cur})", rangemode="tozero")
            fig.update_yaxes(title="Fill rate", tickformat=".0%")
            show(style(fig, 360), key="fq_chart")

# ---------------------------------------------------------------------------
# Tab 5 - purchase plan
# ---------------------------------------------------------------------------
with tab_po:
    plan_day = data.dates[-1] + pd.Timedelta(days=1)
    st.subheader(f"What to order on {plan_day:%A %d %B %Y}")
    st.caption(
        f"Uses the {plan_method} forecast from the last day of data and the error model calibrated on the whole "
        f"backtest. {'Order-up-to level S covers the next order cycle plus lead time (' + str(P) + ' days).' if policy == 'RS' else 'Reorder point s covers the ' + str(L) + '-day lead time; order quantity Q is the EOQ.'}"
        " Enter today's stock below (or upload a stock count) - the starting values are placeholders "
        "(two weeks of average sales).")
    recent = data.Y[:, -28:].mean(axis=1)
    stock0 = pd.DataFrame({"Code": data.skus, "Product": data.names, "On hand": np.round(recent * 14),
                           "On order": np.zeros(n), "Pack size": np.ones(n)})
    up_stock = st.file_uploader("Optional: upload a stock count (CSV/Excel with product code, on hand, on order, pack size)",
                                type=["csv", "txt", "xlsx", "xls"], key="stock_upload")
    if up_stock is not None:
        try:
            sf = ds.read_stock_file(up_stock.getvalue(), up_stock.name).set_index("sku")
            m_ = stock0["Code"].isin(sf.index)
            stock0.loc[m_, "On hand"] = sf.loc[stock0.loc[m_, "Code"], "on_hand"].to_numpy()
            stock0.loc[m_, "On order"] = sf.loc[stock0.loc[m_, "Code"], "on_order"].to_numpy()
            stock0.loc[m_, "Pack size"] = sf.loc[stock0.loc[m_, "Code"], "pack_size"].to_numpy()
            st.success(f"Stock loaded for {int(m_.sum())} of {n} products.")
        except ds.DataError as e:
            st.error(str(e))
    edited = st.data_editor(stock0, key=f"stock_{fp}_{up_stock.name if up_stock else ''}", hide_index=True,
                            width="stretch", height=260, disabled=["Code", "Product"], column_config={
                                "On hand": st.column_config.NumberColumn(min_value=0, step=1, format="%.0f"),
                                "On order": st.column_config.NumberColumn(min_value=0, step=1, format="%.0f"),
                                "Pack size": st.column_config.NumberColumn(min_value=1, step=1, format="%.0f",
                                                                           help="Orders are rounded up to whole packs/cases."),
                            })
    plan = inv.purchase_plan(data.Y, bt, F_plan, fin_plan, pcfg, price, cost, margin,
                             edited["On hand"].fillna(0).to_numpy(float), edited["On order"].fillna(0).to_numpy(float),
                             edited["Pack size"].fillna(1).clip(lower=1).to_numpy(float))
    icon = {"Urgent - stockout likely": "\U0001F534 Urgent", "Order": "\U0001F7E0 Order", "OK": "\U0001F7E2 OK",
            "No sales in 90 days": "⚪ No recent sales"}
    level_col = "Order-up-to S" if policy == "RS" else "Reorder point s"
    out = pd.DataFrame({
        "Status": plan["status"].map(icon), "Code": data.skus, "Product": data.names,
        "Order qty": plan["order_qty"], f"Order value ({cur})": plan["order_value"],
        "Risk of running out before delivery": plan["risk_before_delivery"],
        "On hand + on order": plan["position"], level_col: plan["level"], "Safety stock": plan["safety_stock"],
        "Forecast next 7 days": plan["forecast_7d"], f"Forecast next {P} days": plan["forecast_P"],
        "Days of cover after order": plan["days_cover_after"], "Target CSL": plan["target_csl"],
    })
    if policy == "sQ":
        out.insert(out.columns.get_loc(level_col) + 1, "Order quantity Q", plan["Q"])
    order_rank = {"\U0001F534 Urgent": 0, "\U0001F7E0 Order": 1, "\U0001F7E2 OK": 2, "⚪ No recent sales": 3}
    out = out.assign(_r=out["Status"].map(order_rank)).sort_values(["_r", f"Order value ({cur})"],
                                                                     ascending=[True, False]).drop(columns="_r")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Products to order", int((plan["order_qty"] > 0).sum()), border=True)
    m2.metric("Units to order", f"{plan['order_qty'].sum():,.0f}", border=True)
    m3.metric("Order value (at cost)", money(plan["order_value"].sum(), cur), border=True)
    m4.metric("Urgent (likely to run out first)", int((plan["status"] == "Urgent - stockout likely").sum()), border=True)
    st.dataframe(out, hide_index=True, width="stretch", height=480, column_config={
        "Forecast next 7 days": st.column_config.NumberColumn(format="%.0f"),
        f"Forecast next {P} days": st.column_config.NumberColumn(format="%.0f"),
        "Safety stock": st.column_config.NumberColumn(format="%.0f"),
        "Order-up-to S": st.column_config.NumberColumn(format="%.0f"),
        "Reorder point s": st.column_config.NumberColumn(format="%.0f"),
        "Order quantity Q": st.column_config.NumberColumn(format="%.0f"),
        "On hand + on order": st.column_config.NumberColumn(format="%.0f"),
        "Order qty": st.column_config.NumberColumn(format="%.0f"),
        f"Order value ({cur})": st.column_config.NumberColumn(format="%.0f"),
        "Risk of running out before delivery": st.column_config.NumberColumn(format="percent",
                                                                             help="Chance demand during the lead time exceeds the stock on hand and on order - this order cannot prevent it."),
        "Days of cover after order": st.column_config.NumberColumn(format="%.0f"),
        "Target CSL": st.column_config.NumberColumn(format="percent"),
    })
    po = out[out["Order qty"] > 0]
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download purchase order (CSV)", po.to_csv(index=False).encode("utf-8"),
                       f"purchase_order_{plan_day:%Y%m%d}.csv", "text/csv", type="primary", disabled=po.empty)
    c2.download_button("Download the full plan (CSV)", out.to_csv(index=False).encode("utf-8"),
                       f"stock_plan_{plan_day:%Y%m%d}.csv", "text/csv")
    if not cost_known.any():
        st.caption(f"No unit costs in this data: costs are estimated from selling prices at a {margin_default:.0%} "
                   "gross margin (Costs settings in the sidebar).")

# ---------------------------------------------------------------------------
# Tab 6 - methodology
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(f"""
### The real-world problem

Every shop has to decide, every week, how much of each product to reorder. Order too little and customers leave
empty-handed (and may not come back); order too much and cash sits on shelves, expires or goes out of fashion.
Most small and mid-sized retailers in Sri Lanka and elsewhere decide by habit - *"same as last month"*, *"keep two
weeks of stock"*. That habit is the **rule of thumb** this app compares against. It treats a loaf of bread and a
Christmas cake the same way, ignores holidays, and has no idea how uncertain its own guesses are.

Large retailers use demand-planning systems built on the methods below. This app implements that pipeline end to end,
on any sales history, and - crucially - **tests it on data it has not seen** before claiming anything.

### The pipeline

1. **Clean & understand the history.** Cancellations and returns are netted, non-product lines removed, one-off bulk
   orders optionally capped. Store closures (Saturdays, bank holidays, New Year's Day) are *learned* from the data so a
   closed day is never mistaken for zero demand. Each product is classified by demand pattern (Syntetos-Boylan-Croston
   quadrants) and by ABC-XYZ.
2. **Forecast with eight methods.** Seasonal naive (the benchmark), 28-day moving average, simple exponential smoothing
   (alpha tuned per product), **Croston-SBA** and **TSB** for intermittent demand, **Theta** (SES with drift on
   weekday-adjusted demand), a **global gradient-boosting model** (scikit-learn HistGradientBoosting, Poisson loss,
   trained on all products together with lag, rolling, weekday, holiday and last-year features, predicting demand
   *relative to each product's recent level* so it can follow level shifts trees cannot extrapolate), and an
   **Ensemble** averaging Theta, TSB and the ML model.
3. **Backtest honestly.** Rolling weekly origins over a validation window and a test window; every forecast uses only
   data available on its origin date (the ML model is retrained every {refit_w} weeks on past data only). The test
   suite verifies this by scrambling all future data and checking no forecast changes.
4. **Model the uncertainty.** Demand over the protection period (review cycle + lead time = {P} days here) is modelled
   from the method's *own past errors*: bias-corrected mean, variance proportional to the forecast level
   (over-dispersed Poisson), recent errors weighted more (half-life one quarter), sparse products shrunk toward the
   portfolio. Negative-binomial or weighted-conformal quantiles; the Accuracy tab checks the calibration.
5. **Decide.** Order-up-to / reorder levels are quantiles of that distribution. The service level is either one target
   or **cost-optimal per product**: the newsvendor critical ratio `B / (B + h·R)` (lost sales, periodic review), where
   `B` = margin lost per unit short × penalty and `h` = holding cost per unit per day.
6. **Prove it.** Both policies are replayed day by day over the test window against what customers actually bought -
   deliveries arriving after the lead time, pack-size rounding, lost sales or backorders.

### Measures

- **MASE** (Hyndman & Koehler 2006): mean absolute error ÷ in-sample error of the weekly seasonal naive. < 1 beats it.
- **RMSSE**: the M5 competition's scaled root-mean-square error. **WAPE**: total absolute error ÷ total demand.
- **Protection-period error**: error of the forecast *sum* over the next {P} days - the quantity that sizes safety
  stock. Per-product "Auto" selection uses this criterion rather than daily accuracy.
- **Fill rate**: share of demand served from stock. **Cycle service level (CSL)**: share of order cycles with no stockout.

### Honest limitations

1. **Sales are not demand.** When a product was out of stock, recorded sales understate what customers wanted; the
   app cannot see that censored demand. Real implementations add stockout flags from the POS.
2. **No promotions, prices or weather as inputs.** The demo contains promotions the models are never told about - as in
   most small shops' data. Planned promotions or price changes would need to be fed in as extra features.
3. **Fixed lead times and unlimited supplier capacity.** Lead-time variability, minimum order quantities beyond pack
   rounding, shelf-life/expiry and budget or space limits are not modelled.
4. **Products are planned independently** - no substitution between products, no joint replenishment discounts.
5. **The simulation window is short** (weeks, not years) and from one business at a time; results on the real UK
   wholesaler differ from the demo shop, which is exactly why the app shows them instead of asserting a saving.
6. **With only one year of history, the ML model cannot learn yearly seasonality**; a sudden seasonal surge (the UK
   wholesaler's autumn) is then under-forecast and the stated risk is too low - the calibration chart shows it.

### Data sources

- **UCI Online Retail II** - Chen, D. (2019), UCI Machine Learning Repository, CC BY 4.0,
  [{ds.UCI_DATASET_PAGE}]({ds.UCI_DATASET_PAGE}).
- **Public holidays** - the open-source [`holidays`](https://github.com/vacanza/holidays) package (Sri Lanka, England, ...).
- **Demo shop** - synthetic, generated by `data_sources.py`; prices are illustrative.

### References

Croston (1972) *Operational Research Quarterly* 23 · Syntetos & Boylan (2005) *Int. J. Forecasting* 21 ·
Syntetos, Boylan & Croston (2005) *J. Oper. Res. Soc.* 56 · Teunter, Syntetos & Babai (2011) *EJOR* 214 ·
Assimakopoulos & Nikolopoulos (2000) *Int. J. Forecasting* 16 · Hyndman & Billah (2003) *Int. J. Forecasting* 19 ·
Hyndman & Koehler (2006) *Int. J. Forecasting* 22 · Makridakis, Spiliotis & Assimakopoulos (2022) "M5 accuracy
competition", *Int. J. Forecasting* 38 · Barber, Candès, Ramdas & Tibshirani (2023) "Conformal prediction beyond
exchangeability", *Annals of Statistics* 51 · Silver, Pyke & Thomas (2017) *Inventory and Production Management in
Supply Chains*, 4th ed. · Harris (1913) "How many parts to make at once".
""")

st.divider()
st.caption("Prototype decision-support tool. Next steps: promotions and price as model inputs, stockout-censored "
           "demand estimation, supplier lead-time variability, and budget/shelf-space constraints across products.")
