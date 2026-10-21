"""
data_sources.py
Data-access layer: turns sales history from any of three sources into one
`SalesData` structure - a dense product x day matrix of units sold, plus
prices, costs and a store-wide daily total (used to detect closures).

1. **Demo store** - a synthetic but deliberately realistic Colombo
   mini-supermarket (42 products, two years): weekly rhythm, salary-day
   effects, Sinhala & Tamil New Year build-up and closure, Vesak dansal bulk
   buying, Christmas, school-term stationery, monsoon-season umbrellas,
   promotions the models are never told about, a product launch and a
   product being made obsolete (incandescent bulbs). Every number is
   generated - it is labelled synthetic everywhere it is shown.

2. **UCI Online Retail II** (real) - 1,067,371 real transactions from a
   UK-based online gift wholesaler, Dec 2009 - Dec 2011, published by the
   UCI Machine Learning Repository under CC BY 4.0
   (https://archive.ics.uci.edu/dataset/502/online+retail+ii). Downloaded
   once (about 45 MB) and cached in ./data_cache as a compact daily file.
   Cleaning follows the dataset's documented quirks: cancellations
   (invoice "C...") are netted against same-day sales, postage / fees /
   manual adjustments / vouchers are not products, zero-price lines are
   stock write-offs rather than sales.

3. **Your own data** - any CSV or Excel export from a POS, ERP or
   accounting system with at least a date, a product code and a quantity
   (transaction lines or daily totals; returns as negative quantities are
   fine). Day-first dates (dd/mm/yyyy, the Sri Lankan convention) are
   detected automatically.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

from calendar_utils import holiday_table

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "data_cache"

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_ZIP_PATH = CACHE_DIR / "online_retail_ii.zip"
UCI_DAILY_CACHE = CACHE_DIR / "online_retail_ii_daily.csv.gz"
UCI_PRODUCTS_CACHE = CACHE_DIR / "online_retail_ii_products.csv"
UCI_LOG_CACHE = CACHE_DIR / "online_retail_ii_cleaning_log.json"

MIN_HISTORY_DAYS = 140


class DataError(RuntimeError):
    """Raised with a user-facing message when data can't be loaded or used."""


@dataclass
class SalesData:
    Y: np.ndarray                 # (n_products, n_days) units sold per day (float)
    dates: pd.DatetimeIndex       # contiguous daily dates
    skus: list[str]
    names: list[str]
    price: np.ndarray             # (n_products,) average selling price per unit; NaN if unknown
    cost: np.ndarray              # (n_products,) unit cost; NaN if unknown
    store_total: np.ndarray       # (n_days,) units across *all* products (incl. ones not selected)
    source: str
    currency: str
    default_calendar: Optional[str]
    is_synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def n_products(self) -> int:
        return self.Y.shape[0]

    @property
    def n_days(self) -> int:
        return self.Y.shape[1]

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(self.Y).tobytes())
        h.update(self.dates.asi8.tobytes())
        h.update("|".join(self.skus).encode())
        h.update(np.nan_to_num(self.price, nan=-1).tobytes())
        h.update(np.nan_to_num(self.cost, nan=-1).tobytes())
        h.update(np.ascontiguousarray(self.store_total).tobytes())
        return h.hexdigest()[:16]

    def to_long(self) -> pd.DataFrame:
        n, T = self.Y.shape
        return pd.DataFrame({
            "date": np.tile(self.dates.to_numpy(), n),
            "sku": np.repeat(self.skus, T),
            "name": np.repeat(self.names, T),
            "units": self.Y.reshape(-1),
        })


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _matrix_from_long(daily: pd.DataFrame, skus: list[str], dates: pd.DatetimeIndex) -> np.ndarray:
    """daily: columns date, sku, units -> (len(skus), len(dates)) matrix, zeros where no sales."""
    sub = daily[daily["sku"].isin(skus)]
    wide = sub.pivot_table(index="sku", columns="date", values="units", aggfunc="sum", fill_value=0.0)
    wide = wide.reindex(index=skus, columns=dates, fill_value=0.0)
    return wide.to_numpy(dtype=float)


def cleanse_spikes(Y: np.ndarray, k_iqr: float = 10.0, min_nonzero: int = 12) -> tuple[np.ndarray, int, float]:
    """Cap one-off bulk orders, per product, at Q3 + k*IQR of its non-zero days
    (and never below 3x Q3, so steady sellers with a tiny IQR are not clipped).

    Demand planners call this "history cleansing": a single 5,000-unit
    wholesale order is real revenue but is not repeatable demand, and left in
    it inflates every average, variance and safety stock computed from that
    history. Returns (cleaned copy, number of days capped, units removed).
    """
    Yc = Y.copy()
    n_capped, removed = 0, 0.0
    for i in range(Y.shape[0]):
        v = Y[i][Y[i] > 0]
        if len(v) < min_nonzero:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        cap = max(q3 + k_iqr * (q3 - q1), 3.0 * q3, 1.0)
        over = Yc[i] > cap
        if over.any():
            n_capped += int(over.sum())
            removed += float((Yc[i][over] - cap).sum())
            Yc[i][over] = np.floor(cap)
    return Yc, n_capped, removed


def _select_top(daily: pd.DataFrame, top_n: int, by: str) -> list[str]:
    tot = daily.groupby("sku")[by].sum().sort_values(ascending=False)
    tot = tot[tot > 0]
    return tot.index[:top_n].astype(str).tolist()


# ---------------------------------------------------------------------------
# 1. Demo store (synthetic)
# ---------------------------------------------------------------------------

# (sku, name, price LKR, cost LKR, kind, base daily units, modifiers)
# kind: "nb" = everyday seller (negative binomial noise, k = dispersion),
#       "int" = intermittent (p = chance of any sale on a day, s = extra units per sale),
#       "lumpy" = rare, large bulk purchases.
_DEMO_PRODUCTS: list[dict] = [
    # staples
    dict(sku="RICE-SMB-5", name="Samba rice 5kg", price=1450, cost=1330, kind="nb", base=9, k=18, mods=["payday", "avurudu", "december"]),
    dict(sku="RICE-NAD-5", name="Nadu rice 5kg", price=1250, cost=1150, kind="nb", base=11, k=18, mods=["payday", "avurudu", "december"]),
    dict(sku="RICE-KRS-5", name="Keeri samba rice 5kg", price=1800, cost=1640, kind="nb", base=4, k=6, mods=["payday", "avurudu_festive"]),
    dict(sku="DHAL-RED-1", name="Red dhal 1kg", price=320, cost=285, kind="nb", base=18, k=20, mods=["payday"]),
    dict(sku="SUGAR-1", name="White sugar 1kg", price=270, cost=245, kind="nb", base=22, k=20, mods=["payday", "avurudu", "december"]),
    dict(sku="FLOUR-1", name="Wheat flour 1kg", price=230, cost=205, kind="nb", base=16, k=15, mods=["payday", "avurudu_festive"]),
    dict(sku="COCO-OIL-1L", name="Coconut oil 1L", price=880, cost=790, kind="nb", base=10, k=12, mods=["payday", "avurudu_festive"]),
    dict(sku="MILKPWD-400", name="Full-cream milk powder 400g", price=1150, cost=1040, kind="nb", base=14, k=15, mods=["payday", "trend:-0.06"]),
    dict(sku="TEA-200", name="Ceylon black tea 200g", price=480, cost=410, kind="nb", base=9, k=12, mods=["payday"]),
    dict(sku="SALT-1", name="Table salt 1kg", price=150, cost=120, kind="nb", base=6, k=10, mods=[]),
    dict(sku="CHILLI-100", name="Chilli powder 100g", price=260, cost=215, kind="nb", base=7, k=8, mods=["payday"]),
    dict(sku="CURRY-100", name="Roasted curry powder 100g", price=240, cost=195, kind="nb", base=6, k=8, mods=["payday"]),
    dict(sku="ONION-RED-1", name="Red onions 1kg", price=520, cost=450, kind="nb", base=12, k=4, mods=["payday", "avurudu"]),
    dict(sku="POTATO-1", name="Potatoes 1kg", price=380, cost=330, kind="nb", base=10, k=6, mods=["payday"]),
    # fresh / daily
    dict(sku="BREAD-450", name="Sandwich bread 450g", price=180, cost=160, kind="nb", base=45, k=30, mods=["december"]),
    dict(sku="EGGS-10", name="Eggs (pack of 10)", price=420, cost=380, kind="nb", base=20, k=20, mods=["avurudu_festive", "december"]),
    dict(sku="YOGHURT-80", name="Set yoghurt 80g", price=90, cost=72, kind="nb", base=30, k=12, mods=[]),
    dict(sku="FMILK-1L", name="Fresh milk 1L", price=560, cost=500, kind="nb", base=8, k=10, mods=["payday"]),
    # packaged goods / household
    dict(sku="BISC-CRK-190", name="Cream cracker biscuits 190g", price=230, cost=190, kind="nb", base=18, k=8, mods=["promo", "vesak", "avurudu"]),
    dict(sku="BISC-CHC-100", name="Chocolate cream biscuits 100g", price=160, cost=128, kind="nb", base=14, k=6, mods=["promo", "vesak", "december"]),
    dict(sku="NOODLE-5PK", name="Instant noodles (5-pack)", price=640, cost=550, kind="nb", base=8, k=6, mods=["promo", "trend:0.12"]),
    dict(sku="SARDINE-425", name="Canned sardines 425g", price=520, cost=450, kind="nb", base=7, k=6, mods=["promo", "payday"]),
    dict(sku="SOYA-90", name="Soya meat 90g", price=140, cost=115, kind="nb", base=12, k=8, mods=["promo"]),
    dict(sku="SOAP-100", name="Bath soap bar 100g", price=160, cost=125, kind="nb", base=10, k=8, mods=["promo", "payday"]),
    dict(sku="TPASTE-120", name="Toothpaste 120g", price=330, cost=265, kind="nb", base=5, k=5, mods=["promo", "payday"]),
    dict(sku="DETERG-1", name="Washing powder 1kg", price=760, cost=640, kind="nb", base=5, k=5, mods=["promo", "payday"]),
    dict(sku="WATER-1.5L", name="Bottled water 1.5L", price=150, cost=110, kind="nb", base=24, k=8, mods=["trend:0.25", "vesak"]),
    dict(sku="SODA-1.5L", name="Soft drink 1.5L", price=450, cost=370, kind="nb", base=9, k=5, mods=["promo", "vesak", "avurudu", "december"]),
    dict(sku="MARGRN-250", name="Margarine 250g", price=380, cost=320, kind="nb", base=6, k=6, mods=["avurudu_festive"]),
    dict(sku="KITHUL-500", name="Kithul jaggery 500g", price=650, cost=540, kind="nb", base=3, k=3, mods=["avurudu_festive"]),
    dict(sku="MOSQ-COIL", name="Mosquito coils (box of 10)", price=220, cost=170, kind="nb", base=4, k=5, mods=["rainy"]),
    # slow / intermittent
    dict(sku="SCRAPER", name="Coconut scraper (hand)", price=1200, cost=850, kind="int", p=0.08, s=0.1, mods=["avurudu"]),
    dict(sku="LED-9W", name="LED bulb 9W", price=650, cost=480, kind="int", p=0.28, s=0.4, mods=[]),
    dict(sku="BULB-60W", name="Incandescent bulb 60W", price=220, cost=160, kind="int", p=0.40, s=0.5, mods=["decline:240"]),
    dict(sku="BATT-AA4", name="AA batteries (4-pack)", price=480, cost=360, kind="int", p=0.30, s=0.8, mods=["december"]),
    dict(sku="UMBRELLA", name="Foldable umbrella", price=1650, cost=1150, kind="int", p=0.05, s=0.3, mods=["rainy_strong"]),
    # lumpy (bulk buys for almsgivings, weddings, dansal)
    dict(sku="PLATES-50", name="Paper plates (pack of 50)", price=650, cost=470, kind="lumpy", p=0.10, m=6, k=0.8, mods=["vesak_strong"]),
    dict(sku="BASMATI-5", name="Basmati rice 5kg", price=3400, cost=3050, kind="lumpy", p=0.12, m=3, k=0.7, mods=["december"]),
    # seasonal
    dict(sku="SWEETS-AVR", name="Avurudu sweets pack (kavum, kokis)", price=950, cost=650, kind="nb", base=0.25, k=2, mods=["avurudu_sweets"]),
    dict(sku="XMAS-CAKE-1", name="Christmas cake 1kg", price=2800, cost=2000, kind="nb", base=0.03, k=2, mods=["xmas_item"]),
    dict(sku="BOOKS-6PK", name="Exercise books (pack of 6)", price=780, cost=580, kind="nb", base=1.5, k=3, mods=["school"]),
    # new product
    dict(sku="OATMILK-1L", name="Oat milk 1L", price=1250, cost=980, kind="nb", base=6, k=6, mods=["launch:200"]),
]

_WEEKDAY_PROFILE = np.array([0.92, 0.88, 0.92, 0.95, 1.05, 1.18, 1.10])  # Mon..Sun, Colombo grocery
_WEEKDAY_PROFILE = _WEEKDAY_PROFILE / _WEEKDAY_PROFILE.mean()


def _event_dates(dates: pd.DatetimeIndex, name: str) -> list[pd.Timestamp]:
    tbl = holiday_table("Sri Lanka", dates[0] - pd.Timedelta(days=40), dates[-1] + pd.Timedelta(days=40))
    return sorted(pd.Timestamp(d) for d in tbl.loc[tbl["name"] == name, "date"].unique())


def _window_multiplier(dates: pd.DatetimeIndex, anchors: list[pd.Timestamp], spec: dict[tuple[int, int], float]) -> np.ndarray:
    """spec maps (start_offset, end_offset) inclusive day ranges relative to each anchor -> multiplier."""
    m = np.ones(len(dates))
    for a in anchors:
        for (lo, hi), mult in spec.items():
            sel = (dates >= a + pd.Timedelta(days=lo)) & (dates <= a + pd.Timedelta(days=hi))
            m[sel] *= mult
    return m


def generate_demo_data(end_date=None, n_days: int = 730, seed: int = 42) -> SalesData:
    """Generate the synthetic Colombo mini-supermarket. Deterministic for a given (end_date, seed)."""
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1) if end_date is None else pd.Timestamp(end_date).normalize()
    dates = pd.date_range(end - pd.Timedelta(days=n_days - 1), end, freq="D")
    rng = np.random.default_rng(seed)
    T = len(dates)
    t = np.arange(T)
    dow = dates.dayofweek.to_numpy()
    dom = dates.day.to_numpy()
    month = dates.month.to_numpy()

    payday = np.where(dom >= 25, 1.12, np.where(dom <= 4, 1.08, np.where((dom >= 10) & (dom <= 20), 0.95, 1.0)))

    new_year = _event_dates(dates, "Sinhala and Tamil New Year")
    vesak = _event_dates(dates, "Vesak Full Moon Poya Day")
    xmas = [pd.Timestamp(y, 12, 25) for y in range(dates[0].year - 1, dates[-1].year + 2)]

    avurudu = _window_multiplier(dates, new_year, {(-12, -4): 1.15, (-3, -1): 1.6, (1, 1): 0.7, (2, 5): 0.9})
    avurudu_festive = _window_multiplier(dates, new_year, {(-12, -4): 1.6, (-3, -1): 3.2, (1, 1): 0.5})
    avurudu_sweets = _window_multiplier(dates, new_year, {(-14, -8): 12.0, (-7, -1): 40.0, (1, 3): 6.0})
    vesak_m = _window_multiplier(dates, vesak, {(-2, 0): 2.2})
    vesak_strong = _window_multiplier(dates, vesak, {(-3, 0): 6.0})
    december = _window_multiplier(dates, xmas, {(-10, -1): 1.15, (0, 0): 0.6, (6, 6): 1.2})
    xmas_item = _window_multiplier(dates, xmas, {(-24, -11): 60.0, (-10, -1): 220.0, (0, 6): 25.0})
    school = np.where(((month == 12) & (dom >= 10)) | ((month == 1) & (dom <= 25)), 6.0, 1.0)
    rainy = np.where(np.isin(month, [4, 5, 6, 10, 11]), 1.6, 1.0)
    rainy_strong = np.where(np.isin(month, [5, 6, 10, 11]), 5.0, np.where(month == 4, 2.5, 1.0))

    # The shop closes on Sinhala & Tamil New Year's Day itself.
    closed = np.zeros(T, dtype=bool)
    for a in new_year:
        closed |= dates == a

    n = len(_DEMO_PRODUCTS)
    Y = np.zeros((n, T))
    for i, spec in enumerate(_DEMO_PRODUCTS):
        wd = _WEEKDAY_PROFILE * rng.uniform(0.95, 1.05, 7)
        wd = wd / wd.mean()
        mult = wd[dow].copy()
        mods = spec["mods"]
        for mod in mods:
            if mod == "payday":
                mult *= payday
            elif mod == "avurudu":
                mult *= avurudu
            elif mod == "avurudu_festive":
                mult *= avurudu_festive
            elif mod == "avurudu_sweets":
                mult *= avurudu_sweets
            elif mod == "vesak":
                mult *= vesak_m
            elif mod == "vesak_strong":
                mult *= vesak_strong
            elif mod == "december":
                mult *= december
            elif mod == "xmas_item":
                mult *= xmas_item
            elif mod == "school":
                mult *= school
            elif mod == "rainy":
                mult *= rainy
            elif mod == "rainy_strong":
                mult *= rainy_strong
            elif mod.startswith("trend:"):
                g = float(mod.split(":")[1])
                mult *= (1 + g) ** (t / 365.0)
            elif mod.startswith("decline:"):
                tau = float(mod.split(":")[1])
                mult *= np.exp(-t / tau)
            elif mod.startswith("launch:"):
                days_before_end = int(mod.split(":")[1])
                launch = T - days_before_end
                ramp = np.clip((t - launch) / 60.0, 0, 1) * 0.7 + 0.3
                mult *= np.where(t < launch, 0.0, ramp)
            elif mod == "promo":
                n_promos = rng.poisson(3.0 * T / 365)
                for _ in range(n_promos):
                    s0 = int(rng.integers(0, max(1, T - 7)))
                    ln = int(rng.integers(4, 8))
                    mult[s0:s0 + ln] *= rng.uniform(1.6, 2.2)
        if spec["kind"] == "nb":
            mu = spec["base"] * mult
            k = spec["k"]
            y = rng.negative_binomial(k, k / (k + np.maximum(mu, 1e-9)))
        elif spec["kind"] == "int":
            p = np.clip(spec["p"] * mult, 0, 0.95)
            occur = rng.random(T) < p
            y = occur * (1 + rng.poisson(spec["s"], T))
        else:  # lumpy
            p = np.clip(spec["p"] * mult, 0, 0.95)
            occur = rng.random(T) < p
            m, k = spec["m"], spec["k"]
            size = 1 + rng.negative_binomial(k, k / (k + (m - 1)), T)
            y = occur * size
        y = np.where(closed, 0, y).astype(float)
        Y[i] = y

    price = np.array([p["price"] for p in _DEMO_PRODUCTS], dtype=float)
    cost = np.array([p["cost"] for p in _DEMO_PRODUCTS], dtype=float)
    return SalesData(
        Y=Y, dates=dates, skus=[p["sku"] for p in _DEMO_PRODUCTS], names=[p["name"] for p in _DEMO_PRODUCTS],
        price=price, cost=cost, store_total=Y.sum(axis=0), source="Demo: Colombo mini-supermarket (synthetic)",
        currency="LKR", default_calendar="Sri Lanka", is_synthetic=True,
        notes=[
            "Synthetic demo data generated by this app - not a real shop's records. Prices are illustrative.",
            "Built-in effects the models are NOT told about: promotions, payday cycles, monsoon months. "
            "Known in advance (via the Sri Lanka holiday calendar): New Year, Vesak, Christmas.",
        ],
    )


# ---------------------------------------------------------------------------
# 2. UCI Online Retail II (real data)
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "invoice": "invoice", "invoiceno": "invoice", "stockcode": "sku", "description": "description",
    "quantity": "quantity", "invoicedate": "date", "price": "price", "unitprice": "price",
    "customerid": "customer", "country": "country",
}


def uci_cache_ready() -> bool:
    return UCI_DAILY_CACHE.exists() and UCI_PRODUCTS_CACHE.exists()


def download_uci_online_retail(progress: Optional[Callable[[float, str], None]] = None, timeout: int = 60) -> Path:
    """Download the official zip from UCI into ./data_cache (about 45 MB, once)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UCI_ZIP_PATH.with_suffix(".part")
    try:
        with requests.get(UCI_ZIP_URL, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                raise DataError(f"UCI download failed (HTTP {resp.status_code}). You can download it manually from "
                                f"{UCI_DATASET_PAGE} and place the zip at {UCI_ZIP_PATH}.")
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        frac = done / total if total else 0.0
                        progress(min(frac, 1.0), f"Downloading UCI Online Retail II... {done / 1e6:.1f} MB")
    except requests.RequestException as e:
        raise DataError(f"Could not reach the UCI repository: {e}. Check your internet connection, or download "
                        f"the zip manually from {UCI_DATASET_PAGE} and place it at {UCI_ZIP_PATH}.") from e
    tmp.replace(UCI_ZIP_PATH)
    return UCI_ZIP_PATH


def _read_workbook(source) -> pd.DataFrame:
    """Read every sheet of the Online Retail II workbook (zip or xlsx path, or bytes)."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        raw_bytes = p.read_bytes()
        is_zip = zipfile.is_zipfile(p)
    else:
        raw_bytes = bytes(source)
        is_zip = zipfile.is_zipfile(io.BytesIO(raw_bytes))
    if is_zip:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".xlsx", ".xls", ".csv"))]
            if not members:
                raise DataError("The UCI zip did not contain an Excel or CSV file.")
            name = members[0]
            raw_bytes = zf.read(name)
            if name.lower().endswith(".csv"):
                return pd.read_csv(io.BytesIO(raw_bytes), dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str})
    import importlib.util
    # python-calamine (optional) reads this 1M-row workbook about 10x faster than openpyxl
    engines = (["calamine"] if importlib.util.find_spec("python_calamine") else []) + ["openpyxl"]
    last_err: Optional[Exception] = None
    for eng in engines:
        try:
            sheets = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, engine=eng,
                                   dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str})
            return pd.concat(list(sheets.values()), ignore_index=True)
        except ImportError as e:
            last_err = e
            continue
    raise DataError(f"Could not read the workbook ({last_err}). Install openpyxl: pip install openpyxl")


def clean_retail_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Clean Online-Retail-style transaction lines into daily product totals.

    Returns (daily: date, sku, units, revenue), (products: sku, description), log lines.
    """
    rename = {}
    for c in raw.columns:
        key = re.sub(r"[^a-z]", "", str(c).lower())
        if key in _COLUMN_ALIASES:
            rename[c] = _COLUMN_ALIASES[key]
    df = raw.rename(columns=rename)
    missing = {"invoice", "sku", "quantity", "date", "price"} - set(df.columns)
    if missing:
        raise DataError(f"Online Retail file is missing expected columns: {sorted(missing)}")
    log: list[str] = []
    n0 = len(df)
    df = df.dropna(subset=["sku", "quantity", "date"]).copy()
    df["invoice"] = df["invoice"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["quantity", "price", "date"])

    is_product = df["sku"].str.fullmatch(r"\d{5}[A-Z]{0,2}")
    log.append(f"{n0:,} raw transaction lines read.")
    log.append(f"Removed {int((~is_product).sum()):,} non-product lines (postage, carriage, bank/Amazon fees, "
               f"manual adjustments, samples, gift vouchers).")
    df = df[is_product]
    adj = df["invoice"].str.startswith("A")
    if adj.any():
        log.append(f"Removed {int(adj.sum()):,} bad-debt adjustment lines (invoice 'A...').")
        df = df[~adj]
    nonpos = df["price"] <= 0
    log.append(f"Removed {int(nonpos.sum()):,} zero-price lines (stock write-offs / corrections, not sales).")
    df = df[~nonpos].copy()
    cancel = df["invoice"].str.startswith("C")
    df.loc[cancel, "quantity"] = -df.loc[cancel, "quantity"].abs()
    log.append(f"Netted {int(cancel.sum()):,} cancellation lines (invoice 'C...') against same-day sales.")

    df["day"] = df["date"].dt.normalize()
    df["revenue"] = df["quantity"] * df["price"]
    daily = (df.groupby(["day", "sku"], as_index=False)
               .agg(units=("quantity", "sum"), revenue=("revenue", "sum"))
               .rename(columns={"day": "date"}))
    neg = daily["units"] < 0
    if neg.any():
        log.append(f"{int(neg.sum()):,} product-days were net negative after cancellations (cancelling an earlier "
                   f"day's order) and were set to zero.")
    daily["units"] = daily["units"].clip(lower=0)
    daily["revenue"] = daily["revenue"].where(daily["units"] > 0, 0.0).clip(lower=0)
    daily = daily[daily["units"] > 0].reset_index(drop=True)

    desc = df.loc[~cancel & df["description"].notna(), ["sku", "description"]].copy()
    desc["description"] = desc["description"].astype(str).str.strip()
    products = (desc.groupby(["sku", "description"]).size().reset_index(name="n")
                    .sort_values(["sku", "n"], ascending=[True, False]).drop_duplicates("sku")[["sku", "description"]])
    log.append(f"Result: {daily['sku'].nunique():,} products, {daily['date'].nunique():,} trading days, "
               f"{daily['date'].min():%d %b %Y} to {daily['date'].max():%d %b %Y}.")
    return daily, products.reset_index(drop=True), log


def build_uci_cache(source=None, progress: Optional[Callable[[float, str], None]] = None) -> None:
    """Read the downloaded workbook once, clean it, and save a compact daily cache."""
    source = UCI_ZIP_PATH if source is None else source
    if progress:
        progress(0.0, "Reading the workbook (1M rows - takes a minute or two the first time)...")
    raw = _read_workbook(source)
    if progress:
        progress(0.7, "Cleaning transactions...")
    daily, products, log = clean_retail_transactions(raw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = daily.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(UCI_DAILY_CACHE, index=False, compression="gzip")
    products.to_csv(UCI_PRODUCTS_CACHE, index=False)
    UCI_LOG_CACHE.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if progress:
        progress(1.0, "Cached.")


def load_uci_online_retail(top_n: int = 40, cleanse: bool = True) -> SalesData:
    """Load the cleaned daily cache and keep the `top_n` products by revenue."""
    if not uci_cache_ready():
        raise DataError("UCI Online Retail II has not been downloaded yet.")
    daily = pd.read_csv(UCI_DAILY_CACHE, dtype={"sku": str}, parse_dates=["date"])
    products = pd.read_csv(UCI_PRODUCTS_CACHE, dtype={"sku": str})
    log = json.loads(UCI_LOG_CACHE.read_text(encoding="utf-8")) if UCI_LOG_CACHE.exists() else []
    return _sales_data_from_daily(
        daily, products, top_n=top_n, cleanse=cleanse, rank_by="revenue",
        source="UCI Online Retail II (real UK online wholesaler, 2009-2011)", currency="GBP",
        default_calendar="United Kingdom (England)", notes=log, cost_col=None,
    )


def _sales_data_from_daily(
    daily: pd.DataFrame, products: Optional[pd.DataFrame], top_n: int, cleanse: bool, rank_by: str,
    source: str, currency: str, default_calendar: Optional[str], notes: list[str], cost_col: Optional[str],
) -> SalesData:
    daily = daily.copy()
    daily["sku"] = daily["sku"].astype(str)
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    if len(dates) < MIN_HISTORY_DAYS:
        raise DataError(f"Only {len(dates)} days of history - at least {MIN_HISTORY_DAYS} are needed "
                        f"(a year or more gives far better forecasts).")
    store_total = daily.groupby("date")["units"].sum().reindex(dates, fill_value=0.0).to_numpy(dtype=float)
    by = rank_by if rank_by in daily.columns and daily[rank_by].sum() > 0 else "units"
    skus = _select_top(daily, top_n, by)
    if not skus:
        raise DataError("No product has any positive sales.")
    Y = _matrix_from_long(daily, skus, dates)
    notes = list(notes)
    if cleanse:
        Y, n_cap, removed = cleanse_spikes(Y)
        if n_cap:
            notes.append(f"History cleansing: capped {n_cap} extreme one-off product-days "
                         f"({removed:,.0f} units, {removed / max(Y.sum() + removed, 1):.1%} of volume) at "
                         f"Q3 + 10xIQR of each product's selling days (bulk orders usually handled as special orders).")
    sub = daily[daily["sku"].isin(skus)].groupby("sku").agg(units=("units", "sum"),
                                                              revenue=("revenue", "sum") if "revenue" in daily else ("units", "sum"))
    price = (sub["revenue"] / sub["units"]).reindex(skus).to_numpy(dtype=float) if "revenue" in daily else np.full(len(skus), np.nan)
    if "revenue" in daily and daily["revenue"].sum() <= 0:
        price = np.full(len(skus), np.nan)
    if cost_col and cost_col in daily.columns:
        cs = daily[daily["sku"].isin(skus)].groupby("sku").agg(u=("units", "sum"), c=(cost_col, "sum"))
        cost = (cs["c"] / cs["u"]).reindex(skus).to_numpy(dtype=float)
        cost = np.where(cost > 0, cost, np.nan)
    else:
        cost = np.full(len(skus), np.nan)
    if products is not None and len(products):
        name_map = dict(zip(products["sku"].astype(str), products["description"].astype(str)))
    else:
        name_map = {}
    names = [name_map.get(s, s) or s for s in skus]
    return SalesData(Y=Y, dates=dates, skus=skus, names=names, price=price, cost=cost, store_total=store_total,
                     source=source, currency=currency, default_calendar=default_calendar, notes=notes)


# ---------------------------------------------------------------------------
# 3. Your own file
# ---------------------------------------------------------------------------

ROLE_HINTS: dict[str, list[str]] = {
    "date": ["date", "invoicedate", "orderdate", "transactiondate", "salesdate", "billdate", "txndate",
             "datetime", "day", "createdat", "timestamp"],
    "sku": ["sku", "stockcode", "itemcode", "productcode", "itemid", "productid", "itemno", "barcode",
            "article", "code", "item", "product"],
    "quantity": ["quantity", "qty", "units", "unitssold", "qtysold", "salesqty", "soldqty", "quantitysold", "sold",
                 "volume", "count"],
    "price": ["unitprice", "sellingprice", "saleprice", "price", "rate", "mrp", "retailprice"],
    "cost": ["unitcost", "costprice", "purchaseprice", "buyingprice", "cost", "cogs"],
    "description": ["description", "productname", "itemname", "itemdescription", "productdescription", "name", "title"],
}
ROLES = ["date", "sku", "quantity", "price", "cost", "description"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def guess_columns(columns: list[str]) -> dict[str, Optional[str]]:
    """Best-guess mapping from role -> column name, using common POS/ERP header names."""
    normed = {c: _norm(c) for c in columns}
    used: set[str] = set()
    out: dict[str, Optional[str]] = {}
    for role in ROLES:
        found = None
        for hint in ROLE_HINTS[role]:           # exact matches first
            for c, nc in normed.items():
                if c not in used and nc == hint:
                    found = c
                    break
            if found:
                break
        if not found:
            for hint in ROLE_HINTS[role]:       # then "contains"
                for c, nc in normed.items():
                    if c not in used and hint in nc and len(hint) >= 3:
                        found = c
                        break
                if found:
                    break
        out[role] = found
        if found:
            used.add(found)
    return out


def read_uploaded_table(data: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV / Excel file into a DataFrame (all columns as read)."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(data), sheet_name=0)
        except ImportError as e:
            raise DataError("Reading Excel files needs openpyxl: pip install openpyxl") from e
    last: Optional[Exception] = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            if df.shape[1] >= 2:
                return df
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as e:
            last = e
    raise DataError(f"Could not read '{filename}' as CSV ({last}). Save it as CSV (comma or semicolon separated) or .xlsx.")


_DATE_RE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")


def guess_dayfirst(values: pd.Series) -> bool:
    """True if dates look like dd/mm/yyyy (any first part > 12), False if mm/dd/yyyy
    (any second part > 12). Ambiguous -> True, the Sri Lankan / UK convention."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return True
    firsts, seconds = [], []
    for v in values.dropna().astype(str).head(5000):
        m = _DATE_RE.match(v)
        if m:
            firsts.append(int(m.group(1)))
            seconds.append(int(m.group(2)))
    if any(f > 12 for f in firsts):
        return True
    if any(s > 12 for s in seconds):
        return False
    return True


def _parse_dates(s: pd.Series, dayfirst: bool) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else pd.to_datetime(s)
    txt = s.astype(str).str.strip()
    parsed = pd.to_datetime(txt, dayfirst=dayfirst, errors="coerce")
    if parsed.isna().mean() > 0.02:
        parsed = pd.to_datetime(txt, dayfirst=dayfirst, errors="coerce", format="mixed")
    return parsed


def sales_data_from_table(
    df: pd.DataFrame, mapping: dict[str, Optional[str]], dayfirst: bool, top_n: int = 50,
    cleanse: bool = False, currency: str = "LKR", default_calendar: Optional[str] = "Sri Lanka",
    filename: str = "uploaded file",
) -> SalesData:
    """Build SalesData from an uploaded transaction or daily-sales table."""
    for role in ("date", "sku", "quantity"):
        if not mapping.get(role):
            raise DataError(f"Please choose which column holds the {role}.")
    notes: list[str] = []
    work = pd.DataFrame({
        "date": _parse_dates(df[mapping["date"]], dayfirst),
        "sku": df[mapping["sku"]].astype(str).str.strip(),
        "units": pd.to_numeric(df[mapping["quantity"]], errors="coerce"),
    })
    if mapping.get("price"):
        work["price"] = pd.to_numeric(df[mapping["price"]], errors="coerce")
    if mapping.get("cost"):
        work["cost"] = pd.to_numeric(df[mapping["cost"]], errors="coerce")
    n0 = len(work)
    bad = work["date"].isna() | work["units"].isna() | (work["sku"] == "") | (work["sku"].str.lower() == "nan")
    if bad.mean() > 0.5:
        raise DataError(f"More than half the rows of {filename} have an unreadable date, quantity or product code - "
                        f"check the column choices and the day-first setting.")
    if bad.any():
        notes.append(f"Skipped {int(bad.sum()):,} of {n0:,} rows with an unreadable date, quantity or product code.")
    work = work[~bad].copy()
    work["date"] = work["date"].dt.normalize()
    if "price" in work:
        work["revenue"] = work["units"] * work["price"].fillna(0)
    if "cost" in work:
        work["cost_total"] = work["units"] * work["cost"].fillna(0)
    agg = {"units": ("units", "sum")}
    if "revenue" in work:
        agg["revenue"] = ("revenue", "sum")
    if "cost_total" in work:
        agg["cost_total"] = ("cost_total", "sum")
    daily = work.groupby(["date", "sku"], as_index=False).agg(**agg)
    neg = daily["units"] < 0
    if neg.any():
        notes.append(f"{int(neg.sum()):,} product-days had net negative quantity (returns exceeding sales) and were set to zero.")
    pos = daily["units"] > 0
    for c in [c for c in ("revenue", "cost_total") if c in daily]:
        daily[c] = daily[c].where(pos, 0.0)
    daily["units"] = daily["units"].clip(lower=0)
    daily = daily[daily["units"] > 0]
    if daily.empty:
        raise DataError("No positive sales found in the file.")
    products = None
    if mapping.get("description"):
        d = pd.DataFrame({"sku": df[mapping["sku"]].astype(str).str.strip(),
                          "description": df[mapping["description"]].astype(str).str.strip()})
        products = (d.groupby(["sku", "description"]).size().reset_index(name="n")
                      .sort_values(["sku", "n"], ascending=[True, False]).drop_duplicates("sku")[["sku", "description"]])
    notes.insert(0, f"Loaded {n0:,} rows from {filename}: {daily['sku'].nunique():,} products, "
                    f"{daily['date'].min():%d %b %Y} to {daily['date'].max():%d %b %Y}.")
    return _sales_data_from_daily(
        daily, products, top_n=top_n, cleanse=cleanse, rank_by="revenue" if "revenue" in daily else "units",
        source=f"Your data: {filename}", currency=currency, default_calendar=default_calendar, notes=notes,
        cost_col="cost_total" if "cost_total" in daily else None,
    )


def read_stock_file(data: bytes, filename: str) -> pd.DataFrame:
    """Read a current-stock file: product code + on hand (+ optional on order, pack size)."""
    df = read_uploaded_table(data, filename)
    cols = {c: _norm(c) for c in df.columns}

    def pick(hints):
        for h in hints:
            for c, nc in cols.items():
                if nc == h:
                    return c
        for h in hints:
            for c, nc in cols.items():
                if h in nc:
                    return c
        return None

    sku_c = pick(["sku", "stockcode", "itemcode", "productcode", "code", "item", "product"])
    oh_c = pick(["onhand", "stock", "qtyonhand", "instock", "available", "quantity", "qty"])
    if not sku_c or not oh_c:
        raise DataError("Stock file needs a product-code column and an on-hand quantity column.")
    oo_c = pick(["onorder", "ordered", "intransit", "pending"])
    pk_c = pick(["packsize", "casesize", "pack", "moq", "case"])
    out = pd.DataFrame({"sku": df[sku_c].astype(str).str.strip(),
                        "on_hand": pd.to_numeric(df[oh_c], errors="coerce").fillna(0)})
    out["on_order"] = pd.to_numeric(df[oo_c], errors="coerce").fillna(0) if oo_c else 0.0
    out["pack_size"] = pd.to_numeric(df[pk_c], errors="coerce").fillna(1).clip(lower=1) if pk_c else 1.0
    return out.groupby("sku", as_index=False).agg(on_hand=("on_hand", "sum"), on_order=("on_order", "sum"),
                                                   pack_size=("pack_size", "max"))
