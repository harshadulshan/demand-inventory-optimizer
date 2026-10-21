"""
calendar_utils.py
Holiday calendars and store-closure logic.

Why this matters: retail demand is dominated by the calendar - the weekly
rhythm, public holidays, festival build-ups (Sinhala & Tamil New Year in
April, Christmas, Vesak dansal) and, just as important, the days the shop
is simply closed. A forecast that does not know tomorrow is a closure will
plan stock for sales that cannot happen; a backtest that treats closed days
as "zero demand" will distort every intermittent-demand method (a closed day
is not a day on which nobody wanted the product).

This module does three things:

1. Looks up public holidays for a country (via the open-source `holidays`
   package, https://github.com/vacanza/holidays), with English names.
2. Detects, from the sales history itself, which weekdays the store is
   structurally closed on and which past days were closures (zero sales
   across the *whole* store on a day that normally trades).
3. Learns which named holidays this particular store closes on (e.g. a UK
   wholesaler shuts on every bank holiday; a Colombo grocery opens on Poya
   days but shuts on New Year's Day) and turns that into a *known-in-advance*
   closure forecast for future dates. Only information available before the
   forecast is made is used, so backtests stay honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:  # optional dependency - the app still works without it, just without holiday features
    import holidays as _holidays_lib
except ImportError:  # pragma: no cover - exercised only when the package is missing
    _holidays_lib = None

# Display name -> (ISO country code, subdivision). Subdivision matters for the
# UK: Easter Monday and the August bank holiday are England-only.
HOLIDAY_CALENDARS: dict[str, Optional[tuple[str, Optional[str]]]] = {
    "None": None,
    "Sri Lanka": ("LK", None),
    "United Kingdom (England)": ("GB", "ENG"),
    "India": ("IN", None),
    "Maldives": ("MV", None),
    "Bangladesh": ("BD", None),
    "Pakistan": ("PK", None),
    "Singapore": ("SG", None),
    "Malaysia": ("MY", None),
    "United Arab Emirates": ("AE", None),
    "Australia": ("AU", None),
    "United States": ("US", None),
    "Canada": ("CA", None),
    "Germany": ("DE", None),
}

# Fallback for the bundled Sri Lankan demo if the `holidays` package is not
# installed: the fixed-date festivals plus Vesak full-moon dates.
_LK_FALLBACK = {
    "Sinhala and Tamil New Year": ["2024-04-13", "2025-04-14", "2026-04-14", "2027-04-14"],
    "Day Before Sinhala and Tamil New Year": ["2024-04-12", "2025-04-13", "2026-04-13", "2027-04-13"],
    "Vesak Full Moon Poya Day": ["2024-05-23", "2025-05-12", "2026-05-01"],
    "Christmas Day": ["2023-12-25", "2024-12-25", "2025-12-25", "2026-12-25", "2027-12-25"],
}

HORIZON_CAP_DAYS = 30  # cap for "days to / since holiday" features


def holidays_available() -> bool:
    return _holidays_lib is not None


def _normalize_name(name: str) -> str:
    name = name.strip()
    for suffix in (" (observed)", " (Observed)", " (estimated)", " (Estimated)"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def holiday_table(calendar: Optional[str], start, end) -> pd.DataFrame:
    """One row per (date, holiday name) between start and end inclusive.

    Several holidays can share a date (e.g. 1 May 2026 is both Workers' Day
    and Vesak in Sri Lanka); the `holidays` package joins those with '; ',
    which are split back into separate rows here.
    """
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "name": pd.Series(dtype=str)})
    spec = HOLIDAY_CALENDARS.get(calendar) if calendar else None
    if spec is None:
        return empty
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    years = list(range(start.year, end.year + 1))
    rows: list[tuple[pd.Timestamp, str]] = []
    if _holidays_lib is not None:
        country, subdiv = spec
        try:
            cal = _holidays_lib.country_holidays(country, subdiv=subdiv, years=years, language="en_US")
        except (NotImplementedError, KeyError, ValueError):
            cal = _holidays_lib.country_holidays(country, subdiv=subdiv, years=years)
        for d, joined in cal.items():
            for part in str(joined).split(";"):
                if part.strip():
                    rows.append((pd.Timestamp(d), _normalize_name(part)))
    elif spec[0] == "LK":
        for name, ds in _LK_FALLBACK.items():
            rows.extend((pd.Timestamp(d), name) for d in ds)
    if not rows:
        return empty
    df = pd.DataFrame(rows, columns=["date", "name"]).drop_duplicates()
    df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)
    return df


def years_without_holidays(calendar: Optional[str], start, end) -> list[int]:
    """Years in [start, end] for which the calendar returns no holidays at all.

    The `holidays` package only knows Sri Lankan holidays once they are
    gazetted (lunar Poya days are announced a year ahead), so forecasting
    into a year it does not cover silently loses holiday information. The
    app surfaces this list as a warning.
    """
    if not calendar or HOLIDAY_CALENDARS.get(calendar) is None:
        return []
    tbl = holiday_table(calendar, start, end)
    have = set(tbl["date"].dt.year.tolist())
    return [y for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1) if y not in have]


# ---------------------------------------------------------------------------
# Store closures
# ---------------------------------------------------------------------------

def detect_closures(dates: pd.DatetimeIndex, store_total: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Infer which observed days the store was closed, and which weekdays it
    never trades on, from total store-wide unit sales per day.

    Rules (deliberately conservative, so genuine zero-sale days of a small
    shop are not mistaken for closures):
      * a weekday is "structurally closed" if >= 90% of its dates have zero
        store-wide sales (and it occurs at least 8 times);
      * any other zero-sales day is a closure only if such days are rare
        (<= 15% of trading days) - otherwise the data is too sparse to tell
        a closure from a quiet day, and nothing is flagged.
    """
    store_total = np.asarray(store_total, dtype=float)
    zero = store_total <= 0
    dow = dates.dayofweek.to_numpy()
    closed_wd = []
    for w in range(7):
        m = dow == w
        if m.sum() >= 8 and zero[m].mean() >= 0.90:
            closed_wd.append(w)
    structural = np.isin(dow, closed_wd)
    other = ~structural
    closed = structural.copy()
    if other.any() and zero[other].mean() <= 0.15:
        closed |= zero & other
    return closed, tuple(closed_wd)


@dataclass
class StoreCalendar:
    """Calendar for the full span: observed history plus the forecast horizon."""

    dates: pd.DatetimeIndex            # observed days followed by `horizon` future days
    n_observed: int
    closed_observed: np.ndarray        # (n_observed,) bool - store actually closed (known after the fact)
    closed_weekdays: tuple[int, ...]
    is_holiday: np.ndarray             # (n_total,) bool
    holiday_names: list[list[str]]     # per date
    days_to_holiday: np.ndarray        # (n_total,) float, capped
    days_since_holiday: np.ndarray     # (n_total,) float, capped
    closure_holidays: set[str] = field(default_factory=set)   # holiday names this store closes on
    seen_holidays: set[str] = field(default_factory=set)      # holiday names observed before the cutoff
    store_closes_unseen_holidays: bool = False                 # prior for holidays never seen before
    rule_cutoff: int = 0               # closure rules learned only from days < rule_cutoff
    calendar_name: Optional[str] = None

    def predicted_closed(self) -> np.ndarray:
        """Closures knowable in advance: structural weekdays plus holidays this
        store has (historically) closed on. (n_total,) bool."""
        dow = self.dates.dayofweek.to_numpy()
        pc = np.isin(dow, self.closed_weekdays)
        for t, names in enumerate(self.holiday_names):
            if not names:
                continue
            if any(nm in self.closure_holidays for nm in names):
                pc[t] = True
            elif self.store_closes_unseen_holidays and not any(nm in self.seen_holidays for nm in names):
                pc[t] = True
        return pc

    def open_factor(self) -> np.ndarray:
        """1.0 on days the store is expected to trade, 0.0 on predicted closures."""
        return (~self.predicted_closed()).astype(float)


def build_store_calendar(
    dates: pd.DatetimeIndex,
    store_total: np.ndarray,
    calendar: Optional[str],
    horizon: int,
    rule_cutoff: Optional[int] = None,
) -> StoreCalendar:
    """Build the store calendar over the observed dates plus `horizon` future days.

    `rule_cutoff`: closure rules (closed weekdays, which holidays the store
    shuts on) are learned only from observed days with index < rule_cutoff.
    The backtest passes its first forecast origin here so that nothing it
    predicts was informed by later data; the final forecast uses all data.
    """
    n_obs = len(dates)
    rule_cutoff = n_obs if rule_cutoff is None else int(max(1, min(rule_cutoff, n_obs)))
    full = pd.date_range(dates[0], periods=n_obs + horizon, freq="D")

    closed_obs, _ = detect_closures(dates, store_total)
    _, closed_wd = detect_closures(dates[:rule_cutoff], np.asarray(store_total)[:rule_cutoff])

    hol = holiday_table(calendar, full[0], full[-1])
    names_per_day: list[list[str]] = [[] for _ in range(len(full))]
    if len(hol):
        pos = full.get_indexer(pd.DatetimeIndex(hol["date"]))
        for p, nm in zip(pos, hol["name"]):
            if p >= 0:
                names_per_day[p].append(nm)
    is_hol = np.array([bool(n) for n in names_per_day])

    # days to next / since last holiday (capped)
    big = float(HORIZON_CAP_DAYS)
    to_h = np.full(len(full), big)
    since_h = np.full(len(full), big)
    nxt = None
    for t in range(len(full) - 1, -1, -1):
        if is_hol[t]:
            nxt = t
        if nxt is not None:
            to_h[t] = min(big, nxt - t)
    last = None
    for t in range(len(full)):
        if is_hol[t]:
            last = t
        if last is not None:
            since_h[t] = min(big, t - last)

    # Which named holidays does this store close on? Learned from days < rule_cutoff.
    occ: dict[str, list[bool]] = {}
    for t in range(min(rule_cutoff, n_obs)):
        for nm in names_per_day[t]:
            occ.setdefault(nm, []).append(bool(closed_obs[t]))
    closure_names = {nm for nm, v in occ.items() if len(v) >= 1 and np.mean(v) >= 2 / 3}
    all_occ = [x for v in occ.values() for x in v]
    closes_unseen = bool(all_occ) and float(np.mean(all_occ)) >= 2 / 3

    return StoreCalendar(
        dates=full, n_observed=n_obs, closed_observed=closed_obs, closed_weekdays=closed_wd,
        is_holiday=is_hol, holiday_names=names_per_day, days_to_holiday=to_h,
        days_since_holiday=since_h, closure_holidays=closure_names, seen_holidays=set(occ.keys()),
        store_closes_unseen_holidays=closes_unseen, rule_cutoff=rule_cutoff, calendar_name=calendar,
    )
