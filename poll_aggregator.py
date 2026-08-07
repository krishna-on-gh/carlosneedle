"""
Generic Ballot Poll Aggregator for CarlosNeedle.

Loads polls from data/polls_2026.csv, computes rolling average with
LV (60%) / RV (40%) weighting, exposes a poll-derived national swing
for the engine.

Sign convention: R+, so margin = rep_pct - dem_pct.
  Positive → R lead, Negative → D lead.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

POLLS_CSV = "data/polls_2026.csv"

# 2024 reference vote share (national House). Used to convert
# current poll margin into an "expected_swing" delta.
# From your engine's original notebook: swing_calculate_2024 = 2.15
REFERENCE_2024 = 2.15

# LV vs RV weighting → 60 / 40
WEIGHT_LV = 1.5   # 1.5 / (1.5 + 1.0) = 0.60
WEIGHT_RV = 1.0   # 1.0 / (1.5 + 1.0) = 0.40

# How undecideds break in the projected GCB (this cycle).
# 0.56 to D, 0.44 to R.
UNDECIDED_TO_D = 0.56
UNDECIDED_TO_R = 0.44


# ── Loading ──────────────────────────────────────────────────────────────────
def load_polls(path=POLLS_CSV):
    """Load and normalize the polls CSV."""
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "date", "pollster", "sample_type", "dem_pct", "rep_pct",
            "sample_size", "notes", "margin_r", "weight"
        ])
    if len(df) == 0:
        return df
    df["margin_r"] = df["rep_pct"] - df["dem_pct"]  # R+ convention
    df["sample_type"] = df["sample_type"].astype(str).str.strip().str.upper()
    df["weight"] = df["sample_type"].apply(
        lambda s: WEIGHT_LV if s == "LV" else WEIGHT_RV
    )
    return df.sort_values("date").reset_index(drop=True)


def append_poll(row_dict, path=POLLS_CSV):
    """Append a single poll dict to the CSV."""
    df = load_polls(path)
    # Drop internal computed columns before writing
    keep_cols = ["date", "pollster", "sample_type", "dem_pct", "rep_pct",
                 "sample_size", "notes"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    df = df[keep_cols]
    new_row = pd.DataFrame([{c: row_dict.get(c) for c in keep_cols}])
    combined = pd.concat([df, new_row], ignore_index=True)
    combined.to_csv(path, index=False)


# ── Aggregation ──────────────────────────────────────────────────────────────
def _weighted_recent(polls, as_of_date, window_days, col):
    """Weighted average of `col` over the last window."""
    if len(polls) == 0:
        return None
    cutoff = pd.Timestamp(as_of_date) - timedelta(days=window_days)
    recent = polls[
        (polls["date"] >= cutoff) & (polls["date"] <= pd.Timestamp(as_of_date))
    ]
    if len(recent) == 0:
        return None
    return float(np.average(recent[col], weights=recent["weight"]))


def rolling_average(polls, as_of_date, window_days=21):
    """LV/RV-weighted rolling avg of margin over the last `window_days`."""
    return _weighted_recent(polls, as_of_date, window_days, "margin_r")


def projected_gcb(polls, as_of_date=None, window_days=21,
                  d_share=UNDECIDED_TO_D, r_share=UNDECIDED_TO_R):
    """
    Project the GCB by splitting undecideds `d_share`/`r_share`.

    Returns a dict with:
        avg_dem, avg_rep, undecided,
        proj_dem, proj_rep, proj_margin_r
    """
    if len(polls) == 0:
        return None
    if as_of_date is None:
        as_of_date = polls["date"].max()
    avg_dem = _weighted_recent(polls, as_of_date, window_days, "dem_pct")
    avg_rep = _weighted_recent(polls, as_of_date, window_days, "rep_pct")
    if avg_dem is None or avg_rep is None:
        return None
    undecided = max(0.0, 100.0 - avg_dem - avg_rep)
    proj_dem = avg_dem + d_share * undecided
    proj_rep = avg_rep + r_share * undecided
    proj_margin_r = proj_rep - proj_dem
    return {
        "avg_dem":       avg_dem,
        "avg_rep":       avg_rep,
        "undecided":     undecided,
        "proj_dem":      proj_dem,
        "proj_rep":      proj_rep,
        "proj_margin_r": proj_margin_r,
    }


def projected_swing(polls, reference=REFERENCE_2024, window_days=21,
                    d_share=UNDECIDED_TO_D, r_share=UNDECIDED_TO_R):
    """Poll-derived expected_swing using the *projected* GCB."""
    p = projected_gcb(polls, window_days=window_days,
                      d_share=d_share, r_share=r_share)
    if p is None:
        return None
    return p["proj_margin_r"] - reference


def fever_series(polls, window_days=21):
    """Rolling avg per day for each day in the polls range. Returns DataFrame."""
    if len(polls) == 0:
        return pd.DataFrame(columns=["date", "avg"])
    start = polls["date"].min()
    end   = polls["date"].max()
    days  = pd.date_range(start, end)
    values = [rolling_average(polls, d, window_days) for d in days]
    return pd.DataFrame({"date": days, "avg": values}).dropna()


# ── Model integration ────────────────────────────────────────────────────────
def poll_derived_swing(polls, reference=REFERENCE_2024, window_days=21):
    """
    Convert current poll aggregate into an `expected_swing` value
    (R+ convention). Formula: current_avg - 2024_reference.
    """
    if len(polls) == 0:
        return None
    latest = polls["date"].max()
    avg = rolling_average(polls, latest, window_days)
    if avg is None:
        return None
    return avg - reference


def summary(polls, window_days=21):
    """Compact summary dict for display."""
    if len(polls) == 0:
        return {
            "n_polls": 0, "current_avg": None, "swing": None, "latest_date": None,
            "projected": None, "projected_swing": None,
        }
    latest = polls["date"].max()
    avg = rolling_average(polls, latest, window_days)
    proj = projected_gcb(polls, latest, window_days)
    return {
        "n_polls":         len(polls),
        "current_avg":     avg,
        "swing":           (avg - REFERENCE_2024) if avg is not None else None,
        "latest_date":     latest,
        "window_days":     window_days,
        "projected":       proj,
        "projected_swing": (proj["proj_margin_r"] - REFERENCE_2024) if proj else None,
    }
