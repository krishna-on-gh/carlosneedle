"""
Backtest v1 for the election-night needle.

Given a snapshot of partial election-night returns and a prior-election
baseline per county, project the final statewide margin using a uniform
state-level swing PLUS uncertainty, and produce a Monte Carlo distribution
of possible final margins (the "needle").
"""

import numpy as np
import pandas as pd

MIT_LONG_CSV = r"C:\Users\krish\Documents\CarlosNeedle\data\mit_county_level\countypres_2000-2024.csv"
SNAPSHOT_CSV = r"C:\Users\krish\Documents\CarlosNeedle\data\backtest\ga_pres_2024_snapshot.csv"

STATE_PO = "GA"
BASELINE_YEAR = 2020   # 2020 Pres baseline for 2024 Pres race
ACTUAL_FINAL_MARGIN = 2.20  # Trump +2.20 (R+/D-)

# GA 2020 turnout ~5.0M, GA 2024 turnout ~5.3M -> ratio ~1.06
TURNOUT_RATIO = 1.06

N_SIMS = 10_000
SEED = 42

# Pre-election polling avg had Trump +1.5 in GA (tossup)
PRIOR_MEAN = 1.5
PRIOR_SIGMA = 3.0


def load_pres_baseline(state_po: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(MIT_LONG_CSV)
    d = df[(df["state_po"] == state_po) & (df["year"] == year) &
           (df["office"] == "US PRESIDENT")].copy()
    pv = d.pivot_table(index=["county_name", "totalvotes"],
                       columns="party", values="candidatevotes",
                       aggfunc="sum").reset_index()
    pv = pv.rename(columns={"DEMOCRAT": "d", "REPUBLICAN": "r",
                            "totalvotes": "baseline_votes"})
    pv["baseline_margin"] = (pv["r"] - pv["d"]) / pv["baseline_votes"] * 100
    pv["county_name"] = pv["county_name"].str.upper().str.strip()
    return pv[["county_name", "baseline_margin", "baseline_votes"]]


def run_backtest():
    rng = np.random.default_rng(SEED)
    baseline = load_pres_baseline(STATE_PO, BASELINE_YEAR)
    snap = pd.read_csv(SNAPSHOT_CSV)
    snap["county_key"] = snap["county"].str.upper().str.strip()

    merged = snap.merge(baseline, left_on="county_key", right_on="county_name", how="left")
    if merged["baseline_margin"].isna().any():
        print("UNMATCHED:")
        print(merged[merged["baseline_margin"].isna()][["county"]].to_string())
        return

    reported = merged[merged["reported_margin_rd"].notna() & (merged["reported_votes"] > 0)].copy()
    reported["observed_swing"] = reported["reported_margin_rd"] - reported["baseline_margin"]

    # State-swing MEAN and UNCERTAINTY (weighted mean + weighted SE)
    w = reported["reported_votes"].values
    x = reported["observed_swing"].values
    state_swing_mean = (x * w).sum() / w.sum()
    weighted_var = (w * (x - state_swing_mean)**2).sum() / w.sum()
    n_effective = w.sum()**2 / (w**2).sum()   # Kish effective N
    state_swing_se = np.sqrt(weighted_var / n_effective)

    print(f"== {STATE_PO} Gov 2022 -- snapshot ~73% reporting ==\n")
    print(f"State-swing point:   {state_swing_mean:+.2f}")
    print(f"State-swing SE:      {state_swing_se:.2f}")
    print(f"County-swing spread: {np.sqrt(weighted_var):.2f}")
    print(f"Effective N (Kish):  {n_effective:.1f} of {len(reported)} counties")

    # Prepare vectors for MC
    merged["reported_votes"] = merged["reported_votes"].fillna(0)
    merged["expected_votes"] = merged["baseline_votes"] * TURNOUT_RATIO
    merged["remaining_votes"] = (merged["expected_votes"] - merged["reported_votes"]).clip(lower=0)
    merged["reported_margin_for_agg"] = merged["reported_margin_rd"].fillna(0)

    baseline_arr = merged["baseline_margin"].values
    reported_votes_arr = merged["reported_votes"].values
    remaining_votes_arr = merged["remaining_votes"].values
    expected_votes_arr = merged["expected_votes"].values
    reported_margin_arr = merged["reported_margin_for_agg"].values

    total_expected = expected_votes_arr.sum()

    # Monte Carlo: for each sim draw a state-swing sample, project unreported
    swing_samples = rng.normal(state_swing_mean, state_swing_se, N_SIMS)
    # county-level residual noise (things not captured by uniform swing)
    residual_sigma = np.sqrt(weighted_var)  # per-county swing variability

    state_margins = np.empty(N_SIMS)
    n_counties = len(merged)
    for i, ss in enumerate(swing_samples):
        # each unreported county gets baseline + swing + county-specific residual
        county_residual = rng.normal(0, residual_sigma, n_counties)
        projected_remaining = baseline_arr + ss + county_residual
        total_votes_R_minus_D = (reported_votes_arr * reported_margin_arr +
                                 remaining_votes_arr * projected_remaining)
        state_margins[i] = total_votes_R_minus_D.sum() / total_expected

    # ---- Bayesian update against optional prior ----
    if PRIOR_MEAN is not None:
        live_mean = state_margins.mean()
        live_sigma = state_margins.std()
        post_var = 1.0 / (1.0/PRIOR_SIGMA**2 + 1.0/live_sigma**2)
        post_mean = post_var * (PRIOR_MEAN/PRIOR_SIGMA**2 + live_mean/live_sigma**2)
        post_sigma = np.sqrt(post_var)
        posterior = rng.normal(post_mean, post_sigma, N_SIMS)
    else:
        posterior = state_margins
        live_mean, live_sigma = state_margins.mean(), state_margins.std()

    # ---- Report ----
    print()
    print(f"---- Data-only projection (likelihood) ----")
    print(f"  mean:   {live_mean:+.2f}")
    print(f"  sigma:  {live_sigma:.2f}")
    print(f"  P5..P95: [{np.percentile(state_margins,5):+.2f}, {np.percentile(state_margins,95):+.2f}]")

    if PRIOR_MEAN is not None:
        print(f"\n---- Pre-election prior ----")
        print(f"  mean:  {PRIOR_MEAN:+.2f}   sigma: {PRIOR_SIGMA:.2f}")
        print(f"\n---- Bayesian posterior (needle) ----")
        print(f"  mean:   {posterior.mean():+.2f}")
        print(f"  sigma:  {posterior.std():.2f}")
        print(f"  P5..P95: [{np.percentile(posterior,5):+.2f}, {np.percentile(posterior,95):+.2f}]")

    p_dem = (posterior < 0).mean() * 100
    print(f"\n=== NEEDLE: P(Dem wins) = {p_dem:.1f}% ===")
    print(f"=== Predicted median margin: {np.median(posterior):+.2f} ===")
    print(f"=== ACTUAL final:            {ACTUAL_FINAL_MARGIN:+.2f} ===")
    print(f"=== Error:                    {abs(np.median(posterior) - ACTUAL_FINAL_MARGIN):.2f} pts ===")


if __name__ == "__main__":
    run_backtest()
