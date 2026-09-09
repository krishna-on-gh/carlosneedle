"""
Backtest v2: urban/suburban/rural bucket-aware needle.

Same as v1, but computes swing separately per bucket. Wayne (urban)
projected with urban swing; rural counties with rural swing.
"""

import numpy as np
import pandas as pd

MIT_LONG_CSV = r"C:\Users\krish\Documents\CarlosNeedle\data\mit_county_level\countypres_2000-2024.csv"

# ---- Race config (edit these two lines to switch races) ----
SNAPSHOT_CSV = r"C:\Users\krish\Documents\CarlosNeedle\data\backtest\nv_gov_2022_snapshot_50pct.csv"
STATE_PO = "NV"; BASELINE_YEAR = 2020; ACTUAL_FINAL = 1.5; TURNOUT_RATIO = 0.72
PRIOR_MEAN, PRIOR_SIGMA = 1.0, 4.0

N_SIMS = 10_000
SEED = 42


def assign_bucket(baseline_votes: float) -> str:
    """Population proxy: total 2020 Pres votes cast."""
    if baseline_votes >= 200_000:
        return "urban"
    if baseline_votes >= 40_000:
        return "suburban"
    return "rural"


def load_baseline(state_po, year):
    df = pd.read_csv(MIT_LONG_CSV)
    d = df[(df["state_po"] == state_po) & (df["year"] == year) &
           (df["office"] == "US PRESIDENT")]
    pv = d.pivot_table(index=["county_name", "totalvotes"], columns="party",
                       values="candidatevotes", aggfunc="sum").reset_index()
    pv = pv.rename(columns={"DEMOCRAT": "d", "REPUBLICAN": "r",
                            "totalvotes": "baseline_votes"})
    pv["baseline_margin"] = (pv["r"] - pv["d"]) / pv["baseline_votes"] * 100
    pv["county_name"] = pv["county_name"].str.upper().str.strip()
    return pv[["county_name", "baseline_margin", "baseline_votes"]]


def run():
    rng = np.random.default_rng(SEED)
    baseline = load_baseline(STATE_PO, BASELINE_YEAR)
    snap = pd.read_csv(SNAPSHOT_CSV)
    snap["county_key"] = snap["county"].str.upper().str.strip()
    m = snap.merge(baseline, left_on="county_key", right_on="county_name", how="left")
    if m["baseline_margin"].isna().any():
        print("UNMATCHED:"); print(m[m["baseline_margin"].isna()][["county"]]); return

    m["bucket"] = m["baseline_votes"].apply(assign_bucket)
    m["reported_votes"] = m["reported_votes"].fillna(0)
    m["expected_votes"] = m["baseline_votes"] * TURNOUT_RATIO
    m["remaining_votes"] = (m["expected_votes"] - m["reported_votes"]).clip(lower=0)
    m["reported_margin_for_agg"] = m["reported_margin_rd"].fillna(0)

    print(f"== {STATE_PO} needle v2 (bucketed) ==\n")
    print(f"Bucket composition:")
    for b in ["urban", "suburban", "rural"]:
        sub = m[m["bucket"] == b]
        print(f"  {b:<8} {len(sub):>3} counties  expected votes {sub['expected_votes'].sum()/1e6:.2f}M  "
              f"({sub['expected_votes'].sum()/m['expected_votes'].sum()*100:.1f}% of state)")

    # ---- Per-bucket swing estimates ----
    reported = m[m["reported_margin_rd"].notna() & (m["reported_votes"] > 0)].copy()
    reported["observed_swing"] = reported["reported_margin_rd"] - reported["baseline_margin"]

    bucket_swings = {}
    print(f"\n{'Bucket':<10} {'#rep':>5} {'%repVote':>9} {'swing':>7} {'SE':>6}")
    for b in ["urban", "suburban", "rural"]:
        sub = reported[reported["bucket"] == b]
        w = sub["reported_votes"].values
        x = sub["observed_swing"].values
        if len(sub) == 0 or w.sum() == 0:
            bucket_swings[b] = None
            print(f"{b:<10} {0:>5} (none reported -- will fall back to state swing)")
            continue
        mean = (x * w).sum() / w.sum()
        var  = (w * (x - mean)**2).sum() / w.sum() if len(sub) > 1 else 4.0
        n_eff = w.sum()**2 / (w**2).sum()
        se = np.sqrt(max(var / n_eff, 0.5))
        bucket_swings[b] = (mean, se, np.sqrt(max(var, 0.5)))
        # % of the bucket's expected vote that's reported
        b_exp = m[m["bucket"] == b]["expected_votes"].sum()
        pct_bucket_in = w.sum() / b_exp * 100 if b_exp else 0
        print(f"{b:<10} {len(sub):>5} {pct_bucket_in:>7.1f}%  {mean:>+6.2f}  {se:>5.2f}")

    # Fallback state swing for empty buckets
    w_all = reported["reported_votes"].values
    x_all = reported["observed_swing"].values
    state_mean = (x_all * w_all).sum() / w_all.sum()
    state_var  = (w_all * (x_all - state_mean)**2).sum() / w_all.sum()
    state_se   = np.sqrt(state_var / (w_all.sum()**2 / (w_all**2).sum()))
    for b in bucket_swings:
        if bucket_swings[b] is None:
            bucket_swings[b] = (state_mean, state_se * 1.5, np.sqrt(state_var))

    # ---- Monte Carlo rollup ----
    baseline_arr = m["baseline_margin"].values
    rv, ev, rmv = m["reported_votes"].values, m["expected_votes"].values, m["remaining_votes"].values
    rmg = m["reported_margin_for_agg"].values
    buckets = m["bucket"].values
    n_c = len(m)

    state_margins = np.empty(N_SIMS)
    for i in range(N_SIMS):
        proj_remaining = np.empty(n_c)
        for j in range(n_c):
            mean, se, resid = bucket_swings[buckets[j]]
            swing_draw = rng.normal(mean, se)
            county_residual = rng.normal(0, resid)
            proj_remaining[j] = baseline_arr[j] + swing_draw + county_residual
        total = rv * rmg + rmv * proj_remaining
        state_margins[i] = total.sum() / ev.sum()

    # ---- Bayesian update with prior ----
    live_mean, live_sigma = state_margins.mean(), state_margins.std()
    post_var = 1.0 / (1.0/PRIOR_SIGMA**2 + 1.0/live_sigma**2)
    post_mean = post_var * (PRIOR_MEAN/PRIOR_SIGMA**2 + live_mean/live_sigma**2)
    post_sigma = np.sqrt(post_var)
    posterior = rng.normal(post_mean, post_sigma, N_SIMS)
    p_dem = (posterior < 0).mean() * 100

    print()
    print(f"---- Data-only (v2, bucketed) ----")
    print(f"  mean:  {live_mean:+.2f}   sigma: {live_sigma:.2f}")
    print(f"  P5..P95: [{np.percentile(state_margins,5):+.2f}, {np.percentile(state_margins,95):+.2f}]")
    print(f"\n---- Bayesian posterior ----")
    print(f"  mean:  {posterior.mean():+.2f}   sigma: {posterior.std():.2f}")
    print(f"\n=== NEEDLE v2:  P(Dem wins) = {p_dem:.1f}% ===")
    print(f"=== Predicted median: {np.median(posterior):+.2f} ===")
    print(f"=== ACTUAL:           {ACTUAL_FINAL:+.2f} ===")
    print(f"=== Error:            {abs(np.median(posterior) - ACTUAL_FINAL):.2f} pts ===")


if __name__ == "__main__":
    run()
