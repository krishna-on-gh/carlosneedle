"""
Backtest v2b: shrinkage-bucketed needle.

Each bucket's swing estimate is a precision-weighted blend of:
  - the bucket's own reported swing (mean, SE)
  - the state-level swing (prior)

This prevents partial-reporting bias in one dominant county from distorting
its bucket's swing estimate.
"""

import numpy as np
import pandas as pd
import sys

MIT_LONG_CSV = r"C:\Users\krish\Documents\CarlosNeedle\data\mit_county_level\countypres_2000-2024.csv"

RACES = {
    "MI": dict(
        snapshot=r"C:\Users\krish\Documents\CarlosNeedle\data\backtest\mi_gov_2022_snapshot_73pct.csv",
        state="MI", baseline_year=2020, actual=-10.6, turnout=0.85,
        prior_mean=-7.0, prior_sigma=4.0),
    "NV": dict(
        snapshot=r"C:\Users\krish\Documents\CarlosNeedle\data\backtest\nv_gov_2022_snapshot_50pct.csv",
        state="NV", baseline_year=2020, actual=1.5, turnout=0.72,
        prior_mean=1.0, prior_sigma=4.0),
}
N_SIMS = 10_000
SEED = 42


def assign_bucket(v):
    if v >= 200_000: return "urban"
    if v >= 40_000:  return "suburban"
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


def wmean_wse(x, w):
    if len(x) == 0 or w.sum() == 0: return None, None, None
    mean = (x * w).sum() / w.sum()
    var  = (w * (x - mean)**2).sum() / w.sum() if len(x) > 1 else 4.0
    n_eff = w.sum()**2 / (w**2).sum()
    se = np.sqrt(max(var / n_eff, 0.5))
    return mean, se, np.sqrt(max(var, 0.5))


def run(cfg):
    rng = np.random.default_rng(SEED)
    baseline = load_baseline(cfg["state"], cfg["baseline_year"])
    snap = pd.read_csv(cfg["snapshot"])
    snap["county_key"] = snap["county"].str.upper().str.strip()
    m = snap.merge(baseline, left_on="county_key", right_on="county_name", how="left")
    if m["baseline_margin"].isna().any():
        print("UNMATCHED:"); print(m[m["baseline_margin"].isna()][["county"]]); return

    m["bucket"] = m["baseline_votes"].apply(assign_bucket)
    m["reported_votes"] = m["reported_votes"].fillna(0)
    m["expected_votes"] = m["baseline_votes"] * cfg["turnout"]
    m["remaining_votes"] = (m["expected_votes"] - m["reported_votes"]).clip(lower=0)
    m["reported_margin_for_agg"] = m["reported_margin_rd"].fillna(0)

    reported = m[m["reported_margin_rd"].notna() & (m["reported_votes"] > 0)].copy()
    reported["observed_swing"] = reported["reported_margin_rd"] - reported["baseline_margin"]

    # State-level swing (prior for each bucket)
    state_mean, state_se, state_resid = wmean_wse(
        reported["observed_swing"].values, reported["reported_votes"].values)

    print(f"\n== {cfg['state']} v2b (shrinkage-bucketed) ==")
    print(f"State swing (prior): {state_mean:+.2f}  SE {state_se:.2f}\n")

    bucket_swings = {}
    print(f"{'bucket':<10} {'raw mean':>9} {'raw SE':>7} {'shrunk mean':>12} {'shrunk SE':>10}")
    for b in ["urban", "suburban", "rural"]:
        sub = reported[reported["bucket"] == b]
        raw_mean, raw_se, resid = wmean_wse(sub["observed_swing"].values,
                                            sub["reported_votes"].values)
        if raw_mean is None:
            # No data → fall back to state
            bucket_swings[b] = (state_mean, state_se * 1.5, state_resid)
            print(f"{b:<10} {'(none)':>9} {'--':>7}  {state_mean:>+11.2f} {state_se*1.5:>9.2f}")
            continue
        # Precision-weighted blend of bucket estimate and state prior
        prec_bucket = 1.0 / raw_se**2
        prec_state  = 1.0 / state_se**2
        shrunk_var  = 1.0 / (prec_bucket + prec_state)
        shrunk_mean = shrunk_var * (raw_mean * prec_bucket + state_mean * prec_state)
        shrunk_se   = np.sqrt(shrunk_var)
        bucket_swings[b] = (shrunk_mean, shrunk_se, resid)
        print(f"{b:<10} {raw_mean:>+8.2f}  {raw_se:>6.2f}  {shrunk_mean:>+11.2f} {shrunk_se:>9.2f}")

    # MC rollup
    ba, rv, ev, rmv, rmg = (m["baseline_margin"].values, m["reported_votes"].values,
                            m["expected_votes"].values, m["remaining_votes"].values,
                            m["reported_margin_for_agg"].values)
    buckets = m["bucket"].values
    n_c = len(m)
    state_margins = np.empty(N_SIMS)
    for i in range(N_SIMS):
        proj = np.empty(n_c)
        for j in range(n_c):
            mean, se, resid = bucket_swings[buckets[j]]
            proj[j] = ba[j] + rng.normal(mean, se) + rng.normal(0, resid)
        total = rv * rmg + rmv * proj
        state_margins[i] = total.sum() / ev.sum()

    live_mean, live_sigma = state_margins.mean(), state_margins.std()
    post_var = 1.0 / (1.0/cfg["prior_sigma"]**2 + 1.0/live_sigma**2)
    post_mean = post_var * (cfg["prior_mean"]/cfg["prior_sigma"]**2 + live_mean/live_sigma**2)
    posterior = rng.normal(post_mean, np.sqrt(post_var), N_SIMS)
    p_dem = (posterior < 0).mean() * 100
    err = abs(np.median(posterior) - cfg["actual"])

    print(f"\nData-only (v2b):    {live_mean:+.2f}  sigma {live_sigma:.2f}")
    print(f"Posterior:          {posterior.mean():+.2f}  sigma {posterior.std():.2f}")
    print(f"NEEDLE median:      {np.median(posterior):+.2f}")
    print(f"ACTUAL:             {cfg['actual']:+.2f}")
    print(f"Error:              {err:.2f} pts")
    print(f"P(Dem wins):        {p_dem:.1f}%")
    return err, live_sigma


if __name__ == "__main__":
    picks = sys.argv[1:] if len(sys.argv) > 1 else ["MI", "NV"]
    for p in picks:
        run(RACES[p])
