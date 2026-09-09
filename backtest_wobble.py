"""
Wobble chart: simulate the needle's behavior across the night.

Uses the MI 2022 73% snapshot as a proxy for "eventual county margins"
(most counties in the snapshot are >85% reported, so their margins are
close to final). Then simulates the reporting order:
  - Small rural counties report first
  - Medium counties fill in
  - Large urban counties (Wayne, Oakland) report last

At each threshold (2%, 5%, 10%, 20%, ..., 73%), compute what the needle
would have said. This is the "would-the-needle-embarrass-us-early?" test.
"""

import numpy as np
import pandas as pd

MIT_LONG_CSV = r"C:\Users\krish\Documents\CarlosNeedle\data\mit_county_level\countypres_2000-2024.csv"
SNAPSHOT_CSV = r"C:\Users\krish\Documents\CarlosNeedle\data\backtest\mi_gov_2022_snapshot_73pct.csv"

STATE_PO = "MI"
BASELINE_YEAR = 2020
ACTUAL_FINAL = -10.6
TURNOUT_RATIO = 0.85  # MI 2022 midterm was ~85% of 2020 Pres turnout
N_SIMS = 5000
SEED = 42


def load_baseline():
    df = pd.read_csv(MIT_LONG_CSV)
    d = df[(df["state_po"] == STATE_PO) & (df["year"] == BASELINE_YEAR) &
           (df["office"] == "US PRESIDENT")]
    pv = d.pivot_table(index=["county_name", "totalvotes"], columns="party",
                       values="candidatevotes", aggfunc="sum").reset_index()
    pv = pv.rename(columns={"DEMOCRAT": "d", "REPUBLICAN": "r",
                            "totalvotes": "baseline_votes"})
    pv["baseline_margin"] = (pv["r"] - pv["d"]) / pv["baseline_votes"] * 100
    pv["county_name"] = pv["county_name"].str.upper().str.strip()
    return pv[["county_name", "baseline_margin", "baseline_votes"]]


def needle_at_snapshot(reporting_pct_per_county, county_final_margins,
                       baseline_margins, expected_votes, rng):
    """Given per-county reporting % (0..1) and their assumed final margins,
    compute the needle projection with uncertainty."""
    reported_votes = reporting_pct_per_county * expected_votes
    remaining_votes = expected_votes - reported_votes

    reported_mask = reported_votes > 0
    if reported_mask.sum() < 2:
        # Not enough data to estimate swing — return prior-only
        return None, None, None

    observed_swing = (county_final_margins - baseline_margins)[reported_mask]
    w = reported_votes[reported_mask]
    state_swing_mean = (observed_swing * w).sum() / w.sum()
    weighted_var = (w * (observed_swing - state_swing_mean)**2).sum() / w.sum()
    n_eff = w.sum()**2 / (w**2).sum()
    state_swing_se = np.sqrt(max(weighted_var / n_eff, 0.01))
    residual_sigma = np.sqrt(max(weighted_var, 0.5))

    # MC rollup
    swing_samples = rng.normal(state_swing_mean, state_swing_se, N_SIMS)
    n_counties = len(baseline_margins)
    state_margins = np.empty(N_SIMS)
    for i, ss in enumerate(swing_samples):
        county_residual = rng.normal(0, residual_sigma, n_counties)
        projected_remaining = baseline_margins + ss + county_residual
        total = (reported_votes * county_final_margins +
                 remaining_votes * projected_remaining)
        state_margins[i] = total.sum() / expected_votes.sum()

    return state_margins


def main():
    rng = np.random.default_rng(SEED)
    baseline = load_baseline()
    snap = pd.read_csv(SNAPSHOT_CSV)
    snap["county_key"] = snap["county"].str.upper().str.strip()
    m = snap.merge(baseline, left_on="county_key", right_on="county_name", how="left")

    baseline_margins = m["baseline_margin"].values
    expected_votes = (m["baseline_votes"] * TURNOUT_RATIO).values
    # Use current reported margin as proxy for "eventual final margin"
    # For unreported counties (Benzie), use baseline as a rough proxy
    county_final_margins = m["reported_margin_rd"].fillna(m["baseline_margin"]).values

    # Reporting ORDER: small rural counties first, big urban last
    # Approximation: sort by expected votes ascending (small first)
    order = np.argsort(expected_votes)
    ordered_votes = expected_votes[order]
    total_expected = expected_votes.sum()

    thresholds = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    print(f"MI Gov 2022 -- needle band as vote reports in")
    print(f"Actual final: D+{-ACTUAL_FINAL:.1f}\n")
    print(f"{'% in':>5} | {'P5':>7}  {'median':>8}  {'P95':>7} | width | {'P(Dem)':>7} | band as text")
    print("-" * 105)

    for target_pct in thresholds:
        target_votes = target_pct * total_expected
        reporting_pct = np.zeros(len(m))
        cumvotes = 0.0
        for idx in order:
            if cumvotes >= target_votes: break
            v = expected_votes[idx]
            need = target_votes - cumvotes
            reporting_pct[idx] = min(1.0, need / v)
            cumvotes += min(need, v)

        samples = needle_at_snapshot(reporting_pct, county_final_margins,
                                     baseline_margins, expected_votes, rng)
        if samples is None:
            print(f"{target_pct*100:>4.0f}% | (not enough counties reporting)"); continue
        p5, med, p95 = np.percentile(samples, [5, 50, 95])
        width = p95 - p5
        p_dem = (samples < 0).mean() * 100

        # ASCII band: D-favored on left (negative), R on right (positive)
        # Range -20 to +5 mapped to 50 chars
        def pos(m, lo=-20, hi=5, w=50):
            return max(0, min(w-1, int((m - lo)/(hi - lo) * w)))
        line = [" "] * 50
        line[pos(0)] = "|"  # zero line
        # Mark actual with X
        line[pos(ACTUAL_FINAL)] = "X"
        # Draw band from P5 to P95
        for p in range(pos(p5), pos(p95)+1):
            if line[p] == " ": line[p] = "-"
        line[pos(med)] = "*"
        band_str = "".join(line)
        print(f"{target_pct*100:>4.0f}% | {p5:>+6.2f}  {med:>+7.2f}  {p95:>+6.2f} | {width:>5.2f} | {p_dem:>5.1f}%  | [D{band_str}R]")
    print()
    print("Legend: * = median  | = zero  X = actual final  - = 90% credible band")


if __name__ == "__main__":
    main()
