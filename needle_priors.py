"""
Prior extraction for the election-night needle.

Reads the Monte Carlo forecast output (results_2026.csv) and produces
(prior_mean, prior_sigma) per race_id, ready to feed to the needle's
Bayesian update.

Sign convention (throughout): R+/D- (positive = R won by that many points).
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).parent
RESULTS_CSV       = REPO_ROOT / "results_2026.csv"
NEEDLE_RACES_CSV  = REPO_ROOT / "data" / "needle_races_2026.csv"

# 90% credible interval (P5..P95) covers 3.29 sigmas of a normal distribution
# (1.645 each side). So sigma = (P95 - P5) / 3.29.
CI_WIDTH_TO_SIGMA = 3.29

# Floor on sigma. A race that comes out of the Monte Carlo as a near-certainty
# (P5 ≈ P95, sigma ≈ 0) would make the prior infinitely precise and drown
# out any live data. Floor prevents that.
MIN_SIGMA = 1.0


def load_priors() -> pd.DataFrame:
    """Return a DataFrame with prior_mean and prior_sigma per needle race."""
    races = pd.read_csv(NEEDLE_RACES_CSV)
    results = pd.read_csv(RESULTS_CSV)

    keep = ["race_id", "median_margin", "p5", "p95"]
    missing = [c for c in keep if c not in results.columns]
    if missing:
        raise ValueError(f"results_2026.csv missing columns: {missing}")

    priors = results[keep].copy()
    priors["prior_mean"] = priors["median_margin"]
    priors["prior_sigma"] = ((priors["p95"] - priors["p5"]) / CI_WIDTH_TO_SIGMA).clip(lower=MIN_SIGMA)

    merged = races.merge(priors[["race_id", "prior_mean", "prior_sigma"]],
                         on="race_id", how="left")

    missing_races = merged[merged["prior_mean"].isna()]["race_id"].tolist()
    if missing_races:
        print(f"WARNING: {len(missing_races)} needle races have no matching forecast row:")
        for r in missing_races:
            print(f"  {r}")

    return merged


def get_prior(race_id: str) -> tuple[float, float]:
    """Convenience: return (prior_mean, prior_sigma) for a single race."""
    df = load_priors()
    row = df[df["race_id"] == race_id]
    if len(row) == 0:
        raise KeyError(f"race_id {race_id!r} not found in needle_races_2026.csv")
    return float(row.iloc[0]["prior_mean"]), float(row.iloc[0]["prior_sigma"])


if __name__ == "__main__":
    df = load_priors()
    print(f"Loaded priors for {len(df)} needle races.\n")
    print(f"{'race_id':<10} {'state':<4} {'prior_mean':>11} {'prior_sigma':>12}  interpretation")
    print("-" * 78)
    for _, r in df.sort_values(["office", "state"]).iterrows():
        if pd.isna(r["prior_mean"]):
            print(f"{r['race_id']:<10} {r['state']:<4}   (no forecast row found)")
            continue
        winner = "R" if r["prior_mean"] > 0 else "D"
        magnitude = abs(r["prior_mean"])
        print(f"{r['race_id']:<10} {r['state']:<4} {r['prior_mean']:>+10.2f}  {r['prior_sigma']:>11.2f}  {winner}+{magnitude:.1f} ± {r['prior_sigma']:.1f}")
