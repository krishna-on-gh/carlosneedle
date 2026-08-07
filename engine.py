"""
CarlosNeedle 2026 — Monte Carlo Engine

Exposes:
  - simulate_races(df, swing_shift=0.0, seed=42) -> results DataFrame
    Used by both the CLI (main) and the Streamlit app.

Baseline logic by office:
  - Senate:  blend 0.5*Pres + 0.5*SHAVE (fall back to Pres if SHAVE blank)
  - Gov:     Pres only
  - House:   Pres only

Uncertainty calibrations (from backtests):
  - Senate:  swing_unc=3.5, quality_unc base=1.0
  - Gov:     swing_unc=4.0, quality_unc base=2.0
  - House:   swing_unc=4.5, quality_unc base=2.0

When a Senate baseline blend is used, swing_unc gets a disagreement bump:
  effective_swing_unc = swing_unc_base + 0.25 * |Pres - SHAVE|
"""

import numpy as np
import pandas as pd

DEFAULT_CSV = "data/races_2026.csv"
DEFAULT_OUT = "results_2026.csv"

N_SIMS_DEFAULT = 10_000
SEED_DEFAULT   = 42

OFFICE_SETTINGS = {
    "Senate":   {"swing_unc": 3.5, "qual_base": 1.0, "blend": True},
    "Governor": {"swing_unc": 4.0, "qual_base": 2.0, "blend": False},
    "House":    {"swing_unc": 4.5, "qual_base": 2.0, "blend": False},
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _to_float(x, default=0.0):
    if x is None or x == "" or (isinstance(x, str) and x.strip().upper() == "N/A"):
        return default
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (ValueError, TypeError):
        return default


def _is_blank(x):
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and (x.strip() == "" or x.strip().upper() == "N/A"):
        return True
    return False


def _baseline_and_unc(row, settings):
    pres  = row.get("baseline_pres")
    shave = row.get("baseline_shave")
    base_swing_unc = settings["swing_unc"]

    if _is_blank(pres):
        return None, None, "MISSING_PRES"

    pres = float(pres)
    if settings["blend"] and not _is_blank(shave):
        shave = float(shave)
        baseline = 0.5 * pres + 0.5 * shave
        disagreement = abs(pres - shave)
        swing_unc = base_swing_unc + 0.25 * disagreement
        return baseline, swing_unc, f"Blend (|Pres-SHAVE|={disagreement:.2f})"
    return pres, base_swing_unc, "Pres only"


def _run_one(baseline, swing, swing_unc, scandal, incumbency, quality, spending,
             qual_base, rng, n_sims):
    qual_unc = qual_base + abs(scandal) * 0.5
    quality_adj = scandal + incumbency + quality + spending
    drawn_swings  = rng.normal(swing, swing_unc, size=n_sims)
    drawn_quality = rng.normal(quality_adj, qual_unc, size=n_sims)
    sims = baseline + drawn_swings + drawn_quality
    return (
        float(np.median(sims)),
        float(np.percentile(sims, 5)),
        float(np.percentile(sims, 95)),
        float((sims > 0).mean()),
    )


# ── Public API ───────────────────────────────────────────────────────────────
def simulate_races(df, swing_shift=0.0, n_sims=N_SIMS_DEFAULT, seed=SEED_DEFAULT):
    """
    Run Monte Carlo for every race in df.

    Args:
        df: DataFrame from races_2026.csv
        swing_shift: added to each race's expected_swing (for scenario testing)
        n_sims: sims per race
        seed: RNG seed for reproducibility

    Returns:
        (results_df, skipped_list)
    """
    rng = np.random.default_rng(seed)
    results = []
    skipped = []

    for _, row in df.iterrows():
        office = row.get("office")
        if office not in OFFICE_SETTINGS:
            skipped.append((row.get("race_id"), f"unknown office: {office}"))
            continue

        settings = OFFICE_SETTINGS[office]
        baseline, swing_unc, blend_info = _baseline_and_unc(row, settings)
        if baseline is None:
            skipped.append((row.get("race_id"), blend_info))
            continue

        swing      = _to_float(row.get("expected_swing")) + swing_shift
        scandal    = _to_float(row.get("scandal"))
        incumbency = _to_float(row.get("incumbency_factor"))
        quality    = _to_float(row.get("candidate_quality"))
        spending   = _to_float(row.get("spending_effect"))

        median, p5, p95, win_prob_R = _run_one(
            baseline, swing, swing_unc,
            scandal, incumbency, quality, spending,
            settings["qual_base"], rng, n_sims,
        )

        results.append({
            "race_id":          row.get("race_id"),
            "office":           office,
            "state":            row.get("state"),
            "district":         row.get("district"),
            "tier":             row.get("tier"),
            "incumbent_party":  row.get("incumbent_party"),
            "baseline_used":    round(baseline, 3),
            "blend_info":       blend_info,
            "swing_used":       round(swing, 3),
            "swing_unc_used":   round(swing_unc, 3),
            "median_margin":    round(median, 2),
            "p5":               round(p5, 2),
            "p95":              round(p95, 2),
            "win_prob_R":       round(win_prob_R, 4),
            "win_prob_D":       round(1.0 - win_prob_R, 4),
            "predicted_winner": "R" if median > 0 else "D",
            "notes":            row.get("notes"),
        })

    return pd.DataFrame(results), skipped


def main():
    """CLI: read default CSV, run simulation, write results, print summary."""
    df = pd.read_csv(DEFAULT_CSV)
    print(f"Loaded {len(df)} rows from {DEFAULT_CSV}\n")

    out_df, skipped = simulate_races(df)
    out_df.to_csv(DEFAULT_OUT, index=False)
    print(f"Wrote {len(out_df)} rows to {DEFAULT_OUT}")

    if skipped:
        print(f"\nSkipped {len(skipped)} rows:")
        for rid, reason in skipped:
            print(f"  {rid}: {reason}")

    print("\n" + "=" * 70)
    print("SUMMARY BY OFFICE")
    print("=" * 70)
    for office in ["Senate", "Governor", "House"]:
        sub = out_df[out_df["office"] == office]
        if len(sub) == 0:
            continue
        n = len(sub)
        r_wins = (sub["predicted_winner"] == "R").sum()
        d_wins = n - r_wins
        tossups = ((sub["win_prob_R"] >= 0.4) & (sub["win_prob_R"] <= 0.6)).sum()
        print(f"  {office:<10} | N={n:>3} | R: {r_wins:>3} | D: {d_wins:>3} | Tossups: {tossups:>2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
