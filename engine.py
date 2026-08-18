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
    "Senate":       {"swing_unc": 3.5, "qual_base": 1.0, "blend": True},
    "Governor":     {"swing_unc": 4.0, "qual_base": 2.0, "blend": False},
    "House":        {"swing_unc": 4.5, "qual_base": 2.0, "blend": False},
    # State legislatures — state-level races, use blend (Pres + SHAVE)
    "State House":  {"swing_unc": 4.5, "qual_base": 1.5, "blend": True},
    "State Senate": {"swing_unc": 4.0, "qual_base": 1.5, "blend": True},
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
FEDERAL_OFFICES  = ("Senate", "Governor", "House")
STATELEG_OFFICES = ("State House", "State Senate")


def _simulate_row(row, override_national_swing, csv_national_swing, rng, n_sims,
                   swing_override=None):
    """Simulate one race. Returns dict (or None + reason string)."""
    office = row.get("office")
    if office not in OFFICE_SETTINGS:
        return None, f"unknown office: {office}"

    settings = OFFICE_SETTINGS[office]
    baseline, swing_unc, blend_info = _baseline_and_unc(row, settings)
    if baseline is None:
        return None, blend_info

    if swing_override is not None:
        # Explicit swing override (used for state-leg with derived state swing)
        swing = swing_override
    else:
        csv_swing = _to_float(row.get("expected_swing"))
        if override_national_swing is None or csv_national_swing == 0:
            swing = csv_swing
        else:
            ratio = csv_swing / csv_national_swing
            swing = override_national_swing * ratio

    scandal    = _to_float(row.get("scandal"))
    incumbency = _to_float(row.get("incumbency_factor"))
    quality    = _to_float(row.get("candidate_quality"))
    spending   = _to_float(row.get("spending_effect"))

    median, p5, p95, win_prob_R = _run_one(
        baseline, swing, swing_unc,
        scandal, incumbency, quality, spending,
        settings["qual_base"], rng, n_sims,
    )

    return {
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
        "p95":               round(p95, 2),
        "win_prob_R":       round(win_prob_R, 4),
        "win_prob_D":       round(1.0 - win_prob_R, 4),
        "predicted_winner": "R" if median > 0 else "D",
        "notes":            row.get("notes"),
    }, None


def simulate_races(df, override_national_swing=None, csv_national_swing=-9.65,
                   n_sims=N_SIMS_DEFAULT, seed=SEED_DEFAULT):
    """
    Two-pass Monte Carlo:
      1. Federal races (Senate, Governor, House) — run normally with national swing
      2. State legislatures — auto-derive per-state swing from the state's Gov
         and/or Senate margin in pass 1. Formula:
             derived_state_swing = state_race_predicted_margin − state_pres_baseline
         If a state has both Gov and Sen races, the two are averaged.
         If neither, the state-leg row's own expected_swing is used.

    Args match the prior version.
    Returns (results_df, skipped_list).
    """
    rng = np.random.default_rng(seed)
    results = []
    skipped = []

    # ── Pass 1: federal races ──
    federal = df[df["office"].isin(FEDERAL_OFFICES)]
    for _, row in federal.iterrows():
        res, reason = _simulate_row(row, override_national_swing, csv_national_swing,
                                     rng, n_sims)
        if res is None:
            skipped.append((row.get("race_id"), reason))
        else:
            results.append(res)

    pass1 = pd.DataFrame(results)

    # ── Compute per-state derived swings from Gov and Sen results ──
    state_swing_map = {}
    if len(pass1):
        for state in pass1["state"].unique():
            derived = []
            for _, r in pass1[(pass1["state"] == state) &
                              (pass1["office"].isin(["Governor", "Senate"]))].iterrows():
                # Look up the ORIGINAL Pres baseline for this state's federal race
                src_row = federal[federal["race_id"] == r["race_id"]].iloc[0]
                pres = src_row.get("baseline_pres")
                if _is_blank(pres):
                    continue
                derived.append(r["median_margin"] - float(pres))
            if derived:
                state_swing_map[state] = sum(derived) / len(derived)

    # ── Pass 2: state legislature races ──
    stateleg = df[df["office"].isin(STATELEG_OFFICES)]
    for _, row in stateleg.iterrows():
        state = row.get("state")
        derived_swing = state_swing_map.get(state)
        # Dampen swing by 2.5x for D-leaning districts (baseline_pres < 0).
        # Rationale: already-D districts have less R vote to convert in a D
        # wave (and less D vote to lose in an R wave), so they swing less.
        if derived_swing is not None:
            pres = row.get("baseline_pres")
            if not _is_blank(pres) and float(pres) < 0:
                derived_swing = derived_swing / 2.5
        res, reason = _simulate_row(row, override_national_swing, csv_national_swing,
                                     rng, n_sims, swing_override=derived_swing)
        if res is None:
            skipped.append((row.get("race_id"), reason))
        else:
            results.append(res)

    # ── Anything else (unknown office types) ──
    others = df[~df["office"].isin(FEDERAL_OFFICES + STATELEG_OFFICES)]
    for _, row in others.iterrows():
        res, reason = _simulate_row(row, override_national_swing, csv_national_swing,
                                     rng, n_sims)
        if res is None:
            skipped.append((row.get("race_id"), reason))
        else:
            results.append(res)

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
