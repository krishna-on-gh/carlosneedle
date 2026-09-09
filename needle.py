"""
Election-night Bayesian needle.

Core function: compute_needle(prior, county_reports, baselines, turnout_ratio)
returns a NeedleResult with posterior mean/sigma/percentiles and P(D wins).

Sign convention throughout: positive = R won by that many points.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---- File paths (used only by convenience loaders below) ----
REPO_ROOT           = Path(__file__).parent
COUNTY_BASELINES    = REPO_ROOT / "data" / "county_baselines_2024.csv"
TURNOUT_RATIOS_CSV  = REPO_ROOT / "data" / "state_turnout_ratios_2026.csv"

N_SIMS = 10_000
DEFAULT_TURNOUT_RATIO = 0.80
MIN_STATE_SWING_SE = 0.5     # floor to prevent over-confident tiny data
MIN_RESIDUAL_SIGMA = 0.5     # floor on per-county residual noise


@dataclass
class NeedleResult:
    posterior_mean: float
    posterior_sigma: float
    p5: float
    p50: float          # median (the needle center)
    p95: float
    p_dem_win: float    # 0..1
    n_counties_reported: int
    reported_vote_total: float
    expected_vote_total: float
    state_swing_signal: float | None    # None if no counties reported
    state_swing_se: float | None
    # Diagnostic flags to surface in the UI
    warnings: list[str]

    @property
    def band_width(self) -> float:
        return self.p95 - self.p5


def load_county_baselines(state_po: str) -> pd.DataFrame:
    """Return [county_name (UPPER), baseline_margin, baseline_votes] for a state."""
    df = pd.read_csv(COUNTY_BASELINES)
    sub = df[df["state_po"] == state_po].copy()
    sub["county_name"] = sub["county_name"].str.upper().str.strip()
    return sub[["county_name", "margin_2024", "total_votes_2024"]].rename(
        columns={"margin_2024": "baseline_margin",
                 "total_votes_2024": "baseline_votes"})


def load_turnout_ratio(state_po: str) -> float:
    """Look up a state's midterm turnout ratio, falling back to the default."""
    try:
        df = pd.read_csv(TURNOUT_RATIOS_CSV)
    except FileNotFoundError:
        return DEFAULT_TURNOUT_RATIO
    row = df[df["state_po"] == state_po]
    if len(row) == 0 or pd.isna(row.iloc[0]["turnout_ratio"]):
        return DEFAULT_TURNOUT_RATIO
    try:
        return float(row.iloc[0]["turnout_ratio"])
    except (ValueError, TypeError):
        return DEFAULT_TURNOUT_RATIO


def compute_needle(
    prior_mean: float,
    prior_sigma: float,
    county_reports: pd.DataFrame,   # from paste parser
    state_po: str,
    turnout_ratio: float | None = None,
    n_sims: int = N_SIMS,
    seed: int = 42,
) -> NeedleResult:
    """Combine pre-election prior with live county reports into a posterior needle."""
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    baselines = load_county_baselines(state_po)
    if turnout_ratio is None:
        turnout_ratio = load_turnout_ratio(state_po)

    # Prepare full county universe (baseline).
    full = baselines.copy()
    full["county_name"] = full["county_name"].str.upper().str.strip()
    full["expected_votes"] = full["baseline_votes"] * turnout_ratio

    # Merge in county reports (or default to zero if none provided).
    if len(county_reports) > 0:
        cr = county_reports.copy()
        cr["county_name"] = cr["county"].str.upper().str.strip()
        cr = cr[["county_name", "reported_margin_rd", "reported_votes"]]
        full = full.merge(cr, on="county_name", how="left")
        # Which paste rows didn't match any baseline county? Surface as warnings.
        matched_names = set(full.loc[full["reported_margin_rd"].notna(), "county_name"])
        unmatched = set(cr["county_name"]) - matched_names
        for u in sorted(unmatched):
            warnings.append(f"pasted county not in baselines: {u}")
    else:
        full["reported_margin_rd"] = np.nan
        full["reported_votes"] = 0.0

    full["reported_votes"] = full["reported_votes"].fillna(0.0)
    full["reported_margin"] = full["reported_margin_rd"].fillna(0.0)
    full["has_report"] = full["reported_margin_rd"].notna() & (full["reported_votes"] > 0)

    reported = full[full["has_report"]]
    n_rep = int(reported.shape[0])
    reported_total = float(reported["reported_votes"].sum())
    expected_total = float(full["expected_votes"].sum())

    # ---- Early-out: no data yet. Return prior as posterior. ----
    if n_rep == 0 or reported_total <= 0:
        samples = rng.normal(prior_mean, prior_sigma, n_sims)
        p5, p50, p95 = np.percentile(samples, [5, 50, 95])
        return NeedleResult(
            posterior_mean=float(prior_mean),
            posterior_sigma=float(prior_sigma),
            p5=float(p5), p50=float(p50), p95=float(p95),
            p_dem_win=float((samples < 0).mean()),
            n_counties_reported=0,
            reported_vote_total=0.0,
            expected_vote_total=expected_total,
            state_swing_signal=None,
            state_swing_se=None,
            warnings=warnings + ["No county data reported yet; showing pre-election prior."],
        )

    # ---- Estimate state-level swing signal ----
    x = (reported["reported_margin"] - reported["baseline_margin"]).values
    w = reported["reported_votes"].values
    swing_mean = float((x * w).sum() / w.sum())
    if len(x) > 1:
        weighted_var = float((w * (x - swing_mean) ** 2).sum() / w.sum())
    else:
        weighted_var = 25.0     # generous default with just 1 reporting county
    n_effective = float(w.sum() ** 2 / (w ** 2).sum())
    swing_se = max(np.sqrt(weighted_var / n_effective), MIN_STATE_SWING_SE)
    residual_sigma = max(np.sqrt(weighted_var), MIN_RESIDUAL_SIGMA)

    # Flag when one county dominates (>= 60% of reported vote) — needle over-confident risk.
    top_share = float(w.max() / w.sum()) if w.sum() > 0 else 0.0
    if top_share >= 0.6:
        warnings.append(
            f"one county contributes {top_share*100:.0f}% of reported vote — "
            f"needle may under-estimate uncertainty.")

    # ---- Monte Carlo rollup: project all counties ----
    ba = full["baseline_margin"].values
    rv = full["reported_votes"].values
    ev = full["expected_votes"].values
    rmv = np.maximum(ev - rv, 0.0)
    rm = full["reported_margin"].values

    swing_draws = rng.normal(swing_mean, swing_se, n_sims)
    likelihood_samples = np.empty(n_sims)
    n_c = len(full)
    for i, ss in enumerate(swing_draws):
        county_residual = rng.normal(0.0, residual_sigma, n_c)
        proj_remaining_margin = ba + ss + county_residual
        total_rd = rv * rm + rmv * proj_remaining_margin
        likelihood_samples[i] = total_rd.sum() / expected_total

    # ---- Bayesian update: precision-weighted blend with prior ----
    live_mean = float(likelihood_samples.mean())
    live_sigma = max(float(likelihood_samples.std()), 0.1)
    prec_prior = 1.0 / prior_sigma ** 2
    prec_data  = 1.0 / live_sigma ** 2
    post_var   = 1.0 / (prec_prior + prec_data)
    post_mean  = post_var * (prior_mean * prec_prior + live_mean * prec_data)
    post_sigma = np.sqrt(post_var)
    posterior  = rng.normal(post_mean, post_sigma, n_sims)

    p5, p50, p95 = np.percentile(posterior, [5, 50, 95])
    p_dem = float((posterior < 0).mean())

    return NeedleResult(
        posterior_mean=float(post_mean),
        posterior_sigma=float(post_sigma),
        p5=float(p5), p50=float(p50), p95=float(p95),
        p_dem_win=p_dem,
        n_counties_reported=n_rep,
        reported_vote_total=reported_total,
        expected_vote_total=expected_total,
        state_swing_signal=swing_mean,
        state_swing_se=swing_se,
        warnings=warnings,
    )


# ---- End-to-end convenience for testing ----
if __name__ == "__main__":
    from needle_parser import parse_paste
    from needle_priors import get_prior

    # Test: GA 2024 Pres snapshot with a hand-set prior
    txt = (REPO_ROOT / "data" / "backtest" / "ga_pres_2024_snapshot_raw.txt").read_text()
    parsed = parse_paste(txt, dem_last="Harris", rep_last="Trump")
    print(f"Parsed {len(parsed.rows)} counties from GA snapshot")

    r = compute_needle(prior_mean=1.5, prior_sigma=3.0,
                       county_reports=parsed.rows, state_po="GA",
                       turnout_ratio=1.06)  # 2024 was pres-year, ratio > 1
    print(f"\n== Needle result ==")
    print(f"posterior median: {r.p50:+.2f}")
    print(f"90% band:         [{r.p5:+.2f}, {r.p95:+.2f}]")
    print(f"P(Dem wins):      {r.p_dem_win*100:.1f}%")
    print(f"counties reported: {r.n_counties_reported}")
    print(f"reported vote:    {r.reported_vote_total:,.0f} of expected {r.expected_vote_total:,.0f}")
    if r.state_swing_signal is not None:
        print(f"state swing:      {r.state_swing_signal:+.2f} (SE {r.state_swing_se:.2f})")
    for w in r.warnings:
        print(f"  [warn] {w}")
