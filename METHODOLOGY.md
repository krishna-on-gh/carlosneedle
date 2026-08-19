# CarlosNeedle — Methodology & Backtest Findings

Living document describing how the model works and what the backtests showed.
Last updated: 2026-08-06.

---

## Overview

CarlosNeedle is a fundamentals-based Monte Carlo forecast for the 2026 US Senate,
Governor, and House elections. Every race is simulated 10,000 times using a
three-layer model. The **median of the simulations** is the point prediction;
the **5th–95th percentiles** form the 90% uncertainty band.

**Sign convention:** Positive = Republican, Negative = Democrat.
Everywhere. In the CSV, in the engine, in the dashboard.

Mnemonic: positive is on the *right* on a number line, just like the GOP
sits on the political right. Negative is on the *left*, like Dems. It reads
naturally: `+7` means "R+7," `-3` means "D+3."

---

## The three-layer Monte Carlo

For each simulation of each race:

```
Layer 1: Baseline (state/district partisan lean)
Layer 2: + drawn_swing        (Normal, mean=expected_swing, std=swing_unc)
Layer 3: + drawn_quality      (Normal, mean=quality_adj, std=quality_unc)
─────────────────────────────────────────────────
Final:   simulated_margin
```

Where `quality_adj = scandal + incumbency_factor + candidate_quality + spending_effect`.

Ten thousand draws → distribution → median + P5 + P95 + win probabilities.

---

## Per-office calibration (from backtests)

| Office     | Baseline logic                              | swing_unc | quality_unc base |
|------------|---------------------------------------------|-----------|------------------|
| **Senate** | Blend: `0.5 × Pres + 0.5 × SHAVE`, disagreement bump | 3.5       | 1.0              |
| **Governor** | Pres only                                 | 4.0       | 2.0              |
| **House**  | Pres only                                   | 4.5       | 2.0              |

**Quality uncertainty formula (all offices):**
`quality_unc = quality_unc_base + 0.5 × |scandal|`
(wider bands when a scandal is in play).

**Senate blend uncertainty bump:**
`swing_unc = swing_unc_base + 0.25 × |Pres − SHAVE|`
(wider bands when the two baseline signals disagree).

**Why the calibration differs:**
- Senate races are the most predictable — they're nationalized and party-line.
- Gov races are the least predictable — voters cross parties more, candidate
  quality matters more, state-specific issues dominate.
- House races land between the two.

---

## The baseline blend (Senate)

Senate baselines use `0.5 × 2024 Pres margin + 0.5 × 2024 SHAVE`.

Where **SHAVE** is a national House-vote-derived state proxy that adjusts for
uncontested races. It measures the "generic congressional preference" of the
state.

**Why blend:**
- Pres margins can be distorted by top-of-ticket candidate effects
  (Trump-specific coalition dynamics in 2016 & 2024 especially).
- SHAVE can be inflated in states with weak downballot opposition
  (WV, ND, MO, IN historically).
- Blending hedges both failure modes and produces wider bands where the two
  signals disagree (a healthy honesty about uncertainty).

**Fallback:** if SHAVE is blank for a state (e.g., Alaska — at-large,
unreliable SHAVE), the engine uses pure Pres.

Gov and House use pure 2024 Pres as the baseline (2026 v1). Blends may be
added later.

---

## Per-race swing dampening (House especially)

Some races don't fully participate in the national wave. In the CSV,
`expected_swing = -9.65` denotes "full national swing," while `-3.86` denotes
"dampened by 2.5x."

**Rule:** when the app overrides the national swing based on polls, each
race's effective swing scales proportionally:

```
race_swing = effective_national × (csv_swing / -9.65)
```

So:
- Full-swing race (CSV -9.65) → gets the full effective national swing.
- Dampened race (CSV -3.86) → gets effective_national / 2.5.

**Which districts get dampened:**
- California suburbs (Trump-distorted 2016 Pres baselines)
- New York House (2022 R overperformance carried into 2026 assumptions)
- Some heavily-Hispanic districts in TX/FL where wave translation is weak

Applied selectively, informed by the 2018/2022 House backtest findings.

---

## Race-specific alterations

The uniform methodology handles most races well, but a handful get bespoke
adjustments because a specific dynamic can't be captured by the general
rules. These aren't fudges — each is documented and defensible.

**NH-Sen (dampened swing)**
- New Hampshire has an unusually inelastic Senate electorate (Sununu, Hassan,
  Shaheen have all shown personal-vote effects that outrun the state's
  partisan lean). To avoid overstating the state's swing response, NH-Sen
  gets a dampened swing (about 2.5x less than the national wave) instead
  of the full national swing every other Senate race sees.

**PA-Gov (baseline from 2022 Gov result, not 2024 Pres)**
- Josh Shapiro won PA-Gov by 14.8 in 2022 in the same state Trump won by
  1.7 in 2024. That's a 16-point Gov-vs-Pres divergence — Shapiro's personal
  vote is enormous. Using 2024 Pres as the baseline for PA-Gov 2026 would
  ignore ~15 points of documented incumbent-specific overperformance.
  Using the 2022 Gov result as the baseline better captures the reality
  that Shapiro is running from a very different starting point than the
  state's Pres lean would suggest.

**Why bespoke rules at all?**
The uniform methodology assumes each race's baseline is a reasonable proxy
for the electorate that will show up. When that assumption clearly breaks
(NH's unusual elasticity, Shapiro's outsized personal vote), the model gets
a corrected input rather than a pretend-uniform one that produces a
misleading prediction. Every alteration is one line in the CSV, documented
in the `notes` field.

---

## Incumbency & candidate factors

**Incumbency factor:**
- **Senate (2022 & 2026 predictions):** use Split Ticket WAR directly as
  the `incumbency_factor` value. No transformation. Modern published WAR is
  already multi-race calibrated.
- **Senate (2018 backtest only):** Split Ticket didn't publish WAR back
  then, so the value was hand-computed from a single prior-race residual and
  then shrunk to make it usable: `±2.5 + WAR/4` for normal races,
  `±4 + WAR/4` for incumbents in "hostile territory" (opposition holds
  baseline by 5+ pts — e.g., Manchin, Tester, Collins). This shrinkage was a
  workaround for noisy single-race WAR; it does *not* apply to the live 2026
  predictions.
- **Gov / House:** ±2.5 for incumbents, 0 for open seats.
  No WAR proxy applied.

**Scandal tiers:**
| Value | Meaning                                      |
|-------|----------------------------------------------|
| ±1.5  | Moderate scandal (still gets incumbency)     |
| ±3.0  | Severe scandal (cancels incumbency for the candidate)|

Sign follows R+/D-: positive means the scandal hurts the Dem, negative means
it hurts the Republican.

**Candidate quality:** used sparingly, only when signals are overwhelming
(polling, party investment, race ratings from Cook/Sabato). Most races have
`candidate_quality = 0`.

**Spending effect — the rule differs by chamber:**

**Senate & Governor (conditional application):**
- Apply if it's an **open seat in a competitive state** (`|baseline| < 10`), OR
- Apply if the **opposition is outspending an incumbent 2:1 or more**
- Otherwise → `spending_effect = 0`

Rationale: for Senate/Gov races, spending asymmetries in already-decided
races (e.g., a safe-state D outspending a sacrificial R challenger) don't
move votes. Zero-out those situations to avoid noise.

**House (apply everywhere except scandal-involved races):**
- Apply a spending value to **every** House race based on the actual
  fundraising/ad-spend differential — regardless of open/incumbent or
  competitiveness.
- Exception: if the race has a scandal (`scandal ≠ 0`), leave
  `spending_effect = 0` — the scandal effect dominates and spending is
  noise on top.

Rationale: House races are lower-profile and more responsive to spending
differentials than Senate/Gov. Backtests support this — the 2018 & 2022
House Goliath backtests applied spending on ~99% of rows, and it materially
improved capture rate and mean error.

**Magnitude tier (same for all chambers when applied):**
| Spending ratio                    | Value (in direction of outspender) |
|-----------------------------------|------------------------------------|
| ~1:1 (close, small edge)          | ±0.5                               |
| 2:1                               | ±2                                 |
| 3:1 or more                       | ±3.5                               |

**Sign convention:** positive if it helps the Republican, negative if it
helps the Democrat.

**Zero doesn't always mean "no data":**
- For Senate/Gov, zero means "the race doesn't meet the trigger criteria."
- For House, zero means either "a scandal is present" or a data gap that
  should be filled in.

---

## Poll aggregator

Manually-entered generic ballot polls in `data/polls_2026.csv`.

**Weighting:** LV polls get weight 1.5, RV polls get weight 1.0
(→ 60% / 40% split).

**Rolling window:** 21 days (about 3 weeks).

**Projected GCB:** raw polling averages leave 10-15% undecided. The projection
allocates these:
- **56% to Democrats**
- **44% to Republicans**

The 56/44 split is based on how undecideds broke in the **2018 generic
ballot cycle** — a comparable midterm environment (unpopular Republican
president, engaged Dem base). In 2018 late undecided voters split roughly
56/44 toward Dems on final polls-to-result reconciliation, so the same
split is applied here as a working assumption for 2026.

Formula:
```
projected_dem = raw_dem + 0.56 × undecided
projected_rep = raw_rep + 0.44 × undecided
projected_margin_r = projected_rep − projected_dem
```

**Swing derivation:**
```
projected_swing = projected_margin_r − 2024_reference (+2.15)
```

The `2024_reference = +2.15` is the 2024 House popular vote (R+2.15).

The dashboard always uses the projected swing when polls are loaded, scaling
each race's swing proportionally per the dampening rule above.

---

## Chamber composition (Home tab)

Predicted 2026 outcomes are combined with the seats **not** on the 2026
ballot to produce the final chamber composition.

**Senate:** 35 races modeled + 65 carryover senators (Classes 1 & 3, sitting
through 2028 and 2030).

**Governor:** 36 races modeled + 14 non-2026 governors (elected 2024 or 2025).

**House:** all 435 seats are on the ballot each cycle. The CSV models ~89
competitive races; the remaining ~346 are treated as safe based on 2024
incumbent party. *This section is still under construction — House
composition is not yet displayed on the Home tab.*

Editable constants live at the top of `app.py`:
```
SENATE_CARRYOVER_R = 32
SENATE_CARRYOVER_D = 33
GOV_CARRYOVER_R = 8
GOV_CARRYOVER_D = 6
HOUSE_SAFE_R = 189
HOUSE_SAFE_D = 157
```

---

## Backtest findings

### Aggregate performance across 2018 + 2022

| Subset                                | N    | Accuracy | 90% Capture | Mean Error |
|---------------------------------------|------|----------|-------------|------------|
| All races, both cycles                | 136  | 82.4%    | 75.7%       | 5.42       |
| **Excluding CA 2018 & NY 2022 clusters** | **126** | **~85%** | **~75%** | **~5.0** |

CA 2018 and NY 2022 are known regional failure modes (see limitations below).
Excluded from the "clean" benchmark because they're systematic Trump-baseline
distortions that any fundamentals model would miss.

### By cycle & chamber

**2018 Senate (Blend):** 19 races, **84.2%** accuracy, 73.7% capture, mean err 4.52
**2022 Senate (Pres):** 16 races, **100%** accuracy, 81.2% capture, mean err 3.55
**2022 Governor (BlendSwing):** 21 races, **85.7%** accuracy, 61.9% capture, mean err 6.64
**2018 House (Goliath v2, per-district swings):** 170 races, **82.4%** accuracy, 73.5% capture, mean err 5.79
**2022 House (all):** 42 races, **76.2%** accuracy, 85.7% capture, mean err 5.03
**2022 House (excl. NY):** 37 races, **86.5%** accuracy, 91.9% capture, mean err 4.31

### Key methodological wins from the backtests

1. **Blended Senate baseline (Pres + SHAVE) beats pure Pres in high-divergence
   cycles like 2018.** In low-divergence cycles like 2022, Pres and blend
   perform similarly. Blend hedges both scenarios — the safer default.
2. **Per-district swing tuning for House** improved 2018 Goliath v1→v2 capture
   from 64.7% → 73.5%, and cut mean error by 11%.
3. **Wider uncertainty bands for Gov (4.0/2.0)** are honest — 90% capture of
   ~62% is the realistic ceiling for Gov races without per-race polling.
4. **Split Ticket WAR** is the right incumbency proxy — single-race residuals
   have a 38% sign-flip rate (unreliable). Multi-race calibrated WAR is much
   better.
5. **Undecided allocation (56/44 D)** in the poll aggregator brought the
   projected swing (-9.99) into near-perfect alignment with the CSV's original
   assumption (-9.65) — validating both.

---

## Known limitations

### 1. Hispanic-majority districts

In 2018 the model systematically overshot D wins in R-held districts with
high non-white populations (FL-25, CA-42, GA-07, TX-22/23/24, etc.). Cause:
- 2016 Pres baseline is Trump-distorted in the *opposite* direction from CA
  suburbs — Trump *over*performed with Hispanic voters, so a generic-R
  baseline would be more R than Pres suggests.
- Hispanic voters swing less in midterm waves than college-educated whites do.

### 2. Trump-distorted white suburbs (CA 2018, some NY 2022)

CA 2018 CD misses were systematic — Pres baselines showed Clinton
outperforming a generic D (due to Trump-specific weakness with college
whites), so wave predictions overshot. Same dynamic in reverse in some NY-11,
NY-17-adjacent races.

Partial fix: per-district swing dampening (see above). Full fix would require
a SHAVE-equivalent for CDs or an explicit demographic adjustment.

### 3. Candidate-quality upsets

Races like Mastriano (PA-Gov 2022), Max Rose (NY-11 2018), Mia Love loss
(UT-04 2018), Scott Taylor scandal (VA-02 2018) can't be predicted from
fundamentals. Real-time polling and party investment signals would catch
them, but the model doesn't ingest those.

### 4. Wave environment miscalibration

The model is only as good as the swing input. If polling shifts significantly
between input update and Election Day, predictions are stale. The poll
aggregator mitigates this but doesn't eliminate it.

---

## Data lineage

- **`data/races_2026.csv`** — per-race inputs (baseline, swing, incumbency,
  scandal, quality, spending). Manually curated.
- **`data/polls_2026.csv`** — generic ballot polls. Manually curated.
- **`data/state_leg_2026.csv`** — state legislature district inputs.
- **`engine.py`** — Monte Carlo engine; reads CSV, runs 10k sims per race,
  returns results.
- **`poll_aggregator.py`** — poll loading, weighting, projected GCB,
  swing derivation.
- **`app.py`** — Streamlit dashboard.

### Sources

- **U.S. House and state legislative district data (2024 Presidential
  results by district)** — nearly all district-level Pres margins are
  sourced from **[Dave's Redistricting App](https://davesredistricting.org)**.
  DRA disaggregates precinct-level Pres results to whatever district map
  you're viewing (post-2022 lines for most states) and exports clean CSVs
  with Dem/Rep/Other percentages per district. Huge lift saved.
- **Senate & Governor baselines** — 2024 state-level Presidential margins
  (public / widely reported).
- **Split Ticket WAR** for Senate incumbency values.
- **Generic ballot polls** — manually entered from public releases
  (Marquette, Emerson, Reuters/Ipsos, YouGov, NYT, Cygnal, FOX News, Pew,
  Quinnipiac, CNN, NPR/PBS/Marist, and others).

---

## Model version history

- **v1 (Aug 2026):** Initial deployment. Blend baseline for Senate, Pres for
  Gov & House. LV/RV weighted poll aggregator with 56/44 undecided split.
  Per-district swing dampening for known non-wave-following seats.

---

## Who's behind this

These are predictions — real ones. The model runs 10,000 simulations per race,
uses calibrated inputs, and reports honest uncertainty bands. But I'm not a
pundit and I'm not doing this for money. I'm a hobbyist who loves electoral
trends data and wanted to build a rigorous, transparent forecast for the
2026 cycle.

Everything is open — methodology documented here, code and CSVs public,
inputs traceable. Treat the outputs as informed forecasts from someone who
takes the methodology seriously, not as authoritative calls from an
established forecasting operation.
