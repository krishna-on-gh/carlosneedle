"""
CarlosNeedle — 2026 US Elections Forecast Dashboard

Tabbed Streamlit app:
  1. Home — chamber odds + generic ballot aggregator (WIP)
  2. Senate — detailed Senate predictions
  3. House — detailed House predictions
  4. Governor — detailed Gov predictions
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import simulate_races, _baseline_and_unc, _to_float, OFFICE_SETTINGS
from poll_aggregator import (
    load_polls, append_poll, rolling_average, fever_series,
    poll_derived_swing, projected_gcb, projected_swing,
    summary as poll_summary,
    REFERENCE_2024, POLLS_CSV, UNDECIDED_TO_D, UNDECIDED_TO_R,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarlosNeedle — 2026 Election Forecast",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSV_PATH = "data/races_2026.csv"

# 538-adjacent palette
R_COLOR      = "#cf3d3f"
D_COLOR      = "#4e7be3"
TOSSUP_COLOR = "#9c9c9c"
BG_COLOR     = "#ffffff"

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    h2 { font-weight: 600; border-bottom: 1px solid #e6e6e6; padding-bottom: 6px; margin-top: 24px; }
    h3 { font-weight: 600; color: #333; }
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-weight: 700; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 500;
        padding: 12px 18px;
    }
    /* WIP badge */
    .wip-badge {
        background: #fff4c8;
        border: 1px solid #e6c14a;
        color: #7a5b00;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-left: 8px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached engine wrapper ────────────────────────────────────────────────────
def _file_mtime(path):
    """File modification time — used as a cache-buster when the CSV changes."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


@st.cache_data(show_spinner="Running Monte Carlo…")
def run_engine(csv_path, override_national_swing, csv_national_swing, seed, _mtime):
    """`_mtime` is unused inside, but including it in the signature causes the
    cache to invalidate whenever the CSV file is edited."""
    df = pd.read_csv(csv_path)
    results, _ = simulate_races(
        df,
        override_national_swing=override_national_swing,
        csv_national_swing=csv_national_swing,
        seed=seed,
    )
    return results


# ── Rating helpers ───────────────────────────────────────────────────────────
def rating(p):
    if p >= 0.90: return "Safe R"
    if p >= 0.75: return "Likely R"
    if p >= 0.60: return "Lean R"
    if p > 0.40:  return "Tossup"
    if p > 0.25:  return "Lean D"
    if p > 0.10:  return "Likely D"
    return "Safe D"


def rating_color(r):
    palette = {
        "Safe R":   "#a01a1c",
        "Likely R": "#cf3d3f",
        "Lean R":   "#e88b8c",
        "Tossup":   TOSSUP_COLOR,
        "Lean D":   "#98b3ee",
        "Likely D": "#4e7be3",
        "Safe D":   "#1e4bab",
    }
    return palette.get(r, TOSSUP_COLOR)


RATING_ORDER = ["Safe D", "Likely D", "Lean D", "Tossup", "Lean R", "Likely R", "Safe R"]


# ── Fixed config (sidebar removed) ───────────────────────────────────────────
seed = 42

# CSV's baseline national swing (what the -9.65 values in the CSV represent)
CSV_NATIONAL_SWING = -9.65

# ── Chamber composition: seats NOT on the 2026 ballot ───────────────────────
# Post-2024 sitting membership that carries into the 119th Congress. Update if
# vacancies, party switches, or special-election outcomes change these.
#
# Senate: 65 senators NOT up in 2026 (Class 1 up '24, Class 3 up '22)
SENATE_CARRYOVER_R = 32
SENATE_CARRYOVER_D = 33  # includes 3 indies caucusing with Dems

# Governor: 14 governors NOT up in 2026 (elected in 2024 or 2025)
GOV_CARRYOVER_R = 8
GOV_CARRYOVER_D = 6

# House: 346 seats NOT in the CSV — assumed safe based on 2024 result.
# 89 modeled + 346 safe = 435 total. Adjust when you add/remove CSV rows.
HOUSE_SAFE_R = 189
HOUSE_SAFE_D = 157

# Total seats needed for majority
SENATE_MAJORITY = 51
HOUSE_MAJORITY = 218


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# CarlosNeedle")
st.markdown(
    "<div style='color:#6a6a6a; font-size:1.05rem; margin-top:-8px;'>"
    "A Monte Carlo forecast for the 2026 US Senate, Governor, and House elections. "
    "Ten thousand simulations per race, calibrated on 2018 &amp; 2022 backtests."
    "</div>",
    unsafe_allow_html=True,
)


# ── Load polls & compute poll-derived swing ──────────────────────────────────
polls_df = load_polls(POLLS_CSV)
poll_summary_d = poll_summary(polls_df, window_days=21)
poll_swing_raw = poll_summary_d.get("swing")
poll_swing_proj = poll_summary_d.get("projected_swing")

# Always use the projected poll-derived swing (if polls exist)
# Per-race dampening ratios are preserved by the engine.
use_polls = True
override_national = poll_swing_proj if poll_swing_proj is not None else None
effective_national = override_national if override_national is not None else CSV_NATIONAL_SWING

# ── Run engine ───────────────────────────────────────────────────────────────
results = run_engine(CSV_PATH, override_national, CSV_NATIONAL_SWING, seed, _file_mtime(CSV_PATH))
results["rating"] = results["win_prob_R"].apply(rating)


# ── Flip detection ───────────────────────────────────────────────────────────
def held_party(incumbent):
    """Return 'R', 'D', or None from the incumbent_party field."""
    if incumbent is None:
        return None
    s = str(incumbent).strip().upper()
    if "N/A" in s:
        return None
    if s == "R" or s == "OPEN - R":
        return "R"
    if s == "D" or s == "OPEN - D":
        return "D"
    return None  # bare "Open" or unknown


def flip_label(row):
    """Return 'D→R', 'R→D', or ''."""
    held = held_party(row["incumbent_party"])
    pred = row["predicted_winner"]
    if held is None or held == pred:
        return ""
    return f"{held}→{pred}"


results["flip"] = results.apply(flip_label, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════
def chamber_card(office, sub, small=False):
    """Card showing final chamber composition (Senate/Gov only) + 2026 breakdown."""
    r_wins  = int((sub["predicted_winner"] == "R").sum())
    d_wins  = int(len(sub) - r_wins)
    tossups = int(((sub["win_prob_R"] > 0.40) & (sub["win_prob_R"] < 0.60)).sum())
    r_flips = int((sub["flip"] == "D→R").sum())
    d_flips = int((sub["flip"] == "R→D").sum())

    st.markdown(f"### {office}")

    # Final composition — only for Senate & Governor (House needs full CSV first)
    if office in ("Senate", "Governor"):
        if office == "Senate":
            total_r = r_wins + SENATE_CARRYOVER_R
            total_d = d_wins + SENATE_CARRYOVER_D
            control = "R" if total_r >= SENATE_MAJORITY else "D"
            maj_line = f"{SENATE_MAJORITY} needed for majority"
        else:
            total_r = r_wins + GOV_CARRYOVER_R
            total_d = d_wins + GOV_CARRYOVER_D
            control = "R" if total_r > total_d else "D"
            maj_line = "50 total governorships"

        control_color = R_COLOR if control == "R" else D_COLOR
        st.markdown(
            f"<div style='margin-bottom:8px; font-size:1.9rem; font-weight:700; letter-spacing:-0.5px;'>"
            f"<span style='color:{R_COLOR}'>{total_r}R</span> · "
            f"<span style='color:{D_COLOR}'>{total_d}D</span>"
            f"</div>"
            f"<div style='margin-top:-6px; margin-bottom:14px; color:{control_color}; font-weight:600; font-size:0.95rem;'>"
            f"{'Republican' if control == 'R' else 'Democratic'} control · "
            f"<span style='color:#9c9c9c; font-weight:400;'>{maj_line}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        subhead = "2026 RACES"
    else:
        subhead = None  # House: skip the composition; just show the 2026 breakdown

    if subhead:
        st.markdown(
            f"<div style='color:#6a6a6a; font-size:0.85rem; font-weight:500;'>{subhead}</div>",
            unsafe_allow_html=True,
        )

    # 2026-race breakdown (2x2 grid)
    c1, c2 = st.columns(2)
    c1.metric("R wins", r_wins)
    c2.metric("D wins", d_wins)
    c3, c4 = st.columns(2)
    c3.metric("Tossups", tossups)
    c4.metric("Flips", f"{d_flips}D / {r_flips}R",
              help="R→D pickups on the D side, D→R pickups on the R side")
    st.caption(f"{len(sub)} 2026 races modeled")


def ratings_bar(sub_results, title=None):
    """Stacked horizontal bar showing rating distribution for a subset."""
    fig = go.Figure()
    counts = sub_results["rating"].value_counts().reindex(RATING_ORDER, fill_value=0)
    for rat in RATING_ORDER:
        n = int(counts[rat])
        fig.add_trace(go.Bar(
            name=rat,
            y=[""],
            x=[n],
            orientation="h",
            marker_color=rating_color(rat),
            text=[str(n) if n > 0 else ""],
            textposition="inside",
            textfont=dict(color="white", size=13),
            hovertemplate=f"{rat}: {n}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        height=140,
        margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        title=title or "",
        legend=dict(orientation="h", yanchor="bottom", y=-1.2, xanchor="center", x=0.5),
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(showticklabels=False),
        plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
    )
    return fig


def race_table(sub_results):
    """Sortable/filterable race table."""
    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    tiers = sorted(sub_results["tier"].dropna().unique().tolist())
    tier_filter = fcol1.multiselect("Tier", tiers, default=tiers, key=f"tier_{sub_results['office'].iloc[0]}")
    rating_filter = fcol2.multiselect("Rating", RATING_ORDER, default=RATING_ORDER,
                                       key=f"rating_{sub_results['office'].iloc[0]}")
    search = fcol3.text_input("Search state or race_id", value="",
                              key=f"search_{sub_results['office'].iloc[0]}")

    filtered = sub_results[
        sub_results["tier"].isin(tier_filter) &
        sub_results["rating"].isin(rating_filter)
    ]
    if search:
        s = search.upper()
        filtered = filtered[
            filtered["race_id"].str.upper().str.contains(s) |
            filtered["state"].str.upper().str.contains(s)
        ]

    filtered = filtered.assign(_c=(filtered["win_prob_R"] - 0.5).abs()).sort_values("_c")
    table = filtered[[
        "race_id", "state", "rating", "flip", "median_margin", "p5", "p95",
        "win_prob_R", "win_prob_D", "tier"
    ]].copy()
    table["flip"] = table["flip"].apply(lambda x: f"🔄 {x}" if x else "")
    table["win_prob_R"] = (table["win_prob_R"] * 100).round(1).astype(str) + "%"
    table["win_prob_D"] = (table["win_prob_D"] * 100).round(1).astype(str) + "%"

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "race_id":       st.column_config.TextColumn("Race", width="small"),
            "state":         st.column_config.TextColumn("State", width="small"),
            "rating":        st.column_config.TextColumn("Rating", width="small"),
            "flip":          st.column_config.TextColumn("Flip", width="small",
                                help="Marked if predicted winner differs from party currently holding the seat."),
            "median_margin": st.column_config.NumberColumn("Median", format="%+.1f"),
            "p5":            st.column_config.NumberColumn("P5", format="%+.1f"),
            "p95":           st.column_config.NumberColumn("P95", format="%+.1f"),
            "win_prob_R":    st.column_config.TextColumn("R %"),
            "win_prob_D":    st.column_config.TextColumn("D %"),
            "tier":          st.column_config.NumberColumn("Tier", width="small"),
        },
        height=460,
    )
    st.caption(f"Showing {len(filtered)} of {len(sub_results)} races. Sorted by competitiveness.")


def race_detail(sub_results, office_label):
    """Race detail selector for one chamber."""
    st.markdown("### Race Detail")
    race_choices = ["— Select a race —"] + sorted(sub_results["race_id"].tolist())
    picked = st.selectbox("Pick a race", race_choices, key=f"detail_{office_label}")
    if picked == "— Select a race —":
        return

    r = sub_results[sub_results["race_id"] == picked].iloc[0]
    dcol1, dcol2 = st.columns([2, 1])

    with dcol1:
        # Regenerate distribution for the histogram
        input_df = pd.read_csv(CSV_PATH)
        row_input = input_df[input_df["race_id"] == picked].iloc[0]
        settings = OFFICE_SETTINGS[r["office"]]
        baseline, swing_unc, _ = _baseline_and_unc(row_input, settings)
        # Match the engine: preserve per-race dampening ratio when overriding
        csv_swing_val = _to_float(row_input["expected_swing"])
        if override_national is not None and CSV_NATIONAL_SWING != 0:
            ratio = csv_swing_val / CSV_NATIONAL_SWING
            swing = override_national * ratio
        else:
            swing = csv_swing_val
        scandal    = _to_float(row_input["scandal"])
        incumbency = _to_float(row_input["incumbency_factor"])
        quality    = _to_float(row_input["candidate_quality"])
        spending   = _to_float(row_input["spending_effect"])

        rng = np.random.default_rng(seed)
        qual_unc = settings["qual_base"] + abs(scandal) * 0.5
        drawn_swings  = rng.normal(swing, swing_unc, size=10_000)
        drawn_quality = rng.normal(scandal + incumbency + quality + spending, qual_unc, size=10_000)
        sims = baseline + drawn_swings + drawn_quality
        r_mask = sims > 0

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sims[r_mask],  nbinsx=60, marker_color=R_COLOR, opacity=0.85, name="R wins"))
        fig.add_trace(go.Histogram(x=sims[~r_mask], nbinsx=60, marker_color=D_COLOR, opacity=0.85, name="D wins"))
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
        fig.add_vline(x=r["median_margin"], line_color="purple", line_width=2,
                      annotation_text=f"Median: {r['median_margin']:+.1f}")
        fig.update_layout(
            barmode="overlay",
            title=f"{picked} — Distribution of Simulated Margins",
            xaxis_title="Margin (points)   ← D wins  |  R wins →",
            yaxis_title="Simulated worlds",
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{office_label}_{picked}")

    with dcol2:
        st.markdown(f"#### {picked}")
        st.markdown(
            f"**Office:** {r['office']}  \n"
            f"**State:** {r['state']}  \n"
            f"**Tier:** {int(r['tier']) if pd.notna(r['tier']) else '—'}"
        )
        color = rating_color(r["rating"])
        badges = (
            f"<div style='background:{color}; color:white; padding:8px 14px; "
            f"border-radius:6px; display:inline-block; font-weight:600; margin-top:8px; margin-right:8px;'>"
            f"{r['rating']}</div>"
        )
        if r["flip"]:
            badges += (
                f"<div style='background:#7a1fa2; color:white; padding:8px 14px; "
                f"border-radius:6px; display:inline-block; font-weight:600; margin-top:8px;'>"
                f"🔄 FLIP: {r['flip']}</div>"
            )
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown(f"**Median margin:** {r['median_margin']:+.2f}")
        st.markdown(f"**90% range:** [{r['p5']:+.1f}, {r['p95']:+.1f}]")
        st.markdown(f"**R win prob:** {r['win_prob_R']*100:.1f}%  \n**D win prob:** {r['win_prob_D']*100:.1f}%")

        st.markdown("---")
        st.markdown("**Inputs**")
        st.markdown(f"- Baseline: **{r['baseline_used']:+.2f}** ({r['blend_info']})")
        st.markdown(f"- Swing: **{r['swing_used']:+.2f}** ± {r['swing_unc_used']:.2f}")
        st.markdown(f"- Incumbent party: {r['incumbent_party']}")

        if pd.notna(r["notes"]) and str(r["notes"]).strip():
            st.markdown("---")
            st.markdown("**Notes**")
            st.markdown(f"_{r['notes']}_")


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_home, tab_sen, tab_house, tab_gov = st.tabs(
    ["🏛️ Home", "🏛️ Senate", "🏛️ House", "🏛️ Governor"]
)


# ── HOME TAB ─────────────────────────────────────────────────────────────────
with tab_home:
    st.markdown("## Current Odds")

    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        chamber_card("Senate", results[results["office"] == "Senate"])
    with hc2:
        chamber_card("Governor", results[results["office"] == "Governor"])
    with hc3:
        chamber_card("House", results[results["office"] == "House"])

    st.markdown("### Ratings breakdown")
    all_bar_col1, all_bar_col2, all_bar_col3 = st.columns(3)
    with all_bar_col1:
        st.markdown("**Senate**")
        st.plotly_chart(ratings_bar(results[results["office"] == "Senate"]),
                        use_container_width=True, key="home_bar_sen")
    with all_bar_col2:
        st.markdown("**Governor**")
        st.plotly_chart(ratings_bar(results[results["office"] == "Governor"]),
                        use_container_width=True, key="home_bar_gov")
    with all_bar_col3:
        st.markdown("**House**")
        st.plotly_chart(ratings_bar(results[results["office"] == "House"]),
                        use_container_width=True, key="home_bar_house")

    st.markdown("---")

    # ── Generic Ballot Aggregator ──
    st.markdown("## Generic Ballot Poll Aggregator")
    st.markdown(
        "<div style='color:#6a6a6a; margin-top:-8px;'>"
        "Manually-entered generic-ballot polls, weighted <b>60% LV / 40% RV</b>, "
        "rolling 21-day average. Toggle the switch to feed the poll-derived swing "
        "into the model instead of the CSV's default."
        "</div>",
        unsafe_allow_html=True,
    )

    # Row 1: raw aggregate + projected GCB
    ag1, ag2 = st.columns([1, 1])

    with ag1:
        if poll_summary_d["current_avg"] is not None:
            avg_val = poll_summary_d["current_avg"]
            party = "D" if avg_val < 0 else "R"
            label = f"{party}+{abs(avg_val):.1f}"
            st.metric("Raw polling average (21d)", label,
                      help="LV/RV-weighted rolling average of margin. Negative = D lead.")
            if poll_swing_raw is not None:
                s_party = "D" if poll_swing_raw < 0 else "R"
                st.caption(
                    f"Implied swing vs 2024: **{s_party}+{abs(poll_swing_raw):.2f}**  \n"
                    f"(2024 reference: R+{REFERENCE_2024:.2f})"
                )
        else:
            st.metric("Raw polling average (21d)", "—")
            st.caption("Add polls below.")

    with ag2:
        proj = poll_summary_d.get("projected")
        if proj is not None:
            m = proj["proj_margin_r"]
            party = "D" if m < 0 else "R"
            label = f"{party}+{abs(m):.1f}"
            st.metric(
                f"Projected GCB (undecideds split {int(UNDECIDED_TO_D*100)}/{int(UNDECIDED_TO_R*100)} D)",
                label,
                help=f"After allocating undecideds (100 − D% − R%) at "
                     f"{UNDECIDED_TO_D:.0%} to Dems / {UNDECIDED_TO_R:.0%} to Reps.",
            )
            st.caption(
                f"Proj. D: **{proj['proj_dem']:.1f}%** · "
                f"Proj. R: **{proj['proj_rep']:.1f}%** · "
                f"Undecided pool: **{proj['undecided']:.1f}%**"
            )
            if poll_swing_proj is not None:
                s_party = "D" if poll_swing_proj < 0 else "R"
                st.caption(
                    f"Projected swing vs 2024: **{s_party}+{abs(poll_swing_proj):.2f}**"
                )
        else:
            st.metric("Projected GCB", "—")

    # Model is always fed by the projected GCB swing (when polls exist).
    # Per-race dampening ratios (e.g., -3.86 = -9.65 / 2.5) are preserved.
    st.caption(
        f"**Model swing:** using projected GCB (undecideds split "
        f"{int(UNDECIDED_TO_D*100)}/{int(UNDECIDED_TO_R*100)} D) · "
        f"CSV baseline: **{CSV_NATIONAL_SWING:+.2f}** · "
        f"Effective national swing: **{effective_national:+.2f}** · "
        f"Dampened races scale proportionally (e.g. {CSV_NATIONAL_SWING/2.5:+.2f} → "
        f"{effective_national/2.5:+.2f})"
    )

    # Fever chart
    fever = fever_series(polls_df, window_days=21)
    if len(fever) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fever["date"], y=fever["avg"],
            mode="lines", line=dict(width=3, color="#333"),
            fill="tozeroy",
            hovertemplate="%{x|%b %d}: %{y:+.2f}<extra></extra>",
            name="21d avg",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1,
                      annotation_text="Tie", annotation_position="top right")
        fig.update_layout(
            title="Generic ballot — 21-day rolling average (R+ scale)",
            xaxis_title="",
            yaxis_title="Margin (R − D)",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="fever_chart")
    elif len(polls_df) > 0:
        st.info("Need at least 2 days of polls to draw the fever chart.")

    # Add poll form
    with st.expander("➕ Add a new poll", expanded=False):
        with st.form("poll_form", clear_on_submit=True):
            fcol1, fcol2, fcol3 = st.columns(3)
            new_date = fcol1.date_input("Date")
            new_pollster = fcol2.text_input("Pollster", placeholder="e.g. Marquette")
            new_type = fcol3.selectbox("Sample type", ["LV", "RV"],
                                        help="Likely Voter (weighted 60%) or Registered Voter (40%)")
            f2col1, f2col2, f2col3 = st.columns(3)
            new_dem = f2col1.number_input("Dem %", min_value=0.0, max_value=100.0, value=48.0, step=0.5)
            new_rep = f2col2.number_input("Rep %", min_value=0.0, max_value=100.0, value=44.0, step=0.5)
            new_n = f2col3.number_input("Sample size", min_value=0, value=1000, step=100)
            new_notes = st.text_input("Notes", placeholder="Optional")
            submitted = st.form_submit_button("Add poll")
            if submitted:
                if not new_pollster.strip():
                    st.error("Please enter a pollster name.")
                else:
                    append_poll({
                        "date":         new_date.isoformat(),
                        "pollster":     new_pollster.strip(),
                        "sample_type":  new_type,
                        "dem_pct":      new_dem,
                        "rep_pct":      new_rep,
                        "sample_size":  int(new_n),
                        "notes":        new_notes.strip(),
                    }, path=POLLS_CSV)
                    st.cache_data.clear()
                    st.success(f"Added {new_pollster} poll from {new_date.isoformat()}. Reloading…")
                    st.rerun()

    # Poll table
    st.markdown("### Polls")
    if len(polls_df) == 0:
        st.caption("No polls yet. Use the form above to add one.")
    else:
        display = polls_df.copy()
        display = display.sort_values("date", ascending=False)
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["margin_r"] = display["margin_r"].round(1)
        display["weight"] = display["weight"]
        st.dataframe(
            display[["date", "pollster", "sample_type", "dem_pct", "rep_pct",
                     "margin_r", "sample_size", "weight", "notes"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "date":        st.column_config.TextColumn("Date", width="small"),
                "pollster":    st.column_config.TextColumn("Pollster"),
                "sample_type": st.column_config.TextColumn("Type", width="small"),
                "dem_pct":     st.column_config.NumberColumn("Dem %", format="%.1f"),
                "rep_pct":     st.column_config.NumberColumn("Rep %", format="%.1f"),
                "margin_r":    st.column_config.NumberColumn("Margin (R+)", format="%+.1f"),
                "sample_size": st.column_config.NumberColumn("N", format="%d"),
                "weight":      st.column_config.NumberColumn("Weight", format="%.1f", width="small"),
                "notes":       st.column_config.TextColumn("Notes"),
            },
            height=250,
        )
        st.caption(f"{len(polls_df)} polls loaded. Edit `data/polls_2026.csv` directly to bulk-modify.")


# ── SENATE TAB ───────────────────────────────────────────────────────────────
with tab_sen:
    sen = results[results["office"] == "Senate"]
    st.markdown("## Senate — 2026")
    chamber_card("Senate", sen)

    st.markdown("### Ratings")
    st.plotly_chart(ratings_bar(sen), use_container_width=True, key="tab_bar_sen")

    st.markdown("### Race List")
    race_table(sen)

    st.markdown("---")
    race_detail(sen, "senate")


# ── HOUSE TAB ────────────────────────────────────────────────────────────────
with tab_house:
    house = results[results["office"] == "House"]
    st.markdown("## House — 2026")
    chamber_card("House", house)

    st.markdown("### Ratings")
    st.plotly_chart(ratings_bar(house), use_container_width=True, key="tab_bar_house")

    st.markdown("### Race List")
    race_table(house)

    st.markdown("---")
    race_detail(house, "house")


# ── GOVERNOR TAB ─────────────────────────────────────────────────────────────
with tab_gov:
    gov = results[results["office"] == "Governor"]
    st.markdown("## Governor — 2026")
    chamber_card("Governor", gov)

    st.markdown("### Ratings")
    st.plotly_chart(ratings_bar(gov), use_container_width=True, key="tab_bar_gov")

    st.markdown("### Race List")
    race_table(gov)

    st.markdown("---")
    race_detail(gov, "gov")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='color:#9c9c9c; font-size:0.85rem;'>"
    f"CarlosNeedle · Backtested on 2018 &amp; 2022 (≈85% accuracy across 126 races). "
    f"A hobbyist model — not a professional forecast."
    f"</div>",
    unsafe_allow_html=True,
)
