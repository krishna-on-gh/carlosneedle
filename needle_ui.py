"""
Streamlit UI for the Election Night needle tab.

Renders a grid of race cards (mini-needles) and a detail panel for the
selected race with a paste-box for county results.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from needle import compute_needle, load_turnout_ratio
from needle_parser import parse_paste
from needle_priors import load_priors

R_COLOR = "#cf3d3f"
D_COLOR = "#4e7be3"
TOSSUP_COLOR = "#9c9c9c"


# ---- Session-state helpers ----
def _paste_key(race_id: str) -> str:
    return f"needle_paste::{race_id}"


def _get_paste(race_id: str) -> str:
    return st.session_state.get(_paste_key(race_id), "")


def _set_paste(race_id: str, text: str) -> None:
    st.session_state[_paste_key(race_id)] = text


# ---- Rendering: single race ----
def _win_prob_pct(p_dem_win: float) -> str:
    if p_dem_win >= 0.995: return "D 100%"
    if p_dem_win <= 0.005: return "R 100%"
    if p_dem_win >= 0.5:
        return f"D {p_dem_win*100:.0f}%"
    return f"R {(1-p_dem_win)*100:.0f}%"


def _margin_label(m: float) -> str:
    """+3.2 -> 'R+3.2', -3.2 -> 'D+3.2'."""
    if abs(m) < 0.05: return "Tied"
    who = "R" if m > 0 else "D"
    return f"{who}+{abs(m):.1f}"


def _needle_figure(prior_mean, prior_sigma, result, race_label: str,
                   is_mini: bool = False) -> go.Figure:
    """Horizontal band plot showing posterior distribution."""
    # X range: automatic based on the extremes of prior + posterior
    lo_bound = min(prior_mean - 2*prior_sigma, result.p5) - 2
    hi_bound = max(prior_mean + 2*prior_sigma, result.p95) + 2
    xmin = min(lo_bound, -15)
    xmax = max(hi_bound, 15)

    fig = go.Figure()

    # D zone shading (negative half)
    fig.add_shape(type="rect", xref="x", yref="paper", x0=xmin, x1=0, y0=0, y1=1,
                  fillcolor="rgba(78,123,227,0.10)", line=dict(width=0), layer="below")
    # R zone shading (positive half)
    fig.add_shape(type="rect", xref="x", yref="paper", x0=0, x1=xmax, y0=0, y1=1,
                  fillcolor="rgba(207,61,63,0.10)", line=dict(width=0), layer="below")

    # Posterior 90% band as a thick horizontal bar
    fig.add_shape(type="rect", xref="x", yref="paper",
                  x0=result.p5, x1=result.p95, y0=0.42, y1=0.58,
                  fillcolor="rgba(80,80,80,0.55)", line=dict(width=0))

    # Median needle (vertical line)
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=result.p50, x1=result.p50, y0=0.20, y1=0.80,
                  line=dict(color="black", width=3))

    # Zero (tie) line
    fig.add_shape(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1,
                  line=dict(color="#333", width=1, dash="dot"))

    height = 90 if is_mini else 180
    fig.update_layout(
        xaxis=dict(range=[xmin, xmax], showgrid=False, zeroline=False,
                   tickvals=[xmin+2, 0, xmax-2] if is_mini else None,
                   ticktext=(["D", "Tie", "R"] if is_mini else None),
                   fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def _render_race_detail(race_row, priors_row):
    """Detail view for one race: paste box, full needle, numbers."""
    race_id = race_row["race_id"]
    state = race_row["state"]
    dem = str(race_row.get("dem_last_name") or "").strip()
    rep = str(race_row.get("rep_last_name") or "").strip()

    st.subheader(race_row["display_name"])

    # ---- Left column: paste box. Right: results ----
    left, right = st.columns([1, 1])

    with left:
        st.markdown("##### Paste county results")
        if not dem or not rep:
            st.warning(f"Fill in dem_last_name and rep_last_name for {race_id} "
                       f"in needle_races_2026.csv before pasting.")
        placeholder = ("County  Trump +5  45,000  75%\n"
                       "County  Harris +12  30,000  60%\n"
                       "...")
        paste = st.text_area(
            "Paste from NYT results page (or similar). Click Update below to refresh.",
            value=_get_paste(race_id),
            height=250,
            placeholder=placeholder,
            key=f"paste_area::{race_id}",
        )
        if st.button("Update needle", key=f"update::{race_id}"):
            _set_paste(race_id, paste)
            st.rerun()

    with right:
        prior_mean = float(priors_row["prior_mean"])
        prior_sigma = float(priors_row["prior_sigma"])
        try:
            turnout = load_turnout_ratio(state)
            if dem and rep and _get_paste(race_id).strip():
                parsed = parse_paste(_get_paste(race_id), dem, rep)
                result = compute_needle(prior_mean, prior_sigma,
                                        parsed.rows, state, turnout_ratio=turnout)
                if parsed.skipped:
                    with st.expander(f"Paste parser: skipped {len(parsed.skipped)} rows"):
                        for line, reason in parsed.skipped[:20]:
                            st.text(f"[{reason}]  {line[:100]}")
            else:
                # No paste yet -> show prior as the needle
                result = compute_needle(prior_mean, prior_sigma,
                                        pd.DataFrame(), state, turnout_ratio=turnout)
        except Exception as e:
            st.error(f"Error computing needle: {e}")
            return

        # ---- Result summary metrics ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Median", _margin_label(result.p50))
        c2.metric("Winner odds", _win_prob_pct(result.p_dem_win))
        c3.metric("Vote in",
                  f"{result.reported_vote_total/result.expected_vote_total*100:.1f}%"
                  if result.expected_vote_total > 0 else "0%")

        st.plotly_chart(_needle_figure(prior_mean, prior_sigma, result,
                                       race_row["display_name"]),
                        use_container_width=True)

        st.markdown(
            f"<div style='color:#6a6a6a; font-size:0.9rem;'>"
            f"<b>90% band:</b> {_margin_label(result.p5)} to {_margin_label(result.p95)} "
            f"&nbsp; · &nbsp; "
            f"<b>Counties in:</b> {result.n_counties_reported} "
            f"&nbsp; · &nbsp; "
            f"<b>Prior:</b> {_margin_label(prior_mean)} ± {prior_sigma:.1f}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if result.state_swing_signal is not None:
            st.caption(
                f"State swing signal: {result.state_swing_signal:+.2f} "
                f"(SE {result.state_swing_se:.2f})"
            )

        for w in result.warnings:
            st.info(f"⚠️  {w}")


def _render_mini_card(race_row, priors_row):
    """Small grid card for one race — click sets the active race."""
    race_id = race_row["race_id"]
    state = race_row["state"]
    dem = str(race_row.get("dem_last_name") or "").strip()
    rep = str(race_row.get("rep_last_name") or "").strip()

    prior_mean = float(priors_row["prior_mean"])
    prior_sigma = float(priors_row["prior_sigma"])
    turnout = load_turnout_ratio(state)

    try:
        if dem and rep and _get_paste(race_id).strip():
            parsed = parse_paste(_get_paste(race_id), dem, rep)
            result = compute_needle(prior_mean, prior_sigma, parsed.rows,
                                    state, turnout_ratio=turnout)
            has_data = True
        else:
            result = compute_needle(prior_mean, prior_sigma, pd.DataFrame(),
                                    state, turnout_ratio=turnout)
            has_data = False
    except Exception:
        result = None
        has_data = False

    # Card border/tint based on race state
    if result is None:
        border = "#e6e6e6"
    elif result.p_dem_win >= 0.9:
        border = D_COLOR
    elif result.p_dem_win <= 0.1:
        border = R_COLOR
    else:
        border = TOSSUP_COLOR

    st.markdown(
        f"<div style='border:1.5px solid {border}; border-radius:8px; padding:6px 10px; "
        f"margin-bottom:4px; background:white;'>"
        f"<div style='font-weight:600; font-size:0.85rem;'>{race_row['display_name']}</div>"
        + (
            f"<div style='color:#6a6a6a; font-size:0.75rem;'>"
            f"{_margin_label(result.p50)} · {_win_prob_pct(result.p_dem_win)}"
            f"{' · LIVE' if has_data else ' · prior only'}"
            f"</div>"
            if result is not None else
            "<div style='color:#a00; font-size:0.75rem;'>error</div>"
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Open", key=f"open::{race_id}", use_container_width=True):
        st.session_state["needle_active_race"] = race_id
        st.rerun()


# ---- Main entry point ----
def render_needle_tab():
    st.markdown("## 🗳️ Election Night 2026")
    st.markdown(
        "<div style='color:#6a6a6a; font-size:1rem; margin-top:-8px;'>"
        "Live Bayesian needles for target Senate and Governor races. "
        "Pre-election forecast (prior) blends with pasted county results (likelihood) "
        "into a live posterior. Both narrow as more of the vote reports."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        priors = load_priors()
    except Exception as e:
        st.error(f"Could not load priors from results_2026.csv: {e}")
        return

    valid = priors.dropna(subset=["prior_mean", "prior_sigma"]).reset_index(drop=True)
    if len(valid) == 0:
        st.warning("No needle races found with valid priors.")
        return

    # ---- Active race selection ----
    active = st.session_state.get("needle_active_race", valid.iloc[0]["race_id"])
    if active not in valid["race_id"].values:
        active = valid.iloc[0]["race_id"]

    # ---- Two columns: grid on left, detail on right ----
    with st.expander("All races — mini needles", expanded=False):
        st.caption("Click any race to open its detail view below.")
        sen = valid[valid["office"] == "Senate"].sort_values("state")
        gov = valid[valid["office"] == "Governor"].sort_values("state")

        st.markdown("### Senate")
        cols = st.columns(4)
        for i, (_, row) in enumerate(sen.iterrows()):
            with cols[i % 4]:
                _render_mini_card(row, row)

        st.markdown("### Governor")
        cols = st.columns(4)
        for i, (_, row) in enumerate(gov.iterrows()):
            with cols[i % 4]:
                _render_mini_card(row, row)

    # ---- Race picker (fallback for those who don't want to expand grid) ----
    labels = {row["race_id"]: row["display_name"] for _, row in valid.iterrows()}
    picked = st.selectbox("Select race", options=list(labels.keys()),
                          format_func=lambda k: labels[k],
                          index=list(labels.keys()).index(active),
                          key="needle_race_picker")
    if picked != active:
        st.session_state["needle_active_race"] = picked
        st.rerun()

    st.markdown("---")

    # ---- Detail view for active race ----
    race_row = valid[valid["race_id"] == active].iloc[0]
    _render_race_detail(race_row, race_row)
