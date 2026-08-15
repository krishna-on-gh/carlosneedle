"""
Hex-tile cartogram for the US House.

Each of the 435 congressional districts is one hex. States are placed in an
approximate US-shaped grid; each state's districts fill a compact rectangle at
its origin.

Layout is v1 — geographically approximate, not pixel-perfect. Refine positions
in STATE_ORIGIN as needed.

Coordinate system:
- X grows east (right)
- Y grows north (up) — plotted with y-axis reversed so north is up
"""

from math import sqrt

import pandas as pd
import plotly.graph_objects as go


# Number of House districts per state (post-2022 redistricting; total = 435)
STATE_SEATS = {
    "AL": 7, "AK": 1, "AZ": 9, "AR": 4, "CA": 52, "CO": 8, "CT": 5, "DE": 1,
    "FL": 28, "GA": 14, "HI": 2, "ID": 2, "IL": 17, "IN": 9, "IA": 4, "KS": 4,
    "KY": 6, "LA": 6, "ME": 2, "MD": 8, "MA": 9, "MI": 13, "MN": 8, "MS": 4,
    "MO": 8, "MT": 2, "NE": 3, "NV": 4, "NH": 2, "NJ": 12, "NM": 3, "NY": 26,
    "NC": 14, "ND": 1, "OH": 15, "OK": 5, "OR": 6, "PA": 17, "RI": 2, "SC": 7,
    "SD": 1, "TN": 9, "TX": 38, "UT": 4, "VT": 1, "VA": 11, "WA": 10, "WV": 2,
    "WI": 8, "WY": 1,
}

# State grid origins (x, y). Grid is roughly geographic — states are placed
# so their districts don't overlap neighbors much. Districts are laid out
# within each state's rectangular block starting at these origins.
#
# The (width, height) below controls internal district arrangement.
# Some big states get custom shapes to fit their footprint reasonably.

STATE_LAYOUT = {
    # state: (origin_x, origin_y, width_in_hexes, height_in_hexes)
    # Small blocks first (1 district): centered single hex
    "AK": (0, 6),   # far NW (own cell)
    "HI": (2, 0),   # far SW
    "ME": (30, 8),  "NH": (29, 7), "VT": (27, 7), "MA": (27, 6),
    "RI": (30, 6),  "CT": (25, 6), "NJ": (25, 5), "DE": (25, 4),
    "MD": (23, 3),  "SC": (22, 2),
    "ND": (14, 8),  "SD": (14, 7), "WY": (10, 6),
    # Two districts
    # ...  (all handled below in code)
}

# State ORIGIN (x, y) — bottom-left of the state's block.
# Width and internal layout computed based on district count.
STATE_ORIGIN = {
    "AK": (0, 8),
    "HI": (2, 0),
    "WA": (2, 8),  "OR": (2, 7),  "CA": (2, 3),
    "ID": (5, 7),  "NV": (5, 5),  "UT": (5, 4),  "AZ": (5, 2),
    "MT": (8, 8),  "WY": (8, 7),  "CO": (8, 5),  "NM": (8, 3),
    "ND": (11, 8), "SD": (11, 7), "NE": (11, 6), "KS": (11, 5),  "OK": (11, 4), "TX": (11, 1),
    "MN": (14, 8), "IA": (14, 6), "MO": (14, 5), "AR": (14, 4),  "LA": (14, 2),
    "WI": (17, 8), "IL": (17, 6), "MS": (17, 3), "AL": (17, 2),
    "MI": (20, 8), "IN": (20, 6), "KY": (20, 5), "TN": (20, 4),  "GA": (20, 2), "FL": (24, 1),
    "OH": (23, 7), "WV": (23, 5), "NC": (23, 4), "SC": (23, 3),
    "PA": (26, 7), "VA": (26, 5), "MD": (28, 4), "DE": (30, 4),
    "NY": (26, 8),
    "NJ": (30, 6),  "CT": (30, 7),  "RI": (32, 7),
    "MA": (30, 8),  "VT": (28, 9),  "NH": (30, 9),  "ME": (32, 9),
}


def _district_grid(n):
    """Compact grid dimensions for n districts. Return (cols, rows)."""
    if n == 1: return (1, 1)
    if n == 2: return (2, 1)
    if n == 3: return (3, 1)
    if n == 4: return (2, 2)
    if n <= 6: return (3, 2)
    if n <= 9: return (3, 3)
    if n <= 12: return (4, 3)
    if n <= 16: return (4, 4)
    if n <= 20: return (5, 4)
    if n <= 30: return (6, 5)
    if n <= 42: return (7, 6)
    if n <= 56: return (8, 7)
    return (int(sqrt(n)) + 1, int(sqrt(n)) + 1)


def compute_positions(house_df):
    """Compute (x, y) for each district. Returns DataFrame with x,y columns."""
    rows = []
    for state, seats in STATE_SEATS.items():
        origin_x, origin_y = STATE_ORIGIN[state]
        cols, _rows = _district_grid(seats)
        state_races = house_df[house_df["state"] == state].sort_values("district")
        # Fill left-to-right, bottom-to-top
        for i, (_, race) in enumerate(state_races.iterrows()):
            col = i % cols
            row = i // cols
            # Hex offset: odd rows shifted right by 0.5
            x = origin_x + col + (0.5 if row % 2 else 0)
            y = origin_y + row * 0.86  # sqrt(3)/2 for tight hex packing
            rows.append({
                "race_id": race["race_id"],
                "state": state,
                "district": race["district"],
                "x": x,
                "y": y,
                "median_margin": race["median_margin"],
                "rating": race["rating"],
                "p5": race["p5"],
                "p95": race["p95"],
                "win_prob_R": race["win_prob_R"],
                "win_prob_D": race["win_prob_D"],
                "predicted_winner": race["predicted_winner"],
                "flip": race["flip"],
            })
    return pd.DataFrame(rows)


def render_hex_map(house_df, bg_color="#ffffff"):
    """Return a Plotly figure of the House hex cartogram."""
    pos = compute_positions(house_df)

    # Color mapping (matches rating palette)
    def color_for(margin):
        m = margin
        if m >= 15: return "#a01a1c"    # Safe R
        if m > 7:   return "#cf3d3f"    # Likely R
        if m > 2.5: return "#e88b8c"    # Lean R
        if m >= 0:  return "#f5c8c9"    # Tossup - Tilt R
        if m > -2.5: return "#c8d5f5"   # Tossup - Tilt D
        if m > -7:   return "#98b3ee"   # Lean D
        if m >= -15: return "#4e7be3"   # Likely D
        return "#1e4bab"                # Safe D

    pos["color"] = pos["median_margin"].apply(color_for)

    hovertext = []
    for _, r in pos.iterrows():
        flip_str = f"<br><b>🔄 FLIP: {r['flip']}</b>" if r["flip"] else ""
        win_pct = max(r["win_prob_R"], r["win_prob_D"]) * 100
        hovertext.append(
            f"<b>{r['race_id']}</b><br>"
            f"Rating: <b>{r['rating']}</b><br>"
            f"Median margin: <b>{r['median_margin']:+.1f}</b><br>"
            f"90% range: [{r['p5']:+.1f}, {r['p95']:+.1f}]<br>"
            f"Predicted: <b>{r['predicted_winner']}</b> ({win_pct:.0f}%)"
            f"{flip_str}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pos["x"], y=pos["y"],
        mode="markers",
        marker=dict(
            symbol="hexagon",
            size=18,
            color=pos["color"],
            line=dict(color="white", width=1),
        ),
        hovertext=hovertext,
        hoverinfo="text",
        showlegend=False,
    ))

    # State labels
    for state, seats in STATE_SEATS.items():
        origin_x, origin_y = STATE_ORIGIN[state]
        cols, rows = _district_grid(seats)
        cx = origin_x + (cols - 1) / 2
        cy = origin_y + (rows * 0.86) + 0.55
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{state}</b>",
            showarrow=False,
            font=dict(size=10, color="#333"),
        )

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
