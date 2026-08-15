"""
Hex-tile cartogram for the US House.

Each of the 435 congressional districts is one hex. States are arranged in an
approximate US shape; each state's hexes form a compact block whose aspect
ratio loosely matches the state's geographic footprint.

v2 layout: tighter packing, state shapes better approximate real geography.
"""

import pandas as pd
import plotly.graph_objects as go

# Per-state block layout:
#   (origin_x, origin_y, cols, rows) — bottom-left origin.
#   cols × rows must be ≥ the state's district count.
#   Districts fill left-to-right, bottom-to-top.
#
# Origins were hand-placed to approximate US geography.
STATE_BLOCKS = {
    # Non-contiguous
    "AK": (0,  24, 1, 1),
    "HI": (0,  0,  2, 1),

    # West Coast column
    "WA": (2,  22, 3, 4),   # 12 for 10 (tall block)
    "OR": (2,  19, 3, 2),   # 6
    "CA": (2,  8,  4, 11),  # 44 slots for 52 → need 5x11=55
}
# Rewriting properly below; the dict above is scaffolding to overwrite:
STATE_BLOCKS = {
    # ── Row: Non-contiguous ──
    "AK": (0,  22, 1, 1),
    "HI": (0,  0,  2, 1),

    # ── West Coast (left column) ──
    "CA": (2,  8,  5, 11),   # 52 in 55
    "OR": (2,  20, 3, 2),    # 6
    "WA": (2,  23, 3, 4),    # 10 in 12

    # ── Mountain West ──
    "AZ": (8,  6,  3, 3),    # 9
    "NV": (8,  10, 2, 2),    # 4
    "UT": (8,  13, 2, 2),    # 4
    "ID": (8,  18, 2, 2),    # 2 in 4
    "MT": (8,  22, 2, 2),    # 2 in 4

    "NM": (11, 6,  2, 2),    # 3 in 4
    "CO": (11, 10, 3, 3),    # 8 in 9
    "WY": (11, 14, 2, 2),    # 1 in 4
    "SD": (11, 18, 2, 2),    # 1 in 4
    "ND": (11, 22, 2, 2),    # 1 in 4

    # ── Great Plains ──
    "TX": (15, 0,  7, 6),    # 38 in 42
    "OK": (15, 7,  3, 2),    # 5 in 6
    "KS": (15, 10, 3, 2),    # 4 in 6
    "NE": (15, 13, 3, 2),    # 3 in 6
    "MN": (15, 22, 3, 3),    # 8 in 9
    "IA": (15, 17, 3, 3),    # 4 in 9

    # ── South Central & Mid-South ──
    "LA": (19, 7,  3, 2),    # 6
    "AR": (19, 10, 3, 2),    # 4 in 6
    "MO": (19, 13, 3, 3),    # 8 in 9
    "WI": (19, 22, 3, 3),    # 8 in 9

    # ── Great Lakes / Midwest ──
    "IL": (23, 10, 4, 5),    # 17 in 20
    "MS": (23, 7,  2, 2),    # 4
    "AL": (26, 7,  3, 3),    # 7 in 9
    "MI": (23, 22, 4, 4),    # 13 in 16
    "IN": (23, 17, 3, 3),    # 9

    # ── Ohio Valley & Southeast ──
    "TN": (28, 10, 4, 3),    # 9 in 12
    "KY": (28, 13, 4, 2),    # 6 in 8
    "OH": (28, 17, 4, 4),    # 15 in 16
    "GA": (30, 4,  3, 5),    # 14 in 15
    "FL": (34, 0,  4, 7),    # 28 in 28

    # ── Atlantic South ──
    "SC": (33, 7,  3, 3),    # 7 in 9  (moved up 1 to clear FL)
    "NC": (33, 10, 5, 3),    # 14 in 15
    "VA": (33, 14, 4, 3),    # 11 in 12
    "WV": (33, 17, 2, 1),    # 2

    # ── Mid-Atlantic ──
    "MD": (37, 14, 3, 3),    # 8 in 9  (moved right 1)
    "DE": (40, 14, 1, 1),    # 1  (moved right)
    "PA": (33, 22, 5, 4),    # 17 in 20

    # ── Northeast ──
    "NJ": (39, 18, 3, 4),    # 12
    "NY": (39, 22, 5, 6),    # 26 in 30
    "CT": (44, 20, 3, 2),    # 5 in 6  (shifted right)
    "RI": (44, 22, 2, 1),    # 2  (moved to clear NY)
    "MA": (46, 22, 4, 3),    # 9 in 12
    "VT": (45, 26, 1, 1),    # 1
    "NH": (47, 26, 1, 2),    # 2
    "ME": (49, 25, 1, 2),    # 2
}


def compute_positions(house_df):
    """Compute (x, y) for each district hex."""
    rows = []
    for state, (origin_x, origin_y, cols, _rows) in STATE_BLOCKS.items():
        state_races = house_df[house_df["state"] == state].sort_values("district")
        for i, (_, race) in enumerate(state_races.iterrows()):
            col = i % cols
            row = i // cols
            # Hex offset: odd rows shifted right by 0.5 for tight hex packing
            x = origin_x + col + (0.5 if row % 2 else 0)
            y = origin_y + row * 0.87
            rows.append({
                "race_id": race["race_id"],
                "state": state,
                "district": race["district"],
                "x": x, "y": y,
                "median_margin": race["median_margin"],
                "rating": race["rating"],
                "p5": race["p5"], "p95": race["p95"],
                "win_prob_R": race["win_prob_R"],
                "win_prob_D": race["win_prob_D"],
                "predicted_winner": race["predicted_winner"],
                "flip": race["flip"],
            })
    return pd.DataFrame(rows)


def _color_for(m):
    if m >= 15:  return "#a01a1c"
    if m >  7:   return "#cf3d3f"
    if m >  2.5: return "#e88b8c"
    if m >= 0:   return "#f5c8c9"
    if m > -2.5: return "#c8d5f5"
    if m > -7:   return "#98b3ee"
    if m >= -15: return "#4e7be3"
    return "#1e4bab"


def render_hex_map(house_df, bg_color="#ffffff"):
    pos = compute_positions(house_df)
    pos["color"] = pos["median_margin"].apply(_color_for)

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
            size=22,
            color=pos["color"],
            line=dict(color="white", width=1.2),
        ),
        hovertext=hovertext,
        hoverinfo="text",
        showlegend=False,
    ))

    # State labels — placed above each state's block
    for state, (ox, oy, cols, rows) in STATE_BLOCKS.items():
        cx = ox + (cols - 1) / 2
        cy = oy + rows * 0.87 + 0.15
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{state}</b>",
            showarrow=False,
            font=dict(size=10, color="#333"),
        )

    fig.update_layout(
        height=680,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
