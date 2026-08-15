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
    # ── Non-contiguous ──
    "HI": (0,  0,  2, 1),
    "AK": (0,  26, 1, 1),

    # ── West Coast column ──
    "CA": (2,  8,  5, 11),   # 52 in 55, cols 2-6 rows 8-18
    "OR": (2,  20, 3, 2),    # rows 20-21 (gap row 19 from CA)
    "WA": (2,  23, 3, 4),    # rows 23-26 (gap row 22 from OR)

    # ── Mountain West (col 8-9 buffer from CA at col 6) ──
    "AZ": (8,  6,  3, 3),    # cols 8-10 rows 6-8
    "NV": (8,  10, 2, 2),    # rows 10-11 (gap row 9 from AZ)
    "UT": (8,  13, 2, 2),    # rows 13-14 (gap row 12)
    "ID": (8,  16, 2, 2),    # rows 16-17 (gap row 15)
    "MT": (8,  19, 2, 2),    # rows 19-20 (gap row 18)

    # NM/CO/WY/SD/ND column (col 12+, gap col 11 from AZ/NV/etc.)
    "NM": (12, 6,  2, 2),    # rows 6-7
    "CO": (12, 10, 3, 3),    # rows 10-12
    "WY": (12, 14, 2, 2),    # rows 14-15
    "SD": (12, 17, 2, 2),    # rows 17-18
    "ND": (12, 20, 2, 2),    # rows 20-21

    # ── Great Plains south (TX huge block) ──
    "TX": (16, 0,  7, 6),    # cols 16-22 rows 0-5
    "OK": (16, 8,  3, 2),    # rows 8-9 (gap rows 6-7 from TX)
    "KS": (16, 11, 3, 2),    # rows 11-12
    "NE": (16, 14, 3, 2),    # rows 14-15
    "IA": (16, 17, 3, 2),    # rows 17-18
    "MN": (16, 20, 3, 3),    # rows 20-22

    # ── South-Central / Deep South (col 24+, gap col 23 from OK/etc.) ──
    "LA": (24, 8,  3, 2),    # rows 8-9
    "AR": (24, 11, 3, 2),    # rows 11-12
    "MO": (24, 14, 3, 3),    # rows 14-16
    "WI": (24, 20, 3, 3),    # rows 20-22

    # MS / AL (south-east of LA)
    "MS": (28, 8,  2, 2),    # cols 28-29 rows 8-9 (gap col 27 from LA)
    "AL": (31, 8,  3, 3),    # cols 31-33 rows 8-10 (gap col 30 from MS)

    # ── Illinois cluster (col 28+, gap 27 from MO) ──
    "IL": (28, 12, 4, 5),    # cols 28-31 rows 12-16
    "IN": (33, 12, 3, 3),    # cols 33-35 rows 12-14 (gap col 32 from IL)
    "MI": (28, 24, 4, 4),    # cols 28-31 rows 24-27 (gap rows 17-19 from IL, gap col 27 from WI)

    # ── Ohio Valley / Southeast (col 33+) ──
    "KY": (33, 16, 4, 2),    # cols 33-36 rows 16-17 (gap row 15 from IN, but wait IN ends row 14, gap row 15, KY row 16)
    "TN": (33, 19, 4, 3),    # cols 33-36 rows 19-21 (gap row 18)
    "OH": (33, 24, 4, 4),    # cols 33-36 rows 24-27 (gap rows 22-23 from TN)

    # ── South Atlantic (col 35+, gap col 34 from AL/GA/etc.) ──
    "GA": (35, 4,  3, 5),    # cols 35-37 rows 4-8 (gap col 34 from AL)
    "SC": (39, 8,  3, 3),    # cols 39-41 rows 8-10 (gap col 38 from GA)
    "NC": (38, 12, 5, 3),    # cols 38-42 rows 12-14 (gap row 11 from SC)
    "VA": (38, 16, 4, 3),    # cols 38-41 rows 16-18 (gap row 15 from NC)
    "WV": (38, 20, 2, 1),    # cols 38-39 row 20 (gap row 19 from VA)

    # ── Florida ──
    "FL": (40, 0,  4, 7),    # cols 40-43 rows 0-6 (gap col 39 from GA (37); actually GA ends col 37, FL col 40, cols 38-39 gap; FL top row 6, SC bottom row 8, gap row 7)

    # ── Pennsylvania & Mid-Atlantic (col 38+ north) ──
    "PA": (38, 24, 5, 4),    # cols 38-42 rows 24-27 (gap rows 22-23 from WV)
    "MD": (41, 20, 3, 3),    # cols 41-43 rows 20-22 (gap col 40 from WV)
    "DE": (45, 20, 1, 1),    # col 45 row 20 (gap col 44 from MD)

    # ── Northeast ──
    "NJ": (45, 22, 3, 4),    # cols 45-47 rows 22-25 (gap row 21 from DE)
    "NY": (44, 26, 5, 6),    # cols 44-48 rows 26-31 (gap row 25? NJ ends 25, NY starts 26 - touching)
    #   Move NY up 1 so row 25 gap
    "CT": (49, 24, 3, 2),    # cols 49-51 rows 24-25 (gap col 48 from NJ end col 47; NJ ends 47, CT 49, gap col 48)
    "RI": (49, 22, 2, 1),    # cols 49-50 row 22 (gap row 23 from CT)
    "MA": (53, 24, 4, 3),    # cols 53-56 rows 24-26 (gap col 52 from CT)
    "VT": (50, 28, 1, 1),    # col 50 row 28
    "NH": (52, 28, 1, 2),    # col 52 rows 28-29 (gap col 51 from VT)
    "ME": (54, 28, 1, 2),    # col 54 rows 28-29 (gap col 53 from NH)
}
# Push NY up 1 so it doesn't touch NJ
STATE_BLOCKS["NY"] = (44, 27, 5, 6)


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


def _hover(r):
    flip_str = f"<br><b>🔄 FLIP: {r['flip']}</b>" if r["flip"] else ""
    win_pct = max(r["win_prob_R"], r["win_prob_D"]) * 100
    return (
        f"<b>{r['race_id']}</b><br>"
        f"Rating: <b>{r['rating']}</b><br>"
        f"Median margin: <b>{r['median_margin']:+.1f}</b><br>"
        f"90% range: [{r['p5']:+.1f}, {r['p95']:+.1f}]<br>"
        f"Predicted: <b>{r['predicted_winner']}</b> ({win_pct:.0f}%)"
        f"{flip_str}"
    )


def render_hex_map(house_df, bg_color="#ffffff"):
    pos = compute_positions(house_df)
    pos["color"] = pos["median_margin"].apply(_color_for)
    pos["is_flip"] = pos["flip"].astype(bool) & (pos["flip"] != "")

    fig = go.Figure()

    # Non-flipped hexes: standard thin white border
    non_flip = pos[~pos["is_flip"]]
    if len(non_flip):
        fig.add_trace(go.Scatter(
            x=non_flip["x"], y=non_flip["y"],
            mode="markers",
            marker=dict(
                symbol="hexagon",
                size=22,
                color=non_flip["color"],
                line=dict(color="white", width=1.2),
            ),
            hovertext=[_hover(r) for _, r in non_flip.iterrows()],
            hoverinfo="text",
            showlegend=False,
            name="races",
        ))

    # Flipped hexes: thick black border to signal the flip
    flip = pos[pos["is_flip"]]
    if len(flip):
        fig.add_trace(go.Scatter(
            x=flip["x"], y=flip["y"],
            mode="markers",
            marker=dict(
                symbol="hexagon",
                size=22,
                color=flip["color"],
                line=dict(color="#000000", width=3),
            ),
            hovertext=[_hover(r) for _, r in flip.iterrows()],
            hoverinfo="text",
            showlegend=False,
            name="flips",
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
