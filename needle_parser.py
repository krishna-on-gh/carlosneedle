"""
Paste parser for the election-night needle.

Accepts raw text pasted from NYT results pages (or similar formats) plus
the (dem_last_name, rep_last_name) for the race being pasted, and returns
a DataFrame of county rows the needle can consume.

Handles:
  - Single-line rows: "County   Trump +5   45,000   75%"
  - Multi-line rows: county+margin on one line, pct on next, votes on third
  - Tab-separated or 2+ space-separated
  - Multi-word counties: "St. Clair", "Grand Traverse", "White Pine"
  - Percent variations: "75%", ">95%", "<1%"
  - Blank/no-data rows: "—", "--", "0%" (skipped with note)
"""

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class ParseResult:
    rows: pd.DataFrame          # county, reported_margin_rd, reported_votes, pct_reporting
    skipped: list[tuple[str, str]]  # (raw line, reason)


PCT_RE = re.compile(r">?<?\s*(\d+(?:\.\d+)?)\s*%")
VOTES_RE = re.compile(r"([\d,]{3,})")


def _clean(s: str) -> str:
    return s.replace("—", "-").replace("–", "-").replace("\xa0", " ").strip()


def parse_paste(raw_text: str, dem_last: str, rep_last: str) -> ParseResult:
    """Parse pasted election-night results text.

    Sign convention on output margin: positive = R won by that many pts.
    """
    if not dem_last or not rep_last:
        raise ValueError("dem_last_name and rep_last_name must be set on the race "
                         "before parsing. Fill them in needle_races_2026.csv.")

    dem_last = dem_last.strip()
    rep_last = rep_last.strip()
    name_re = re.compile(rf"\b({re.escape(dem_last)}|{re.escape(rep_last)})\s*\+\s*(-?\d+(?:\.\d+)?)",
                         re.IGNORECASE)

    lines = [_clean(ln) for ln in raw_text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty lines

    # Try single-line format first (most common).
    rows_single = _try_single_line(lines, name_re, dem_last, rep_last)
    if rows_single is not None and len(rows_single.rows) >= 1:
        return rows_single

    # Fall back to multi-line format (NYT sometimes wraps).
    return _try_multi_line(lines, name_re, dem_last, rep_last)


def _extract_pct(s: str):
    m = PCT_RE.search(s)
    return float(m.group(1)) if m else None


def _extract_votes(s: str):
    """Grab the largest comma-formatted number as vote count."""
    matches = VOTES_RE.findall(s)
    if not matches:
        return 0
    nums = [int(m.replace(",", "")) for m in matches]
    return max(nums)


def _try_single_line(lines, name_re, dem_last, rep_last):
    good_rows = []
    skipped = []
    header_words = {"county", "margin", "votes", "percent", "reporting", "results"}
    for raw in lines:
        # Skip header/comment/separator lines.
        lower = raw.lower()
        if any(raw.startswith(prefix) for prefix in ("#", "//", "---")):
            continue
        if lower.startswith("county") or all(w in lower for w in ("margin", "votes")):
            continue

        # Detect blank/no-data rows.
        m = name_re.search(raw)
        if not m:
            # Rows like "Benzie   -   0   0%" or "Fulton   —   —" get skipped silently
            if re.search(r"[-—–]\s*(0|-|—)?\s*\d*\s*(0%)?$", raw):
                continue
            skipped.append((raw, "no candidate name found"))
            continue

        who = m.group(1)
        val = float(m.group(2))

        # County name = everything before the candidate name, cleaned.
        pre = raw[:m.start()].rstrip().rstrip("\t").rstrip()
        # Strip trailing tab-delimited empties and normalize inner whitespace.
        county = re.sub(r"\s+", " ", pre).strip().upper()
        if not county:
            skipped.append((raw, "no county name before candidate")); continue

        pct = _extract_pct(raw[m.end():])
        votes = _extract_votes(raw[m.end():])

        margin = val if who.lower() == rep_last.lower() else -val
        good_rows.append({
            "county": county,
            "reported_margin_rd": margin,
            "reported_votes": votes,
            "pct_reporting": pct if pct is not None else 0.0,
        })

    return ParseResult(pd.DataFrame(good_rows), skipped) if good_rows else None


def _try_multi_line(lines, name_re, dem_last, rep_last):
    """NYT sometimes emits 3-line rows: [name+margin], [pct%], [votes...].

    Consume greedily: at each line that matches the name regex, look at the next
    1-2 lines for pct/votes.
    """
    good_rows = []
    skipped = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        m = name_re.search(raw)
        if not m:
            i += 1
            continue
        who = m.group(1)
        val = float(m.group(2))
        pre = raw[:m.start()].rstrip().rstrip("\t").rstrip()
        county = re.sub(r"\s+", " ", pre).strip().upper()

        # Look at next 1-2 lines for pct and votes.
        pct = None
        votes = 0
        j = i + 1
        while j < min(i + 3, len(lines)):
            nxt = lines[j]
            if pct is None:
                p = _extract_pct(nxt)
                if p is not None:
                    pct = p
            if votes == 0:
                v = _extract_votes(nxt)
                if v > 0:
                    votes = v
            # Stop if we hit the next candidate row.
            if name_re.search(nxt):
                break
            j += 1

        margin = val if who.lower() == rep_last.lower() else -val
        good_rows.append({
            "county": county,
            "reported_margin_rd": margin,
            "reported_votes": votes,
            "pct_reporting": pct if pct is not None else 0.0,
        })
        i = j if j > i + 1 else i + 1

    return ParseResult(pd.DataFrame(good_rows), skipped)


if __name__ == "__main__":
    # Quick smoke test against the existing snapshot files.
    from pathlib import Path

    root = Path(__file__).parent / "data" / "backtest"
    cases = [
        (root / "mi_gov_2022_snapshot_73pct.txt", "Whitmer", "Dixon"),
        (root / "nv_gov_2022_snapshot_50pct.txt", "Sisolak", "Lombardo"),
        (root / "pa_gov_2022_snapshot_67pct.txt", "Shapiro", "Mastriano"),
        (root / "ga_pres_2024_snapshot_raw.txt", "Harris", "Trump"),
    ]
    for path, dem, rep in cases:
        if not path.exists():
            print(f"skip {path.name} (missing)"); continue
        raw = path.read_text()
        result = parse_paste(raw, dem, rep)
        print(f"\n== {path.name} ==")
        print(f"  parsed {len(result.rows)} rows, {len(result.skipped)} skipped")
        if len(result.rows):
            print(f"  first 3 rows:")
            print(result.rows.head(3).to_string(index=False))
        if result.skipped:
            print(f"  first 3 skipped:")
            for line, reason in result.skipped[:3]:
                print(f"    [{reason}]  {line[:80]}")
