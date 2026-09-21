"""Put the engines' answers to the same question side by side.

Each cell measures one physical question -- how thick does this asset settle, how
far does it give under the plate -- so the five answers belong in one table where
a reader can see the spread at a glance.  That spread is the product: Newton
1.2.1's XPBD settles this banana at 3.7 mm where the others say 25-31, because
its kernel ignores the asset's material, and reporting that is the point.

**There is deliberately no threshold here.** A ratio-to-median rule was written
and thrown away: the legitimate spread between engines (8.4x on the banana) is
wider than the defects worth catching (2.0x for a VBD cloth that was settling at
twice everyone else's thickness), so no threshold separates the two, and a mark
that cannot tell them apart is worse than no mark.  The one case a rule would
have caught -- 63.3 mm against a 0.5 mm median -- a reader catches instantly from
the table.

This is a legitimate duplicate under the one-owner rule because it restates
nothing: it reads the numbers the runners produced and lays them out.
"""

# The keys the runners actually write, which arrive as strings off a RESULT line.
COMPARED = {"drop": ("thickness_mm", "settled thickness"),
            "press": ("compressed_mm", "how much the asset gave")}


def _number(value):
    """The value as a float, or None when the cell did not report one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def compare(results):
    """-> {experiment: (label, median, [(env, value, spread_from_median)])}."""
    out = {}
    for experiment, (key, label) in COMPARED.items():
        measured = [(r["env"], _number(r.get(key))) for r in results.values()
                    if r.get("experiment") == experiment]
        measured = [(env, value) for env, value in measured if value is not None]
        if len(measured) < 2:
            continue
        middle = _median([v for _, v in measured])
        out[experiment] = (label, middle,
                           [(env, value, value - middle) for env, value in sorted(measured)])
    return out


def section(results):
    """The markdown for `summary.md`, or nothing when there is nothing to compare."""
    tables = compare(results)
    if not tables:
        return ""
    lines = ["## the engines side by side", "",
             "The same asset, the same experiment, five engines. The spread is the finding,",
             "not an error: where an engine sits far from the rest, either it is doing",
             "something the others are not, or our setup for it is wrong -- and the run log",
             "for that cell says which.", ""]
    for experiment, (label, middle, rows) in sorted(tables.items()):
        lines += [f"**{experiment}** -- {label}, median {middle:.2f} mm", "",
                  "| environment | value (mm) | from median (mm) |", "|---|---|---|"]
        for env, value, delta in rows:
            lines.append(f"| {env} | {value:.2f} | {delta:+.2f} |")
        lines.append("")
    return "\n".join(lines)
