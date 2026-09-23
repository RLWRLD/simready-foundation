# SPDX-License-Identifier: Apache-2.0
"""One page over a sweep of setups: the same cells, one run directory per setup.

    python -m asset_checks.setups_report <sweep dir> [-o report.md]

<sweep dir>/<setup>/ is a run directory as `asset_checks.report` reads it, and each carries one
setup (structure-contact-stepping; native/setups.py). This lays them side by side: per asset and
experiment, setups down the side and environments across, the cell's own word in the box, and
under it the measurements that say whether a multi-body asset stayed one object (`integrity.py`).
It reads `report.cells` and decides nothing.
"""
import argparse
import collections
import pathlib
import sys

from asset_checks import report

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "native"))
import setups  # noqa: E402

MEASURES = ("seal_median_mm", "seal_max_mm", "escaped_nodes", "below_floor_mm", "p99_speed",
            "height_kept", "compressed_mm", "recovered_frac", "seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_dir")
    ap.add_argument("-o", "--out", default=None, help="default: <sweep dir>/report.md")
    args = ap.parse_args()
    sweep = pathlib.Path(args.sweep_dir).resolve()
    runs = {}
    for sub in sorted(sweep.iterdir()):
        if sub.is_dir() and sub.name in setups.NAMES:
            found = report.cells(sub)
            if found:
                runs[sub.name] = found
    if not runs:
        raise SystemExit(f"{sweep} holds no <setup>/ run directory with cells")
    order = [name for name in setups.NAMES if name in runs]
    keys = sorted({k for found in runs.values() for k in found})
    assets = sorted({a for a, _, _ in keys})
    envs = sorted({e for _, e, _ in keys})
    experiments = sorted({x for _, _, x in keys})
    total = sum(len(f) for f in runs.values())
    tally = collections.Counter(v for f in runs.values() for v, _ in f.values())

    out = [f"# {sweep.name}", "",
           f"{total} cells over {len(order)} setup(s): {len(assets)} asset(s) x {len(envs)} "
           f"environment(s) x {len(experiments)} experiment(s). A setup is "
           f"structure-contact-stepping (native/setups.py): " + "; ".join(
               f"{f} in {{{', '.join(l)}}}" for f, l in setups.FACTORS.items()) + ".", "",
           "  ".join(f"`{word}` {n}" for word, n in tally.most_common()), ""]
    for asset in assets:
        for experiment in experiments:
            rows = [(name, [runs[name].get((asset, env, experiment)) for env in envs]) for name in order]
            if not any(cell for _, cells in rows for cell in cells):
                continue
            out += [f"## {asset} / {experiment}", "",
                    f"| setup | {' | '.join(envs)} |", f"|---|{'---|' * len(envs)}"]
            for name, cells in rows:
                out.append(f"| {name} | " + " | ".join(c[0] if c else "--" for c in cells) + " |")
            out.append("")
            present = [k for k in MEASURES if any(c and k in c[1] for _, cells in rows for c in cells)]
            if present:
                out += ["<details><summary>measurements</summary>", "",
                        f"| setup | environment | {' | '.join(present)} |", f"|---|---|{'---|' * len(present)}"]
                for name, cells in rows:
                    for env, c in zip(envs, cells):
                        if c and any(k in c[1] for k in present):
                            out.append(f"| {name} | {env} | " + " | ".join(str(c[1].get(k, '--')) for k in present) + " |")
                out += ["", "</details>", ""]
    target = pathlib.Path(args.out) if args.out else sweep / "report.md"
    target.write_text("\n".join(out))
    print(f"[setups_report] {target}: {total} cells over {len(order)} setups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
