# SPDX-License-Identifier: Apache-2.0
"""What a run directory says, as one page.

    python -m asset_checks.report <run dir> [-o report.md]

A cell's own `result.json` is the record; this reads every one of them and writes the table a
person actually looks at -- asset down the side, environment across, the experiment's own word for
what happened in each box -- plus the measurements behind the boxes and the cells that produced no
verdict at all, with the reason from their log.

It is a *reader*. It holds no list of assets, no list of environments and no list of experiments:
it finds what is there. A run with a new asset in it grows a row, a run on one engine has one
column, and neither costs an edit here. Nothing in it decides anything either -- a verdict is the
runner's word, printed as the runner wrote it -- because a second opinion about a cell is a second
place for the truth to live.
"""
import argparse
import collections
import json
import pathlib

# What a cell that never reported looks like from its log, most specific first. These are the ends
# a run can come to outside its own verdict, and each is a different thing to do about it, so they
# are told apart rather than all called "failed".
WITHOUT_A_VERDICT = (
    ("tensor entity is not valid", "no body",
     "the engine built no simulated body from this asset"),
    ("Warp copy error", "solver crash",
     "the solver faulted and took the process with it"),
    ("illegal memory access", "solver crash",
     "the solver faulted and took the process with it"),
    # Last, because an allocation failing right after a kernel fault is the fault, not the
    # machine: the patterns above claim it first when both are in the log.
    ("Failed to allocate", "out of memory",
     "the device could not allocate, with no fault before it: the machine, not the asset"),
    ("the PhysX copy", "copy refused",
     "the PhysX copy of the asset failed parity or could not be written; PhysX never ran"),
    ("timed out", "timed out", "the cell exceeded its time limit"),
)


def cells(run):
    """(asset, environment, experiment) -> (what happened, the numbers) for every cell present."""
    out = {}
    for path in sorted(run.glob("*/*/*/result.json")):
        asset, env, experiment = path.parts[-4], path.parts[-3], path.parts[-2]
        record = json.loads(path.read_text())
        out[(asset, env, experiment)] = (verdict_of(record, path.parent), record)
    # A cell refused before launch is a cell too: `check` wrote the reason where the result would
    # have gone. It is one word in the box and one line under the table, never a verdict.
    for path in sorted(run.glob("*/*/*/refused.json")):
        key = tuple(path.parts[-4:-1])
        if key not in out:
            out[key] = ("refused", json.loads(path.read_text()))
    return out


def verdict_of(record, cell):
    """The one word for this cell.

    The runner's own verdict when it reached one. When it did not, the log says why, and which
    'why' it is matters: an engine that would not build a body is a finding about the asset and the
    engine, a solver that faulted is a finding about the solver, and a failed allocation is neither
    -- it is the machine, and the cell should simply be run again.
    """
    if record.get("diverged"):
        return "diverged"
    if record.get("outcome"):
        return record["outcome"]
    if record.get("verdict") not in (None, "fail", "pass"):
        return record["verdict"]
    # A run that ended without a RESULT line says how in `error` (its log's last lines, or the
    # harness's own reason); it is classified on that and on nothing else in the log.
    log = cell / "run.log"
    ending = record.get("error")
    if ending is None and record.get("message") == "no result" and log.is_file():
        ending = "\n".join(l for l in log.read_text(errors="ignore").splitlines() if l.strip())[-1200:]
    if ending is not None:
        for needle, word, _ in WITHOUT_A_VERDICT:
            if needle in ending:
                return word
        return "no result"
    # `pass`/`fail` alone means the runner printed a RESULT line and the harness graded it; the
    # experiment's own word for a failure is in that line and is the more useful one.
    if log.is_file():
        for line in log.read_text(errors="ignore").splitlines():
            if " RESULT " in line and "verdict=" in line:
                return line.rsplit("verdict=", 1)[1].strip()
    return record.get("verdict", "?")


def table(found, experiment, assets, envs):
    rows = [f"| asset | {' | '.join(envs)} |", f"|---|{'---|' * len(envs)}"]
    for asset in assets:
        boxes = [found.get((asset, env, experiment), ("--", {}))[0] for env in envs]
        rows.append(f"| {asset} | {' | '.join(boxes)} |")
    return "\n".join(rows)


def numbers(found, experiment, assets, envs, keys):
    """The measurements behind the boxes, for the keys this experiment actually recorded."""
    present = [k for k in keys if any(
        k in found.get((a, e, experiment), ("", {}))[1] for a in assets for e in envs)]
    if not present:
        return ""
    rows = [f"| asset | environment | {' | '.join(present)} |", f"|---|---|{'---|' * len(present)}"]
    for asset in assets:
        for env in envs:
            record = found.get((asset, env, experiment), (None, None))[1]
            if not record or not any(k in record for k in present):
                continue
            rows.append(f"| {asset} | {env} | "
                        + " | ".join(str(record.get(k, "--")) for k in present) + " |")
    return "\n".join(rows)


# The measurements worth putting under a table, in the order a reader wants them. A key that no
# cell recorded is left out, so one list serves every experiment.
KEYS = ("fell_mm", "thickness_mm", "height_kept", "below_floor_mm", "first_frame_x",
        "p99_speed", "max_speed", "indented_mm", "compressed_mm", "compressed_frac",
        "recovered_frac", "pressed_nodes", "seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None, help="default: <run dir>/report.md")
    args = ap.parse_args()
    run = pathlib.Path(args.run_dir).resolve()
    found = cells(run)
    if not found:
        raise SystemExit(f"{run} holds no cell with a result.json")

    assets = sorted({a for a, _, _ in found})
    envs = sorted({e for _, e, _ in found})
    experiments = sorted({x for _, _, x in found})
    tally = collections.Counter(v for v, _ in found.values())

    out = [f"# {run.name}", "",
           f"{len(found)} cells: {len(assets)} asset(s) x {len(envs)} environment(s) "
           f"x {len(experiments)} experiment(s).", "",
           "  ".join(f"`{word}` {n}" for word, n in tally.most_common()), ""]
    for experiment in experiments:
        out += [f"## {experiment}", "", table(found, experiment, assets, envs), ""]
        rows = numbers(found, experiment, assets, envs, KEYS)
        if rows:
            out += ["<details><summary>measurements</summary>", "", rows, "", "</details>", ""]

    without = {(a, e, x): (v, rec) for (a, e, x), (v, rec) in found.items()
               if v in {w for _, w, _ in WITHOUT_A_VERDICT} | {"no result"}}
    if without:
        out += ["## cells that reached no verdict", ""]
        why = dict((w, s) for _, w, s in WITHOUT_A_VERDICT)
        for (a, e, x), (v, rec) in sorted(without.items()):
            last = (rec.get("error") or "").strip().splitlines()
            said = f" -- `{last[-1][:160]}`" if last else ""
            out.append(f"- `{a}` / `{e}` / `{x}`: **{v}** -- {why.get(v, 'see its run.log')}{said}")
        out.append("")

    refused = {k: rec for k, (v, rec) in found.items() if v == "refused"}
    if refused:
        out += ["## cells refused before launch", "",
                "Nothing was simulated: the asset, the engine and the experiment cannot mean "
                "anything together, and the reason is the one the runner would have given.", ""]
        for (a, e, x), rec in sorted(refused.items()):
            out += [f"- `{a}` / `{e}` / `{x}`: {rec['reason']}"]
        out.append("")

    # Anything missing is a cell that was meant to run and did not, which is worth seeing.
    missing = sorted({(a, e, x) for a in assets for e in envs for x in experiments} - set(found))
    if missing:
        out += ["## cells that were meant to run and did not", "",
                "Not refused and not recorded: a launch that never finished. Its run.log, if there "
                "is one, says how far it got.", ""]
        out += [f"- `{a}` / `{e}` / `{x}`" for a, e, x in missing] + [""]

    strips = sorted((run / "compare").glob("*.mp4"))
    videos = [p for p in run.glob("*/*/*/*.mp4")]
    out += ["## videos", "",
            f"{len(videos)} cell video(s) and {len(strips)} side-by-side strip(s) in `compare/`.", ""]
    out += [f"- `compare/{p.name}`" for p in strips] + [""]

    target = pathlib.Path(args.out) if args.out else run / "report.md"
    target.write_text("\n".join(out))
    print(f"[report] {target}: {len(found)} cells, {len(assets)} assets, {len(envs)} environments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
