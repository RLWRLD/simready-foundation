"""Run a deformable asset through its experiments in Newton 1.5 VBD, and make the videos.

    python asset_checks/native/run.py <asset.usda> --bench <simready-bench> --out <dir>
        [--experiments drop,press] [--setup <setup>] [--timeout 1200] [--no-render]

One asset, one experiment, one engine, one solver -- the four things the user names -- and this
turns them into a measured run and a video. It is the deformable counterpart of the Kit runner:
NVIDIA's three tests read rigid-body transforms and refuse an asset without RigidBodyAPI, and
Isaac's Fabric sync carries rigid-body transforms only, so a deformable simulated through that
path is both unmeasurable and invisible. Here Newton is driven directly, the solver's own state is
what gets measured, and the animated USD each run writes is what gets photographed.

Deformables are checked in one environment, `envs.DEFORMABLE` (Newton 1.5 VBD; the scope decision
of 2026-09-29). Each cell is a separate process in that environment's virtual environment.

Exit status: 0 every cell passed, 1 a cell ran and failed its experiment, 2 a cell has no verdict
(refused, errored, timed out).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from asset_checks import envs, experiments as rigid_experiments, video  # noqa: E402
import setups  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ENV = envs.ENVIRONMENTS[envs.DEFORMABLE]
# Which experiments a deformable asset has, and which script drives each, from the one place
# experiments are declared.
EXPERIMENTS = tuple(sorted(rigid_experiments.for_kind("deformable")))
SCRIPTS = {name: module.DEFORMABLE for name, module in rigid_experiments.for_kind("deformable").items()}


def cell_command(bench, experiment, asset, usd, seconds, setup):
    script = SCRIPTS.get(experiment)
    if script is None:
        return None, f"there is no deformable {experiment} experiment"
    return [str(bench / f".venv-{ENV.venv}" / "bin" / "python"), str(HERE / script), asset,
            "--seconds", str(seconds), "--usd", str(usd), "--setup", setup], None


def read_result(log):
    """Pull the numbers each experiment prints, without teaching this file what they mean.

    The experiment owns its verdict; the runner only collects it. A line that starts RESULT is
    a run of `name=value`, and everything else in the log is there for a person to read.
    """
    found = {}
    for line in log.splitlines():
        marker = line.partition("] ")[2]
        if marker.startswith("RESULT "):
            for token in marker[len("RESULT "):].split():
                name, _, value = token.partition("=")
                found[name] = value
        elif marker.startswith("t="):
            found["last_frame"] = marker
        elif "diverged" in marker:
            found["diverged"] = marker
    return found


def ending(log_path):
    """How a run ended, from its log's last lines: the last few non-empty ones, trimmed."""
    lines = [l.strip() for l in log_path.read_text(errors="replace").splitlines() if l.strip()]
    return "\n".join(l[:300] for l in lines[-4:])


def run(command, log_path, timeout):
    started = time.time()
    with open(log_path, "w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return code, round(time.time() - started, 1)


# What each experiment is worth reading, and in what order. The runner does not know what these
# mean -- the experiment prints them and this only lays them out -- but a table nobody can read
# is a table nobody checks.
COLUMNS = {
    "drop": [("verdict", "verdict"), ("fell_mm", "fell (mm)"), ("thickness_mm", "settled (mm)"),
             ("height_kept", "height kept"),
             ("below_floor_mm", "below floor (mm)"), ("p99_speed", "p99 speed"),
             ("realtime_x", "x real time")],
    "press": [("verdict", "verdict"), ("indented_mm", "plate went in (mm)"),
              ("compressed_mm", "asset gave (mm)"),
              ("compressed_frac", "of height"), ("recovered_frac", "recovered"),
              ("below_floor_mm", "below floor (mm)"), ("pressed_nodes", "nodes pressed"),
              ("realtime_x", "x real time")],
}


def summary(asset, results):
    """One table per experiment, plus where each video is."""
    lines = [f"# {pathlib.Path(asset).name}", "", video.describe(), ""]
    for experiment, columns in COLUMNS.items():
        rows = {c: r for c, r in results.items() if r.get("experiment") == experiment}
        if not rows:
            continue
        lines += [f"## {experiment}", "",
                  "| environment | " + " | ".join(label for _, label in columns) + " | videos |",
                  "|---" * (len(columns) + 2) + "|"]
        for cell, row in sorted(rows.items()):
            if row.get("skipped") or row.get("error"):
                lines.append(f"| {row['env']} | {row.get('skipped') or row['error']} |"
                             + " |" * len(columns))
                continue
            values = " | ".join(str(row.get(key, "-")) for key, _ in columns)
            videos = ", ".join(f"`{name}`" for name in (row.get("videos") or {}).values()) or "-"
            lines.append(f"| {row['env']} | {values} | {videos} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asset")
    ap.add_argument("--bench", default=None, help=f"simready-bench (default ${envs.BENCH_VARIABLE})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--setup", default=setups.CANON, choices=setups.NAMES,
                    help=f"where the {', '.join(setups.FACTORS)} come from (setups.py)")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    bench = envs.bench(args.bench)
    asset = str(pathlib.Path(args.asset).resolve())
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    code = envs.code_version()

    results = {}
    stem = pathlib.Path(asset).stem
    (out / stem).mkdir(parents=True, exist_ok=True)
    for env in (ENV.name,):
        for experiment in experiments:
            cell = f"{stem}__{env}__{experiment}"
            # One directory per cell -- <out>/<asset>/<env>/<experiment>/ -- the shape the rigid
            # runs write and `asset_checks.compare` reads. The recording, the logs, the two videos
            # and a result.json all live in it.
            cell_dir = out / stem / env / experiment
            cell_dir.mkdir(parents=True, exist_ok=True)
            usd = cell_dir / "recording.usda"
            seconds = rigid_experiments.get(experiment).SECONDS   # the experiment's, nobody else's
            row = {"env": env, "experiment": experiment, "setup": args.setup, "code": code}
            command, refusal = cell_command(bench, experiment, asset, usd, seconds, args.setup)
            if refusal:
                print(f"[run] {cell}: skipped -- {refusal}", flush=True)
                row["skipped"] = refusal
            else:
                print(f"[run] {cell}", flush=True)
                log_path = cell_dir / "run.log"
                try:
                    code, seconds_taken = run(command, log_path, args.timeout)
                except subprocess.TimeoutExpired:
                    row["error"] = "timed out"
                    print(f"[run] {cell}: TIMED OUT", flush=True)
                else:
                    found = read_result(log_path.read_text(errors="replace"))
                    row.update({"exit": code, "seconds": seconds_taken,
                                "usd": usd.name if usd.exists() else None, **found})
                    if "verdict" not in found:
                        # No RESULT line: the run ended some other way, and how is in its
                        # last lines -- a refusal's sentence, a traceback's last line, a
                        # kernel fault. Kept with the record so the report need not guess.
                        row["error"] = ending(log_path)
                    print(f"[run] {cell}: exit {code} in {seconds_taken}s -- "
                          f"{found or 'nothing reported'}", flush=True)
            results[cell] = row
            row["videos"], media = {}, []
            if not args.no_render and usd.exists():
                # One run, four videos: the geometry the solver moved and collided with, and the
                # asset's own textured mesh carried along by it, each at both speeds. Frame for
                # frame the same numbers. The role and the speed stay separate fields, because
                # that is what the compositor selects on.
                made, failed = video.draw(bench, cell_dir, usd, experiment, args.timeout, "")
                for view, speed, name, frames in made:
                    row["videos"][f"{view}__{speed}"] = name
                    media.append({"filename": name, "kind": "video", "role": view, "speed": speed})
                    print(f"[run] {cell}: {frames} {view} frames -> {name}", flush=True)
                if failed:
                    row["render_error"] = failed
            # What `asset_checks.compare` reads: a verdict it can colour, the experiment's own word
            # for what happened after FAIL, and the videos by role.
            said = row.get("verdict") or row.get("skipped") or row.get("error") or row.get("diverged")
            # `verdict` is pass/fail for the compositor's colour; `outcome` is the experiment's own
            # word (`through-the-floor`, `did-not-spring-back`, ...), which is what a table wants.
            (cell_dir / "result.json").write_text(json.dumps(
                {**row, "verdict": "pass" if row.get("verdict") == "pass" else "fail",
                 "outcome": row.get("verdict"), "message": said or "no result", "media": media},
                indent=1))

    if not args.no_render:
        for view in video.VIEWS:
            for speed in video.SPEEDS:
                subprocess.call([str(bench / f".venv-{ENV.venv}" / "bin" / "python"), "-m", "asset_checks.compare",
                                 str(out), "--role", view, "--speed", speed, "--envs", ENV.name],
                                env={**os.environ, "PYTHONPATH": str(HERE.parents[1])})
    # Per asset, under its own directory: several assets share one <out>, as they do in a rigid run.
    (out / stem / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (out / stem / "summary.md").write_text(summary(asset, results))
    print(f"\n[run] wrote {out / stem / 'results.json'}")
    for cell, row in sorted(results.items()):
        state = row.get("skipped") or row.get("error") or row.get("diverged") or "ok"
        made = ", ".join((row.get("videos") or {}).values()) or "-"
        print(f"[run] {cell:<46} {state:<24} {made}")
    if any(r.get("verdict") is None or r.get("error") or r.get("skipped") for r in results.values()):
        return 2
    return 0 if all(r.get("verdict") == "pass" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
