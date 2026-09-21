"""Run a deformable asset through the named environments and experiments, and make the videos.

    python asset_checks/native/run.py <asset.usda> --out <dir>
        [--envs newton12_vbd,newton15_vbd,newton12_xpbd,newton15_xpbd,physx]
        [--experiments drop,press] [--seconds 2] [--size 768] [--no-render]

One asset, one experiment, one engine, one solver -- the four things the user names -- and this
turns them into a measured run and a video. It is the deformable counterpart of the Kit runner:
NVIDIA's three tests read rigid-body transforms and refuse an asset without RigidBodyAPI, and
Isaac's Fabric sync carries rigid-body transforms only, so a deformable simulated through that
path is both unmeasurable and invisible. Here each engine is driven directly, the solver's own
state is what gets measured, and the animated USD each run writes is what gets photographed.

Every cell is a separate process in its own virtual environment, because the two Newton versions
cannot share one: isaacsim-core pins `newton[sim]==1.2.1` against Isaac 6.0.1 and `==1.5.0`
against 6.1.0. That is also why a result is only meaningful with all four names attached.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import agreement

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from asset_checks import envs as rigid_envs, video  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BENCH = pathlib.Path("/home/wongyun/Workspace/Research/Robotics/simready-bench")

# The environments are `asset_checks.envs`' -- the same table the rigid runs use, down to the
# header text a comparison video prints. Only the solvers that can move particles are here:
# `kit/scene.py::SOLVER_SIMULATES` records that MuJoCo refuses a stage whose bodies are particles,
# measured rather than assumed, so the two `mujoco` environments are not offered for a deformable.
PARTICLE_SOLVERS = ("physx", "vbd", "xpbd")
ENVIRONMENTS = {name: (e.venv, e.engine, e.solver) for name, e in rigid_envs.ENVIRONMENTS.items()
                if e.solver in PARTICLE_SOLVERS}
EXPERIMENTS = ("drop", "press")
# PhysX is driven from inside Kit, so it needs the isaac-run launcher; Newton is a plain import.
SCRIPTS = {("newton", "drop"): "newton_drop.py", ("newton", "press"): "newton_press.py",
           ("physx", "drop"): "physx_drop.py", ("physx", "press"): "physx_press.py"}


# PhysX and Newton do not read the same schemas -- Newton reads the AOUSD public `Physics*`
# names, PhysX reads the same physics under an `OmniPhysics` prefix and translates neither -- so
# a PhysX cell runs a converted copy of the asset. `to_physx.py` makes it from the same
# tetrahedra and the same declared material; anything else would compare the conversion.
def physx_asset(asset):
    """The PhysX copy of the asset, but only once it has been shown to still be the asset.

    Every error in the conversion arrives disguised as an engine difference -- a wrong modulus or
    a dropped tetrahedron would read as "PhysX behaves differently" and nothing downstream could
    tell. `physx_parity` re-derives from both USDs what the conversion claims to have carried, and
    it shares no code with the conversion, so it is a measurement against a statement rather than
    a second copy of the same statement. A cell whose asset failed it must not run at all: a
    number from it would look exactly like a number from a good one.
    """
    converted = pathlib.Path(asset).with_name(pathlib.Path(asset).stem + "_physx.usda")
    if not converted.exists():
        raise SystemExit(f"{asset} has no PhysX counterpart at {converted}. Make one with:\n"
                         f"  {BENCH}/isaac-run isaac610 {HERE / 'to_physx.py'} {asset} {converted}")
    # Run in a venv rather than imported: this runner is plain Python and `pxr` lives in the
    # Isaac environments, the same reason every cell is a subprocess.
    check = subprocess.run([str(BENCH / ".venv-isaac610" / "bin" / "python"),
                            str(HERE / "physx_parity.py"), str(asset), str(converted)],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONNOUSERSITE="1"))
    for line in check.stdout.splitlines():
        if line.startswith("[parity]"):
            print(f"[run] {line}", flush=True)
    if check.returncode != 0:
        said = [l for l in check.stdout.splitlines() if "MISMATCH" in l] or [check.stderr.strip()[-400:]]
        raise ParityFailure("the PhysX copy is not the same asset as the one being evaluated: "
                            + "; ".join(s.partition("MISMATCH: ")[2] or s for s in said))
    return str(converted)


class ParityFailure(RuntimeError):
    """The PhysX copy differs from the asset, so its results would not be about the asset."""


def cell_command(env, experiment, asset, usd, seconds):
    venv, engine, solver = ENVIRONMENTS[env]
    script = SCRIPTS.get((engine, experiment))
    if script is None:
        return None, f"{engine} has no {experiment} experiment yet"
    if engine == "physx":
        # The PhysX copy is what is simulated; the original is where the textures live, and the
        # conversion deactivates the subtree they are on.
        return [str(BENCH / "isaac-run"), venv, str(HERE / script), physx_asset(asset),
                "--seconds", str(seconds), "--usd", str(usd), "--visual-asset", asset], None
    return [str(BENCH / f".venv-{venv}" / "bin" / "python"), str(HERE / script), asset,
            "--solver", solver, "--seconds", str(seconds), "--usd", str(usd)], None


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
        elif marker.startswith("most soft contacts"):
            found["soft_contacts"] = marker.rsplit(": ", 1)[-1]
    return found


def run(command, log_path, timeout):
    started = time.time()
    with open(log_path, "w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return code, round(time.time() - started, 1)


# What each experiment is worth reading, and in what order. The runner does not know what these
# mean -- the experiment prints them and this only lays them out -- but a table nobody can read
# is a table nobody checks.
# A 5 cm drop takes 0.101 s. At real speed that is three frames and nobody sees it happen, which
# is why the videos looked fast-forwarded: they were not, the event is simply that short. Every
# simulated frame is captured and played back this many times slower, and the factor is burned
# into the file name so no one has to remember it.
COLUMNS = {
    "drop": [("verdict", "verdict"), ("fell_mm", "fell (mm)"), ("thickness_mm", "settled (mm)"),
             ("height_kept", "height kept"),
             ("below_floor_mm", "below floor (mm)"), ("p99_speed", "p99 speed")],
    "press": [("verdict", "verdict"), ("indented_mm", "plate went in (mm)"),
              ("compressed_mm", "asset gave (mm)"),
              ("compressed_frac", "of height"), ("recovered_frac", "recovered"),
              ("below_floor_mm", "below floor (mm)"), ("soft_contacts", "plate contacts")],
}


def summary(asset, results):
    """One table per experiment, plus where each video is."""
    lines = [f"# {pathlib.Path(asset).name}", "",
             "Two videos per cell, from one run: `__collision` is the geometry the solver moved and",
             "collided with, `__visual` is the asset's own textured mesh carried along by it. The",
             "side-by-side strips are in `compare/`.", ""]
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
    table = agreement.section(results)
    if table:
        lines.append(table)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", default="newton12_vbd,newton12_xpbd,newton15_vbd,newton15_xpbd,physx")
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--press-seconds", type=float, default=4.0)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    asset = str(pathlib.Path(args.asset).resolve())
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    unknown = [e for e in envs if e not in ENVIRONMENTS]
    if unknown:
        raise SystemExit(f"unknown environment(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(ENVIRONMENTS)}")
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    results = {}
    stem = pathlib.Path(asset).stem
    for env in envs:
        for experiment in experiments:
            cell = f"{stem}__{env}__{experiment}"
            # One directory per cell -- <out>/<asset>/<env>/<experiment>/ -- the shape the rigid
            # runs write and `asset_checks.compare` reads. The recording, the logs, the two videos
            # and a result.json all live in it.
            cell_dir = out / stem / env / experiment
            cell_dir.mkdir(parents=True, exist_ok=True)
            usd = cell_dir / "recording.usda"
            seconds = args.press_seconds if experiment == "press" else args.seconds
            row = {"env": env, "experiment": experiment}
            try:
                command, refusal = cell_command(env, experiment, asset, usd, seconds)
            except ParityFailure as failure:
                print(f"[run] {cell}: REFUSED -- {failure}", flush=True)
                row["error"] = str(failure)
            else:
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
                        print(f"[run] {cell}: exit {code} in {seconds_taken}s -- "
                              f"{found or 'nothing reported'}", flush=True)
            results[cell] = row
            row["videos"] = {}
            if not args.no_render and usd.exists():
                # One run, two videos: the geometry the solver moved and collided with, and the
                # asset's own textured mesh carried along by it. Frame for frame the same numbers.
                for view, name, frames in video.draw(BENCH, cell_dir, usd, experiment, args.timeout, ""):
                    row["videos"][view] = name
                    print(f"[run] {cell}: {frames} {view} frames -> {name}", flush=True)
            # What `asset_checks.compare` reads: a verdict it can colour, the experiment's own word
            # for what happened after FAIL, and the videos by role.
            said = row.get("verdict") or row.get("skipped") or row.get("error") or row.get("diverged")
            (cell_dir / "result.json").write_text(json.dumps(
                {**row, "verdict": "pass" if row.get("verdict") == "pass" else "fail",
                 "message": said or "no result",
                 "media": [{"filename": name, "kind": "video", "role": view}
                           for view, name in row["videos"].items()]}, indent=1))

    if not args.no_render:
        for view in video.VIEWS:
            subprocess.call([str(BENCH / ".venv-isaac610" / "bin" / "python"), "-m", "asset_checks.compare",
                             str(out), "--role", view, "--envs", ",".join(envs)],
                            env={**os.environ, "PYTHONPATH": str(HERE.parents[1])})
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (out / "summary.md").write_text(summary(asset, results))
    print(f"\n[run] wrote {out / 'results.json'}")
    for cell, row in sorted(results.items()):
        state = row.get("skipped") or row.get("error") or row.get("diverged") or "ok"
        made = ", ".join((row.get("videos") or {}).values()) or "-"
        print(f"[run] {cell:<46} {state:<24} {made}")


if __name__ == "__main__":
    sys.exit(main())
