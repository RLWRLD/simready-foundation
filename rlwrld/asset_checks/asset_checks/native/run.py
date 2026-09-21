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
import subprocess
import sys
import time

import agreement

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BENCH = pathlib.Path("/home/wongyun/Workspace/Research/Robotics/simready-bench")

# Environment name -> (venv tag, engine, solver). The names are the ones in asset_checks.envs;
# what changes here is only how a deformable is driven in each.
ENVIRONMENTS = {
    "physx": ("isaac610", "physx", "physx"),
    "physx601": ("isaac601", "physx", "physx"),
    "newton12_vbd": ("isaac601", "newton", "vbd"),
    "newton12_xpbd": ("isaac601", "newton", "xpbd"),
    "newton15_vbd": ("isaac610", "newton", "vbd"),
    "newton15_xpbd": ("isaac610", "newton", "xpbd"),
}
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
PLAYBACK_SLOWDOWN = 4.0

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
             "Two videos per cell, from one run: `__sim` is the tetrahedral surface the solver",
             "moved, `__visual` is the asset's own textured mesh carried along by it.", ""]
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
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--fps", type=int, default=60, help="frames captured per simulated second")
    ap.add_argument("--slowdown", type=float, default=PLAYBACK_SLOWDOWN,
                    help="how many times slower than real time the videos play")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    asset = str(pathlib.Path(args.asset).resolve())
    out = pathlib.Path(args.out).resolve()
    for folder in ("logs", "usd", "frames", "videos"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    unknown = [e for e in envs if e not in ENVIRONMENTS]
    if unknown:
        raise SystemExit(f"unknown environment(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(ENVIRONMENTS)}")
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    results = {}
    for env in envs:
        for experiment in experiments:
            cell = f"{pathlib.Path(asset).stem}__{env}__{experiment}"
            usd = out / "usd" / f"{cell}.usda"
            seconds = args.press_seconds if experiment == "press" else args.seconds
            try:
                command, refusal = cell_command(env, experiment, asset, usd, seconds)
            except ParityFailure as failure:
                print(f"[run] {cell}: REFUSED -- {failure}", flush=True)
                results[cell] = {"env": env, "experiment": experiment, "error": str(failure)}
                continue
            if refusal:
                print(f"[run] {cell}: skipped -- {refusal}", flush=True)
                results[cell] = {"env": env, "experiment": experiment, "skipped": refusal}
                continue
            print(f"[run] {cell}", flush=True)
            log_path = out / "logs" / f"{cell}.log"
            try:
                code, seconds_taken = run(command, log_path, args.timeout)
            except subprocess.TimeoutExpired:
                results[cell] = {"env": env, "experiment": experiment, "error": "timed out"}
                print(f"[run] {cell}: TIMED OUT", flush=True)
                continue
            found = read_result(log_path.read_text(errors="replace"))
            results[cell] = {"env": env, "experiment": experiment, "exit": code,
                             "seconds": seconds_taken, "usd": usd.name if usd.exists() else None,
                             **found}
            print(f"[run] {cell}: exit {code} in {seconds_taken}s -- "
                  f"{found or 'nothing reported'}", flush=True)

            if args.no_render or not usd.exists():
                continue
            # One run, two videos. The recording holds both meshes, so these are the same physics
            # photographed twice: `sim` is the tetrahedral surface the solver actually moved, and
            # `visual` is the asset's own textured mesh carried along by it. They line up frame
            # for frame, because they came out of the same numbers.
            results[cell]["videos"] = {}
            for mesh in ("sim", "visual"):
                frames = out / "frames" / f"{cell}__{mesh}"
                render = [str(BENCH / "isaac-run"), "isaac610", str(HERE / "render_usd.py"), str(usd),
                          str(frames), "--fps", str(args.fps), "--size", str(args.size),
                          "--show", mesh]
                try:
                    code, _ = run(render, out / "logs" / f"{cell}.{mesh}.render.log", args.timeout)
                except subprocess.TimeoutExpired:
                    code = -1
                written = sorted(frames.glob("frame_*.png"))
                if not written:
                    print(f"[run] {cell}: no {mesh} frames (exit {code})", flush=True)
                    continue
                slow = f"{args.slowdown:g}x" if args.slowdown != 1.0 else "realtime"
                video = out / "videos" / f"{cell}__{mesh}__{slow}slower.mp4"
                subprocess.call(["ffmpeg", "-y", "-loglevel", "error",
                                 "-framerate", str(args.fps / args.slowdown),
                                 "-i", str(frames / "frame_%05d.png"), "-c:v", "libx264",
                                 "-pix_fmt", "yuv420p", "-crf", "20", str(video)])
                if video.exists():
                    results[cell]["videos"][mesh] = video.name
                print(f"[run] {cell}: {len(written)} {mesh} frames -> {video.name}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (out / "summary.md").write_text(summary(asset, results))
    print(f"\n[run] wrote {out / 'results.json'}")
    for cell, row in sorted(results.items()):
        state = row.get("skipped") or row.get("error") or row.get("diverged") or "ok"
        made = ", ".join((row.get("videos") or {}).values()) or "-"
        print(f"[run] {cell:<46} {state:<24} {made}")


if __name__ == "__main__":
    sys.exit(main())
