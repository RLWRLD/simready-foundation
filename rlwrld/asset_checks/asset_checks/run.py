# SPDX-License-Identifier: Apache-2.0
"""Run physics checks on assets in the three environments; one fresh Kit process per run.

    PYTHONPATH=rlwrld/asset_checks python3 -m asset_checks.run --bench <simready-bench> --out <new dir> \
        [--envs physx,newton12,newton15] [--experiments drop,slope,grasp] <asset.usd> [...]

Each asset is first validated with NVIDIA's simready-validate; its passed features gate the tests as
NVIDIA's runner would. A run counts only if its result.json says "done" and Kit demonstrably had
what was asked: the engine that actually simulated, the pose source for that engine, the GPU
settings, the Newton version, and no [BENCHMARK_COMPAT] line (that bridge must not be installed).
Anything else is reported INVALID.
"""
import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys

from asset_checks import envs, video

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]  # rlwrld/asset_checks: goes on Kit's PYTHONPATH
ENTRY = pathlib.Path(__file__).resolve().parent / "kit" / "entry.py"
TORCH_CHECK = (
    "import sys, torch\n"
    "try:\n    x = torch.randn(256, 256, device='cuda'); (x @ x).sum().item()\n"
    "except Exception as e:\n    sys.exit(f'torch {torch.__version__} cannot run CUDA work: {str(e)[:160]}')\n"
    "print(f'torch {torch.__version__} (CUDA {torch.version.cuda}) runs on {torch.cuda.get_device_name(0)}')\n"
)


def preflight(bench, venv):
    out = subprocess.run([str(bench / "isaac-run"), venv, "-c", TORCH_CHECK], capture_output=True, text=True, timeout=120)
    line = (out.stdout.strip().splitlines() or [""])[-1] if out.returncode == 0 else out.stderr.strip().splitlines()[-1]
    if out.returncode != 0:
        sys.exit(f"[asset_checks] {venv}: {line}")
    print(f"[asset_checks] {venv}: {line}", flush=True)


def validated_features(bench, asset, out_dir):
    """Features NVIDIA's static validator passes for the asset (its project config: the fork's samples')."""
    project = PACKAGE_ROOT.parents[1] / "sample_content" / "project_config.toml"
    report = out_dir / "validation.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(bench / "isaac-run"), "isaac610", str(bench / ".venv-isaac610" / "bin" / "simready-validate"),
           "--project-config", str(project), "--output", str(report), str(asset)]
    with open(out_dir / "validation.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300, cwd=str(project.parent))
    entry = json.loads(report.read_text()).get(str(asset), {}) if report.exists() else {}
    summary = entry.get("features_summary")
    if not summary:
        return None
    return sorted(fid for fid, v in summary.items() if v.get("passed"))


def code_version():
    """Commit of the fork checkout this package runs from, and whether the tree has uncommitted changes."""
    head = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(PACKAGE_ROOT), "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def frames_per_step(traj):
    """How many frames each recorded step advanced the timeline, per play (the timeline restarts at
    each play). A play's first step is reported apart: it also carries the time between play() and
    that step. Measured from the timeline, independent of what the stepping code intends."""
    t, tl = traj.get("t") or [], traj.get("timeline_t") or []
    if len(t) < 2 or len(tl) != len(t):
        return None
    frame = t[1] - t[0]
    plays, start = [], 0
    for i in range(1, len(tl) + 1):
        if i == len(tl) or tl[i] < tl[i - 1] - 1e-6:
            seg = tl[start:i]
            steps = {}
            for a, b in zip(seg[1:-1], seg[2:]):
                k = round((b - a) / frame)
                steps[k] = steps.get(k, 0) + 1
            first = round((seg[1] - seg[0]) / frame) if len(seg) > 1 else None
            plays.append({"first_step_frames": first, "steps": steps})
            start = i
    return {"frame_s": frame, "plays": plays}


REACHED_ENTRY = "[asset_checks] entry reached"  # printed by kit/entry.py on its first line


def started_the_test(out_dir) -> bool:
    """Whether Kit got as far as running our entry script. A Kit that hangs in startup (extension
    registry, material library, shader cache) writes a kit.log without this line and no result.json;
    a test that ran and failed always has it. The marker is printed by the script itself, so an
    unrelated log line cannot stand in for it."""
    log = out_dir / "kit.log"
    return log.exists() and REACHED_ENTRY in log.read_text(errors="replace")


def keep_valid(cell):
    """A finished cell's result when it is valid, else None so the caller re-runs it. `--resume` is
    the one way a run directory is filled in: every cell is either kept whole or replaced whole, so
    a repaired cell can never end up nested inside the one it replaces."""
    path = cell / "result.json"
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return result if result.get("verdict") and not result.get("invalid") else None


def run_one(bench, gpu, env, experiment, asset, out_dir, timeout, capture_px, validated, contact_profile, dump_physics=False, trace_contacts=0, camera="fixed",
            visual_cues=True, startup_retries=1, solver_settings=None, particle_radius=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = envs.gpu_settings(gpu)
    request = {"asset": str(asset), "experiment": experiment, "engine": env.engine, "env": env.name,
               "out_dir": str(out_dir), "expected_settings": expected, "capture_px": capture_px,
               "validated_features": validated, "contact_profile": contact_profile, "dump_physics": dump_physics,
               "trace_contacts": trace_contacts, "camera": camera, "visual_cues": visual_cues, "solver": env.solver, "solver_settings": solver_settings, "particle_radius": particle_radius,
               "code": code_version()}
    (out_dir / "request.json").write_text(json.dumps(request, indent=1))
    cmd = [str(bench / "isaac-run"), env.venv, str(bench / f".venv-{env.venv}" / "bin" / "isaacsim"), env.experience,
           "--exec", f"{ENTRY} {out_dir / 'request.json'}", *envs.kit_flags(gpu)]
    environ = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT), SIMREADY_PHYSICS_RUNTIME=envs.PHYSICS_RUNTIME[env.engine])
    startups = []
    for attempt in range(startup_retries + 1):
        with open(out_dir / "kit.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=environ, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        # Only a Kit that never reached the entry script is retried, and the test itself is never
        # re-run: a test that started and failed keeps its verdict, however it ended.
        if (out_dir / "result.json").exists() or started_the_test(out_dir) or attempt == startup_retries:
            break
        startups.append({"attempt": attempt + 1, "kit_exit": proc.returncode, "timeout_s": timeout,
                         "kit_log": (out_dir / f"kit.startup{attempt + 1}.log").name})
        (out_dir / "kit.log").rename(out_dir / f"kit.startup{attempt + 1}.log")
        print(f"[asset_checks]   Kit never started the test (exit {proc.returncode}); retrying", flush=True)
    problems = []
    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if not result:
        where = "never started the test" if not started_the_test(out_dir) else "started the test and wrote nothing"
        problems.append(f"no result.json: Kit {where} (exit {proc.returncode}, timeout {timeout}s)")
    elif result.get("status") != "done":
        error = (result.get("error") or "").strip()
        problems.append(("error: " + error.splitlines()[-1]) if error else "status not done")
    else:
        stepped = bool((result.get("trajectory") or {}).get("t"))
        if stepped and result.get("engine_observed") != env.engine:
            problems.append(f"engine {result.get('engine_observed')!r} simulated, {env.engine!r} expected")
        if result.get("kit_settings") != expected:
            problems.append(f"Kit settings {result.get('kit_settings')} != {expected}")
        asked = (result.get("solver") or {}).get("requested")
        if stepped and asked != env.solver:
            problems.append(f"solver {asked!r} selected, {env.solver!r} expected")
        ran = ((result.get("solver_seen") or [{}])[-1]).get("solver")
        expected_class = {"mujoco": "SolverMuJoCo", "vbd": "SolverVBD", "xpbd": "SolverXPBD"}.get(env.solver)
        if stepped and expected_class and ran and ran != expected_class:
            problems.append(f"{ran} integrated the scene, {expected_class} expected")
        newton = (result.get("versions") or {}).get("newton") or ""
        if env.newton and not newton.startswith(env.newton):
            problems.append(f"Newton {newton!r}, {env.newton}x expected")
        if result.get("verdict") not in ("skipped",) and not result.get("media") and not result.get("test_exception"):
            problems.append("no media recorded")
        if result.get("media") and not result.get("camera"):
            problems.append("media recorded but the camera placement was not")
        if result.get("media") and visual_cues and not (result.get("look") or {}).get("added"):
            problems.append(f"visual cues requested, not added: {(result.get('look') or {}).get('reason')}")
        if dump_physics and env.engine == "newton" and stepped:
            dumps = result.get("physics_dumps") or []
            plays = len(result.get("solver_seen") or [])
            if len(dumps) != plays or not all(pathlib.Path(d["path"]).is_file() for d in dumps):
                problems.append(f"{len(dumps)} physics dumps for {plays} plays")
        if trace_contacts and env.engine == "newton" and stepped and not result.get("contact_trace"):
            problems.append("contact trace requested, none recorded")
        if stepped:
            stepping = result["frames_per_step"] = frames_per_step(result["trajectory"])
            if stepping is None:
                problems.append("no timeline time recorded: frames per step unknown")
            else:
                off = {k: n for play in stepping["plays"] for k, n in play["steps"].items() if k != 1}
                if off:
                    problems.append("steps advanced " + ", ".join(f"{k} frames x{n}" for k, n in sorted(off.items())) + " (expected 1)")
        expected_source = ["usd"] if env.engine == "physx" else ["fabric"]
        if stepped and result.get("pose_source") != expected_source:
            problems.append(f"poses read from {result.get('pose_source')}, {expected_source} expected under {env.engine}")
    if "[BENCHMARK_COMPAT]" in (out_dir / "kit.log").read_text(errors="replace"):
        problems.append("the benchmark compat bridge is active in this Kit")
    result["invalid"] = problems
    if startups:
        result["startup_retries"] = startups
    # The verdict on the run itself belongs next to the run: without it on disk a later reader --
    # `--resume`, a report, a person -- cannot tell a cell that passed its checks from one that
    # failed them, since Kit's own result.json knows nothing about them.
    if result:
        result_path.write_text(json.dumps(result, indent=1))
    return result


KEY_METRICS = {
    "ground_drop": ("ground_drop_touch_time", "ground_drop_rest_time"),
    "slope_drop": ("slope_drop_horiz_time", "slope_drop_penetrated"),
    "grasp_and_lift": ("grasp_passed", "grasp_total"),
}


MARKS = {"pass": "O", "fail": "X", "skipped": "-"}


def _cell_mark(result):
    """One character for the matrix: what the test said, or ! when the run itself did not count."""
    if result.get("invalid"):
        return "!"
    return MARKS.get(result.get("verdict"), "?")


def _reason(result):
    if result.get("invalid"):
        return "INVALID: " + "; ".join(result["invalid"])
    message = (result.get("message") or "").strip().splitlines()
    if result.get("verdict") == "pass" or not message:
        return ""
    return (message[1] if len(message) > 1 else message[0]).split(": ", 1)[-1].strip()


def summarize(rows, out):
    """<out>/summary.md: a verdict matrix per experiment (assets down, environments across), then
    every cell's reason, then the per-run detail. The matrix is the point -- a run is a comparison
    across environments, and one row per run buries that once there is more than a handful."""
    assets = sorted({asset for asset, _, _ in rows})
    environments = [name for name in envs.ENVIRONMENTS if any(env == name for _, env, _ in rows)]
    experiments = sorted({(r.get("request") or {}).get("experiment") or r.get("test", "?") for _, _, r in rows})
    by_cell = {(asset, (r.get("request") or {}).get("experiment") or r.get("test", "?"), env): r for asset, env, r in rows}
    lines, reasons = [], []
    strips = sorted(p.name for p in (out / "compare").glob("*.mp4")) if (out / "compare").is_dir() else []
    if strips:
        lines += ["Each run is drawn twice from its recorded poses -- the asset's textured mesh (`visual`) and",
                  "the colliders it declares (`collision`) -- and `compare/` holds one strip per asset,",
                  f"experiment and view, environments side by side: {len(strips)} strips.", ""]
    for experiment in experiments:
        lines += [f"## {experiment}", "", "| asset | " + " | ".join(environments) + " |",
                  "|---" * (len(environments) + 1) + "|"]
        for asset in assets:
            marks = []
            for env in environments:
                result = by_cell.get((asset, experiment, env))
                marks.append("" if result is None else _cell_mark(result))
                why = "" if result is None else _reason(result)
                if why:
                    reasons.append(f"- `{asset}` / {env} / {experiment}: {why}")
            lines.append(f"| {asset} | " + " | ".join(marks) + " |")
        reference = "physx" if "physx" in environments else (environments[0] if environments else None)
        agreement = []
        for env in environments:
            if env == reference:
                continue
            pairs = [(by_cell.get((a, experiment, reference)), by_cell.get((a, experiment, env))) for a in assets]
            pairs = [(x, y) for x, y in pairs if x and y and not x.get("invalid") and not y.get("invalid")
                     and x.get("verdict") != "skipped" and y.get("verdict") != "skipped"]
            if pairs:
                agreement.append(f"{env} {sum(x['verdict'] == y['verdict'] for x, y in pairs)}/{len(pairs)}")
        if agreement:
            lines += ["", f"Agreement with {reference} (cells both ran): " + ", ".join(agreement)]
        lines.append("")
    lines += ["Legend: O pass, X fail, - skipped by the test, ! the run did not count, blank not run.", ""]
    seen = {}
    for asset, env, r in rows:
        for note in r.get("notes") or []:
            seen.setdefault((note["note"], note["detail"]), []).append(f"{asset}/{env}")
    if seen:
        lines += ["## Read these verdicts with", ""]
        for (name, detail), where in sorted(seen.items()):
            lines += [f"- **{name}** ({len(where)} runs, e.g. {', '.join(sorted(where)[:3])}): {detail}"]
        lines.append("")
    if reasons:
        lines += ["## Why", ""] + reasons + [""]

    head = ["asset", "env", "experiment", "contact", "solver", "NVIDIA verdict", "NVIDIA metrics",
            "PR #2 criteria on this trajectory", "max dip mm", "tilt deg", "runtime variant", "payload schemas lost"]
    lines += ["## Every run", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for asset, env, r in rows:
        experiment = (r.get("request") or {}).get("experiment") or r.get("test", "?")
        if r.get("invalid"):
            lines.append("| " + " | ".join([asset, env, experiment, "", "", "INVALID: " + "; ".join(r["invalid"])[:160]]
                                           + [""] * (len(head) - 6)) + " |")
            continue
        v, p = r.get("variant") or {}, r.get("pr2_criteria")
        sel = f"{v['selected']['variantSet']}={v['selected']['option']}" if v.get("selected") else ("none declared" if not v.get("declared") else "not declared for this engine")
        lost = sum(len(x) for x in (v.get("payload_schemas_not_composed") or {}).values())
        metrics = r.get("metrics") or {}
        shown = ", ".join(f"{k.split('_', 2)[-1]}={metrics[k]['value']}" for k in KEY_METRICS.get(r["test"], ()) if k in metrics)
        verdict = r["verdict"] + (": " + _reason(r)[:90] if _reason(r) else "")
        pr2 = "" if p is None else (("pass" if p["passed"] else "FAIL: " + p["message"][:50]) + (f" ({p['message'][:60]})" if p["passed"] and p["message"] else ""))
        dip = "" if p is None else f"{p['max_penetration_m'] * 1000:.1f}"
        traj = r.get("trajectory") or {}
        tilt = str(traj["tilt_deg"][-1]) if traj.get("tilt_deg") and r["test"] != "grasp_and_lift" else ""  # a grasp lifts and shakes the asset
        ran = ((r.get("solver_seen") or [{}])[-1]).get("solver") or (r.get("solver") or {}).get("requested", "")
        lines.append("| " + " | ".join([asset, env, experiment, r.get("contact_profile", ""), str(ran), verdict, shown,
                                        pr2, dip, tilt, sel, str(lost)]) + " |")
    notes = [
        "",
        "NVIDIA's tests run unmodified; their verdicts and metrics are the tests' own. PR #2 (newton15_cert.py) runs its",
        "own scenes (a 1 cm drop, a 15 degree walled slope); its column applies PR #2's settle / tunnel / tilt criteria to",
        "NVIDIA's trajectory. `max dip`: deepest collider vertex below the floor; `tilt`: final rotation from the first step.",
    ]
    (out / "summary.md").write_text("\n".join(lines + notes) + "\n")
    print("\n".join(lines[:lines.index("Legend: O pass, X fail, - skipped by the test, ! the run did not count, blank not run.") + 1]))


def draw(bench, cell, asset, result, experiment, timeout):
    """The cell drawn again from what it recorded: the asset's textured mesh and its declared
    colliders placed by the engine's own poses, each as a video beside NVIDIA's own capture.

    Nothing about the experiment changes here -- the test ran and wrote its result already. This
    reads the poses it recorded, writes them as an animated USD, and photographs that file the way
    a deformable run's is photographed, so a rigid strip and a deformable strip are the same picture
    of the same kind of thing.
    """
    usda = cell / "recording.usda"
    code = video.record_rigid(bench, cell / "result.json", asset, usda, cell / "recording.log")
    if code != 0 or not usda.exists():
        print(f"[asset_checks]   no recording written (exit {code}, see recording.log)", flush=True)
        return
    for view, name, frames in video.draw(bench, cell, usda, experiment, timeout, ""):
        result.setdefault("media", []).append({"filename": name, "kind": "video", "role": view})
        print(f"[asset_checks]   {view}: {frames} frames -> {name}", flush=True)
    (cell / "result.json").write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=os.environ.get("SIMREADY_BENCH"), help="simready-bench directory (isaac-run, venvs, GPU)")
    ap.add_argument("--out", required=True, help="new directory for results (with --resume, an existing one)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse an existing --out: keep every cell that already holds a valid result, run the rest")
    ap.add_argument("--envs", default=",".join(envs.DEFAULT_ENVIRONMENTS),
                    help=f"comma-separated; known: {', '.join(envs.ENVIRONMENTS)}")
    ap.add_argument("--experiments", default="drop")
    ap.add_argument("--newton-contact", default="stock", choices=("stock", "pr2"),
                    help="Newton contact settings: stock, or PR #2's (newton15_cert.py) solref/condim/cone/impratio")
    ap.add_argument("--dump-physics", action="store_true",
                    help="Newton runs also write physics_play<n>.npz: the compiled MuJoCo model, its GPU copy, the Newton model and Isaac's config")
    ap.add_argument("--trace-contacts", type=int, default=0, metavar="N",
                    help="Newton runs also record the solver's pad/asset contacts every N physics steps (0: off)")
    ap.add_argument("--no-validation", action="store_true", help="run without NVIDIA's static validation (tests see no validated features)")
    ap.add_argument("--timeout", type=int, default=180, help="seconds per Kit run")
    ap.add_argument("--capture-px", type=int, default=None, help="capture size in px (default: engine-kit's own)")
    ap.add_argument("--plain-scene", action="store_true",
                    help="render NVIDIA's test room as is, without the floor grid and key light")
    ap.add_argument("--camera", default="fixed", choices=("fixed", "follow"),
                    help="fixed: one camera framing the test's whole motion (slope keeps follow); follow: engine-kit's follow camera")
    ap.add_argument("--particle-radius", type=float, default=None, metavar="M",
                    help="radius for a deformable's particles (default: half the median distance to a neighbour; "
                         "Newton's own default of 0.1 m explodes any centimetre-scale asset)")
    ap.add_argument("--solver-setting", action="append", default=[], metavar="NAME=VALUE",
                    help="a setting for the environment's Newton solver, e.g. iterations=30 (repeatable)")
    ap.add_argument("--keep-going", action="store_true",
                    help="finish the matrix even when an environment's first cell does not count (default: stop)")
    ap.add_argument("--no-video", action="store_true",
                    help="skip drawing each run again from its recorded poses (the visual and collision videos and their side-by-side strips)")
    ap.add_argument("assets", nargs="+")
    args = ap.parse_args()
    if not args.bench:
        sys.exit("[asset_checks] --bench or SIMREADY_BENCH is required")
    bench, out = pathlib.Path(args.bench).resolve(), pathlib.Path(args.out).resolve()
    if out.exists() and not args.resume:
        sys.exit(f"[asset_checks] {out} exists; results never mix with an earlier run (use --resume to fill in its missing cells)")
    unknown = set(args.envs.split(",")) - set(envs.ENVIRONMENTS)
    if unknown:
        sys.exit(f"[asset_checks] unknown environments {sorted(unknown)}; known: {list(envs.ENVIRONMENTS)}")
    gpu = int((bench / "GPU").read_text().strip())
    selected = [envs.ENVIRONMENTS[name] for name in args.envs.split(",")]
    for venv in sorted({env.venv for env in selected}):
        preflight(bench, venv)
    solver_settings = {}
    for item in args.solver_setting:
        name, _, value = item.partition("=")
        try:
            solver_settings[name] = int(value) if value.isdigit() else float(value)
        except ValueError:
            solver_settings[name] = value
    rows, first_cell = [], {}
    for asset in args.assets:
        asset = pathlib.Path(asset).resolve()
        validated = None
        if not args.no_validation:
            validated = validated_features(bench, asset, out / asset.stem)
            if validated is None:
                sys.exit(f"[asset_checks] NVIDIA's static validation produced no result for {asset} (see {out / asset.stem / 'validation.log'}); use --no-validation to run without it")
            print(f"[asset_checks] {asset.name}: validated {', '.join(v for v in validated if v.startswith('FET_003')) or 'no FET_003 feature'}", flush=True)
        for env in selected:
            for experiment in args.experiments.split(","):
                cell = out / asset.stem / env.name / experiment
                if args.resume:
                    kept = keep_valid(cell)
                    if kept is not None:
                        rows.append((asset.stem, env.name, kept))
                        print(f"[asset_checks] {asset.name} / {env.name} / {experiment}: kept ({kept['verdict']})", flush=True)
                        continue
                    if cell.exists():  # an invalid or half-written cell is replaced, never merged into
                        shutil.rmtree(cell)
                print(f"[asset_checks] {asset.name} / {env.name} / {experiment} ...", flush=True)
                result = run_one(bench, gpu, env, experiment, asset, cell, args.timeout, args.capture_px, validated, args.newton_contact, args.dump_physics, args.trace_contacts, args.camera, not args.plain_scene,
                                 solver_settings=solver_settings, particle_radius=args.particle_radius)
                rows.append((asset.stem, env.name, result))
                if not args.no_video and (result.get("trajectory") or {}).get("pose"):
                    draw(bench, cell, asset, result, experiment, args.timeout)
                first = ((result.get("message") or "").strip().splitlines() or [""])[0][:120]
                print(f"[asset_checks]   {'INVALID: ' + '; '.join(result['invalid']) if result['invalid'] else (result['verdict'] + ' ' + first).strip()}", flush=True)
                # An environment whose very first cell does not count is broken for every cell in it
                # -- a launch, a flag or a supplement that environment cannot take. Stopping here
                # costs one cell; carrying on has cost a whole matrix twice.
                if env.name not in first_cell:
                    first_cell[env.name] = result["invalid"]
                    if result["invalid"] and not args.keep_going:
                        summarize(rows, out)
                        sys.exit(f"[asset_checks] {env.name}'s first cell did not count: {'; '.join(result['invalid'])}\n"
                                 f"[asset_checks] stopping before the rest of the matrix; fix it and re-run with --resume, "
                                 f"or pass --keep-going to run it anyway")
    if not args.no_video:
        for view in video.VIEWS:
            # `asset_checks.compare`, the compositor that made the 2026-09-19 strips, on the videos
            # drawn from the recordings: one strip per asset and experiment per view.
            subprocess.call([str(bench / ".venv-isaac610" / "bin" / "python"), "-m", "asset_checks.compare", str(out),
                             "--role", view, "--envs", args.envs], env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)})
    summarize(rows, out)
    sys.exit(1 if any(r["invalid"] for _, _, r in rows) else 0)


if __name__ == "__main__":
    main()
