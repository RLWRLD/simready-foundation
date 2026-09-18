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
import signal
import subprocess
import sys

from asset_checks import envs

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


def run_one(bench, gpu, env, experiment, asset, out_dir, timeout, capture_px, validated, contact_profile, dump_physics=False, trace_contacts=0):
    out_dir.mkdir(parents=True)
    expected = envs.gpu_settings(gpu)
    request = {"asset": str(asset), "experiment": experiment, "engine": env.engine, "env": env.name,
               "out_dir": str(out_dir), "expected_settings": expected, "capture_px": capture_px,
               "validated_features": validated, "contact_profile": contact_profile, "dump_physics": dump_physics,
               "trace_contacts": trace_contacts, "code": code_version()}
    (out_dir / "request.json").write_text(json.dumps(request, indent=1))
    cmd = [str(bench / "isaac-run"), env.venv, str(bench / f".venv-{env.venv}" / "bin" / "isaacsim"), env.experience,
           "--exec", f"{ENTRY} {out_dir / 'request.json'}", *envs.kit_flags(gpu)]
    environ = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT), SIMREADY_PHYSICS_RUNTIME=envs.PHYSICS_RUNTIME[env.engine])
    with open(out_dir / "kit.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=environ, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    problems = []
    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if not result:
        problems.append(f"no result.json (Kit exit {proc.returncode}, timeout {timeout}s)")
    elif result.get("status") != "done":
        error = (result.get("error") or "").strip()
        problems.append(("error: " + error.splitlines()[-1]) if error else "status not done")
    else:
        stepped = bool((result.get("trajectory") or {}).get("t"))
        if stepped and result.get("engine_observed") != env.engine:
            problems.append(f"engine {result.get('engine_observed')!r} simulated, {env.engine!r} expected")
        if result.get("kit_settings") != expected:
            problems.append(f"Kit settings {result.get('kit_settings')} != {expected}")
        newton = (result.get("versions") or {}).get("newton") or ""
        if env.newton and not newton.startswith(env.newton):
            problems.append(f"Newton {newton!r}, {env.newton}x expected")
        if result.get("verdict") != "skipped" and not result.get("media"):
            problems.append("no media recorded")
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
    return result


KEY_METRICS = {
    "ground_drop": ("ground_drop_touch_time", "ground_drop_rest_time"),
    "slope_drop": ("slope_drop_horiz_time", "slope_drop_penetrated"),
    "grasp_and_lift": ("grasp_passed", "grasp_total"),
}


def summarize(rows, out):
    head = ["asset", "env", "contact", "NVIDIA test", "NVIDIA verdict", "NVIDIA metrics", "PR #2 criteria on this trajectory",
            "max dip mm", "tilt deg", "runtime variant", "payload schemas lost"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for asset, env, r in rows:
        if r.get("invalid"):
            lines.append("| " + " | ".join([asset, env, "", (r.get("request") or {}).get("experiment", ""), "INVALID: " + "; ".join(r["invalid"])[:160]] + [""] * (len(head) - 5)) + " |")
            continue
        v, p = r.get("variant") or {}, r.get("pr2_criteria")
        sel = f"{v['selected']['variantSet']}={v['selected']['option']}" if v.get("selected") else ("none declared" if not v.get("declared") else "not declared for this engine")
        lost = sum(len(x) for x in (v.get("payload_schemas_not_composed") or {}).values())
        metrics = r.get("metrics") or {}
        shown = ", ".join(f"{k.split('_', 2)[-1]}={metrics[k]['value']}" for k in KEY_METRICS.get(r["test"], ()) if k in metrics)
        message = (r.get("message") or "").strip().splitlines()
        verdict = r["verdict"] + ("" if r["verdict"] == "pass" or not message else ": " + message[0][:90])
        pr2 = "" if p is None else (("pass" if p["passed"] else "FAIL: " + p["message"][:50]) + (f" ({p['message'][:60]})" if p["passed"] and p["message"] else ""))
        dip = "" if p is None else f"{p['max_penetration_m'] * 1000:.1f}"
        traj = r.get("trajectory") or {}
        tilt = str(traj["tilt_deg"][-1]) if traj.get("tilt_deg") and r["test"] != "grasp_and_lift" else ""  # a grasp lifts and shakes the asset
        lines.append("| " + " | ".join([asset, env, r.get("contact_profile", ""), r["test"], verdict, shown, pr2, dip, tilt, sel, str(lost)]) + " |")
    notes = [
        "",
        "NVIDIA's tests run unmodified; their verdicts and metrics are the tests' own. PR #2 (newton15_cert.py) runs its",
        "own scenes (a 1 cm drop, a 15 degree walled slope); its column applies PR #2's settle / tunnel / tilt criteria to",
        "NVIDIA's trajectory. `max dip`: deepest collider vertex below the floor; `tilt`: final rotation from the first step.",
    ]
    (out / "summary.md").write_text("\n".join(lines + notes) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=os.environ.get("SIMREADY_BENCH"), help="simready-bench directory (isaac-run, venvs, GPU)")
    ap.add_argument("--out", required=True, help="new directory for results")
    ap.add_argument("--envs", default=",".join(envs.ENVIRONMENTS))
    ap.add_argument("--experiments", default="drop")
    ap.add_argument("--newton-contact", default="stock", choices=("stock", "pr2"),
                    help="Newton contact settings: stock, or PR #2's (newton15_cert.py) solref/condim/cone/impratio")
    ap.add_argument("--dump-physics", action="store_true",
                    help="Newton runs also write physics_play<n>.npz: the compiled MuJoCo model, its GPU copy, the Newton model and Isaac's config")
    ap.add_argument("--trace-contacts", type=int, default=0, metavar="N",
                    help="Newton runs also record the solver's pad/asset contacts every N physics steps (0: off)")
    ap.add_argument("--no-validation", action="store_true", help="run without NVIDIA's static validation (tests see no validated features)")
    ap.add_argument("--timeout", type=int, default=180, help="seconds per Kit run")
    ap.add_argument("--capture-px", type=int, default=512)
    ap.add_argument("assets", nargs="+")
    args = ap.parse_args()
    if not args.bench:
        sys.exit("[asset_checks] --bench or SIMREADY_BENCH is required")
    bench, out = pathlib.Path(args.bench).resolve(), pathlib.Path(args.out).resolve()
    if out.exists():
        sys.exit(f"[asset_checks] {out} exists; results never mix with an earlier run")
    unknown = set(args.envs.split(",")) - set(envs.ENVIRONMENTS)
    if unknown:
        sys.exit(f"[asset_checks] unknown environments {sorted(unknown)}; known: {list(envs.ENVIRONMENTS)}")
    gpu = int((bench / "GPU").read_text().strip())
    selected = [envs.ENVIRONMENTS[name] for name in args.envs.split(",")]
    for venv in sorted({env.venv for env in selected}):
        preflight(bench, venv)
    rows = []
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
                print(f"[asset_checks] {asset.name} / {env.name} / {experiment} ...", flush=True)
                result = run_one(bench, gpu, env, experiment, asset, out / asset.stem / env.name / experiment, args.timeout, args.capture_px, validated, args.newton_contact, args.dump_physics, args.trace_contacts)
                rows.append((asset.stem, env.name, result))
                first = ((result.get("message") or "").strip().splitlines() or [""])[0][:120]
                print(f"[asset_checks]   {'INVALID: ' + '; '.join(result['invalid']) if result['invalid'] else (result['verdict'] + ' ' + first).strip()}", flush=True)
    summarize(rows, out)
    sys.exit(1 if any(r["invalid"] for _, _, r in rows) else 0)


if __name__ == "__main__":
    main()
