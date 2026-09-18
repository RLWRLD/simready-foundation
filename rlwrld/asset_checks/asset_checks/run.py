# SPDX-License-Identifier: Apache-2.0
"""Run physics checks on assets in the three environments; one fresh Kit process per run.

    PYTHONPATH=rlwrld/asset_checks python3 -m asset_checks.run --bench <simready-bench> --out <new dir> \
        [--envs physx,newton12,newton15] [--experiments drop] <asset.usd> [...]

A run counts only if its result.json says "done" and Kit demonstrably had what was asked: the
engine that actually simulated, the GPU settings, the Newton version, frames captured, and no
[BENCHMARK_COMPAT] line (that bridge must not be installed). Anything else is reported INVALID.
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


def run_one(bench, gpu, env, experiment, asset, out_dir, timeout, capture_px):
    out_dir.mkdir(parents=True)
    expected = envs.gpu_settings(gpu)
    request = {"asset": str(asset), "experiment": experiment, "engine": env.engine, "env": env.name,
               "out_dir": str(out_dir), "expected_settings": expected, "capture_px": capture_px}
    (out_dir / "request.json").write_text(json.dumps(request, indent=1))
    cmd = [str(bench / "isaac-run"), env.venv, str(bench / f".venv-{env.venv}" / "bin" / "isaacsim"), env.experience,
           "--exec", f"{ENTRY} {out_dir / 'request.json'}", *envs.kit_flags(gpu)]
    environ = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT))
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
        if result.get("engine_observed") != env.engine:
            problems.append(f"engine {result.get('engine_observed')!r} simulated, {env.engine!r} expected")
        if result.get("kit_settings") != expected:
            problems.append(f"Kit settings {result.get('kit_settings')} != {expected}")
        newton = (result.get("versions") or {}).get("newton") or ""
        if env.newton and not newton.startswith(env.newton):
            problems.append(f"Newton {newton!r}, {env.newton}x expected")
        if not result.get("frames"):
            problems.append("no frames captured")
        expected_source = ["usd"] if env.engine == "physx" else ["fabric"]
        if result.get("pose_source") != expected_source:
            problems.append(f"poses read from {result.get('pose_source')}, {expected_source} expected under {env.engine}")
    if "[BENCHMARK_COMPAT]" in (out_dir / "kit.log").read_text(errors="replace"):
        problems.append("the benchmark compat bridge is active in this Kit")
    result["invalid"] = problems
    return result


def summarize(rows, out):
    head = ["asset", "env", "NVIDIA ground_drop", "PR #2 criteria on this trajectory", "touch s", "rest s", "max dip mm", "tilt deg", "runtime variant", "payload schemas lost"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for asset, env, r in rows:
        if r.get("invalid"):
            lines.append("| " + " | ".join([asset, env, "INVALID: " + "; ".join(r["invalid"])[:160]] + [""] * (len(head) - 3)) + " |")
            continue
        n, p, v = r["nvidia"], r["pr2_criteria"], r["variant"]
        sel = f"{v['selected']['variantSet']}={v['selected']['option']}" if v.get("selected") else ("none declared" if not v.get("declared") else "not declared for this engine")
        lost = sum(len(x) for x in v.get("payload_schemas_not_composed", {}).values())
        lines.append(" | ".join([
            f"| {asset}", env, ("pass" if n["passed"] else "FAIL: " + n["message"][:70]),
            ("pass" if p["passed"] else "FAIL: " + p["message"][:50]) + (f" ({p['message'][:60]})" if p["passed"] and p["message"] else ""),
            str(n["touch_time_s"]), str(n["rest_time_s"]), f"{p['max_penetration_m'] * 1000:.1f}", str(p["tilt_deg"]), sel, str(lost),
        ]) + " |")
    notes = [
        "",
        "NVIDIA ground_drop drops from twice the bounding-box height; PR #2 (newton15_cert.py) drops from 1 cm.",
        "The PR #2 column applies PR #2's settle / tunnel / tilt criteria to this (NVIDIA) trajectory; it is not a PR #2 run.",
        "`max dip` is the deepest collider vertex below the floor over the run; `tilt` is the final rotation from the drop pose.",
    ]
    (out / "summary.md").write_text("\n".join(lines + notes) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=os.environ.get("SIMREADY_BENCH"), help="simready-bench directory (isaac-run, venvs, GPU)")
    ap.add_argument("--out", required=True, help="new directory for results")
    ap.add_argument("--envs", default=",".join(envs.ENVIRONMENTS))
    ap.add_argument("--experiments", default="drop")
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
        for env in selected:
            for experiment in args.experiments.split(","):
                print(f"[asset_checks] {asset.name} / {env.name} / {experiment} ...", flush=True)
                result = run_one(bench, gpu, env, experiment, asset, out / asset.stem / env.name / experiment, args.timeout, args.capture_px)
                rows.append((asset.stem, env.name, result))
                print(f"[asset_checks]   {'INVALID: ' + '; '.join(result['invalid']) if result['invalid'] else result['nvidia']['message']}", flush=True)
    summarize(rows, out)
    sys.exit(1 if any(r["invalid"] for _, _, r in rows) else 0)


if __name__ == "__main__":
    main()
