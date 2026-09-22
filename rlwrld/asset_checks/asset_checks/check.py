# SPDX-License-Identifier: Apache-2.0
"""Run one experiment on one asset in one engine, and say what happened.

    python -m asset_checks.check <asset.usd> --experiment drop --engine newton1.5 --solver vbd

Four things are named and nothing else has to be: the USD, the experiment, the engine and the
solver. Everything else is read -- whether the asset is rigid or deformable comes from its own
schemas, which runner drives it follows from that, and what it is made of comes from the USD.

**The engine carries its version.** Newton 1.2.1 and 1.5.0 are not interchangeable and neither is
free-standing: `isaacsim-core` pins `newton[sim]==1.2.1` to Isaac 6.0.1 and `==1.5.0` to 6.1.0, so
a Newton version *is* an Isaac build. `envs.py` says the same thing in its docstring -- a result
means nothing until the build, the engine and the solver are all named -- so `--engine newton1.5`
names two of the three and `--solver` the last.

A combination that cannot work is refused before anything is launched, with the reason: MuJoCo is
a rigid-body engine and will not simulate a deformable, and not every experiment means something
for both kinds.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from asset_checks import envs, experiments

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent


def engines():
    """{command-line engine name: {solver: environment}} -- derived from `envs.ENVIRONMENTS`.

    The name a person types is the engine and its version: `physx`, `newton1.2`, `newton1.5`. Each
    environment already carries all of it, so this is a reading of that table rather than a second
    copy of it.
    """
    out = {}
    for env in envs.ENVIRONMENTS.values():
        name = "physx" if env.engine == "physx" else "newton" + env.newton.rstrip(".")
        if env.engine == "physx" and env.isaac != "6.1.0":
            name = f"physx{env.isaac}"
        out.setdefault(name, {})[env.solver] = env
    return out


READ_ASSET = """
import sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {native!r})
from pxr import Usd
from asset_checks.kit.scene import asset_kinds
import usd_deformable
stage = Usd.Stage.Open(sys.argv[1])
kinds = sorted(asset_kinds(stage, stage.GetDefaultPrim().GetPath()))
print('KINDS', ' '.join(kinds))
if 'deformable' in kinds:
    reason = usd_deformable.why_not_one_body(stage, sys.argv[1])
    if reason:
        print('REFUSE', reason)
"""


def read_asset(bench, asset):
    """What the asset is, and whether it is shaped like something this pipeline can run.

    Both answers come from the asset's own schemas, read once, in the venv that has USD. The
    second is `usd_deformable`'s rule rather than a copy of it, so a refusal here says exactly what
    a runner would have said -- only before a Kit launch instead of after one.
    """
    out = subprocess.run(
        [str(bench / ".venv-isaac610" / "bin" / "python"), "-c",
         READ_ASSET.format(root=str(PACKAGE_ROOT), native=str(HERE / "native")), str(asset)],
        capture_output=True, text=True)
    kinds = None
    for line in out.stdout.splitlines():
        if line.startswith("KINDS"):
            kinds = set(line.split()[1:])
        elif line.startswith("REFUSE"):
            raise SystemExit(line[len("REFUSE "):])
    if kinds is None:
        raise SystemExit(f"could not read what {asset} is:\n{(out.stdout + out.stderr).strip()[-600:]}")
    return kinds


def main():
    known = engines()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="experiments: " + "; ".join(
                                     f"{n} -- {m.SUMMARY}" for n, m in sorted(experiments.ALL.items())),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset", help="the USD to evaluate")
    ap.add_argument("--experiment", required=True, choices=sorted(experiments.ALL),
                    help="what to do to it")
    ap.add_argument("--engine", required=True, choices=sorted(known),
                    help="the physics engine and its version (a Newton version is an Isaac build)")
    ap.add_argument("--solver", required=True,
                    help="what integrates the scene: " + ", ".join(
                        sorted({s for by_solver in known.values() for s in by_solver})))
    ap.add_argument("--out", help="the run directory; the cell lands in "
                     "<out>/<asset>/<env>/<experiment>/ (default: ./results)")
    ap.add_argument("--bench", default="/home/wongyun/Workspace/Research/Robotics/simready-bench",
                    help="simready-bench: the venvs, isaac-run and the GPU pin")
    args = ap.parse_args()

    bench = pathlib.Path(args.bench).resolve()
    asset = pathlib.Path(args.asset).resolve()
    if not asset.is_file():
        raise SystemExit(f"no such asset: {asset}")

    if args.solver not in known[args.engine]:
        raise SystemExit(f"{args.engine} does not offer the {args.solver} solver; it has "
                         + ", ".join(sorted(known[args.engine])))
    env = known[args.engine][args.solver]
    experiment = experiments.get(args.experiment)

    kinds = read_asset(bench, asset)
    if not kinds:
        raise SystemExit(f"{asset.name} declares neither a rigid body nor a deformable; there is "
                         f"nothing here to simulate")
    kind = "deformable" if "deformable" in kinds else "rigid"
    print(f"[check] {asset.name} is {kind} ({', '.join(sorted(kinds))} declared)", flush=True)

    # Refusals, before anything is launched, each saying what it is that cannot be done.
    from asset_checks.kit.scene import SOLVER_SIMULATES

    if kind not in SOLVER_SIMULATES.get(env.solver, set()):
        raise SystemExit(f"{env.solver} cannot simulate a {kind} asset: it does "
                         + ", ".join(sorted(SOLVER_SIMULATES.get(env.solver, ()))) + " only. "
                         f"For a {kind} asset on {args.engine}, the solvers are "
                         + ", ".join(sorted(s for s, e in known[args.engine].items()
                                            if kind in SOLVER_SIMULATES.get(e.solver, set()))))
    if kind not in experiment.KINDS:
        raise SystemExit(f"the {experiment.NAME} experiment is for "
                         + " and ".join(sorted(experiment.KINDS)) + f" assets, and {asset.name} is "
                         f"{kind}. For a {kind} asset the experiments are "
                         + ", ".join(sorted(experiments.for_kind(kind))))

    # `--out` is the run directory. Both runners lay the same tree inside it --
    # <out>/<asset>/<env>/<experiment>/ -- and put the side-by-side strips in <out>/compare/, so a
    # second engine or a second experiment written to the same `--out` joins the same comparison.
    out = pathlib.Path(args.out).resolve() if args.out else pathlib.Path("results").resolve()
    cell = out / asset.stem / env.name / experiment.NAME
    print(f"[check] {experiment.NAME} on {env.name} (Isaac {env.isaac}, {env.engine}"
          + (f" {env.newton.rstrip('.')}" if env.newton else "") + f", {env.solver}) -> {cell}", flush=True)

    # This cell is run again whatever is there; every other cell in the run directory is left
    # alone. The rigid runner refuses an existing directory outright, because a *matrix* must not
    # mix with an earlier one -- `--resume` is how it is told that one cell is meant, and clearing
    # the cell first is what makes "again" mean again rather than "keep what you have".
    if cell.exists():
        shutil.rmtree(cell)
    if kind == "deformable":
        command = [sys.executable, str(HERE / "native" / "run.py"), str(asset),
                   "--out", str(out), "--envs", env.name, "--experiments", experiment.NAME]
    else:
        command = [sys.executable, "-m", "asset_checks.run", "--bench", str(bench),
                   "--out", str(out), "--envs", env.name,
                   "--experiments", experiment.NAME, str(asset)]
        if out.exists():
            command.insert(-1, "--resume")
    return subprocess.call(command, cwd=str(PACKAGE_ROOT),
                           env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)})


if __name__ == "__main__":
    sys.exit(main())
