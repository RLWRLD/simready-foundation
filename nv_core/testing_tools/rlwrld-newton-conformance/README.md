# RLWRLD Newton conformance tools

The SimReady runtime tests (FET003 drops, FET005 grasp-and-lift) run on Isaac Sim / PhysX. DexBench
also runs its scenes on Newton, so we needed the same three tests on Newton to learn which
`Prop-Robotics-Neutral` packages hold there and what the engine needs from an asset that PhysX did
not. Everything here runs on the standalone Newton 1.5.2 wheel (MuJoCo-Warp rigid solver, warp 1.17),
not inside Kit.

| file | what |
|---|---|
| `newton15_cert.py` | The runner. Per package: parse with Newton's USD importer, ground drop (1 cm), slope drop (15°, walled), grasp-and-lift on the authored `grasp_identifier_01` with a two-pad gantry (close, lift, hold 1 s, shake 1 cm at 2 Hz, hold, open). Verdict wording follows NVIDIA's. Driver mode runs every asset in its own subprocess (a crash is a finding, not the end of the run). |
| `newton15_cert_report.py` | Turns the merged `results.json` (plus the PhysX verdicts) into the report: totals, Newton-vs-PhysX disagreements, drop tests, collider approximations the importer could not honour. |
| `make_newton_variant.py` | Authors `usd/variants/<name>_newton.usd` inside a package: a flattened copy whose `sdf`, `convexDecomposition` and unauthored mesh colliders become pre-decomposed convex pieces (CoACD, ≤ 32 hulls per collider, ≤ 64 vertices per hull, pieces thinner than 1 mm extruded to 1 mm), parsed back through Newton's importer before it is accepted. Records itself in the package sidecar and changelog (`rlwrld_sidecar.py`). |
| `examples/newton15_soft_grasp.py`, `examples/newton15_polybag_grasp.py` | Grasp-and-deform demos for tetrahedral (fruit) and cloth (polybag) packages on Newton's VBD soft-body solver: the same pad gantry, on assets SimReady has no runtime test for. |

## Running

```bash
python -m venv newton15 && newton15/bin/pip install "newton==1.5.2" "newton[importers]" usd-core imageio[ffmpeg] pyglet
# the three tests over a manifest of package USDs (one path per line, relative to --assets-root)
newton15/bin/python newton15_cert.py --assets-root <pool> --manifest packages.txt --out out/ --tests parse,ground_drop,slope_drop,grasp_and_lift
# the report, against the PhysX verdicts of the same packages
newton15/bin/python newton15_cert_report.py --main out/results.json --physx grasp_verdicts.json --fet003 fet003_results.json --out report.md
# Newton collision variants for the packages the importer cannot honour
newton15/bin/python make_newton_variant.py --assets-root <pool> --candidates candidates.json --write
```

`PXR_WORK_THREAD_LIMIT=1` is set by the runner before any stage opens: `UsdPhysics.LoadUsdPhysicsFromRange`
races on a rigid body with many collision prims (heap corruption in about one load of four with usd-core
26.3 on a 32-piece body; never with the work pool single-threaded).

## What it found on 171 packages (2026-09-16)

| test | pass / 171 |
|---|---|
| parse | 171 |
| ground drop | 171 |
| slope drop | 170 (a curved single-hull fork rocks and creeps) |
| grasp-and-lift, line on the body | 152 |
| grasp-and-lift, stock Newton contact settings | 0 |

The 19 grasp failures were almost all colliders: MuJoCo-Warp reduces every mesh collider to one 64-vertex
convex hull, so an `sdf` bowl becomes a lid and a `convexDecomposition` depends on CoACD running at load.
With the Newton variants (85 packages, 2460 pieces) the bowls, plates and the dragon cutlery hold, grasp
holds go from 74 to 79 of the 85, and the importer replaces or thickens nothing. The full write-up with
frame strips is `docs/rlwrld/newton_conformance.md`.
