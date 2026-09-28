# RLWRLD Newton conformance tools

Grasp and drop checks for SimReady props on **Newton 1.5.2 with tuned contact settings**, and the tool that
authors their Newton collision pieces. DexBench-Arena runs Isaac Sim 6.1 with Newton 1.5.2; these tools answer
"will this package hold in our stack", which the upstream kit cannot answer there (see below). Everything
runs on the standalone Newton 1.5.2 wheel (MuJoCo-Warp rigid solver, warp 1.17), not inside Kit.

| file | what |
|---|---|
| `newton15_cert.py` | The runner. Per package: parse with Newton's USD importer, ground drop (1 cm), slope drop (15°, walled), grasp-and-lift on the authored `grasp_identifier_01` with a two-pad gantry (close, lift, hold 1 s, shake 1 cm along X at 2 Hz, hold, open). A procedure similar to NVIDIA's FET003 / FET005 with different parameters (drop height, slope angle, shake path, lift cap, settle and slip criteria; see the table in `docs/rlwrld/newton_conformance.md`); failure messages reuse NVIDIA's phase names, not its thresholds. Driver mode runs every asset in its own subprocess (a crash is a finding, not the end of the run). |
| `newton15_cert_report.py` | Turns the merged `results.json` (plus the PhysX verdicts) into the report: totals, Newton-vs-PhysX disagreements, drop tests, collider approximations the importer could not honour. |
| `test_newton_conformance_tools.py` | Self-checks on synthetic stages (usd-core, numpy, scipy; no Newton or GPU): degenerate hulls, per-collider holders, disabled colliders, inherited physics materials, a rejected regeneration keeping the accepted variant, slope placement, stale results after a worker crash, the report's drop sentence. `python test_newton_conformance_tools.py` or pytest. |
| `make_newton_variant.py` | Authors `usd/variants/<name>_newton.usd` inside a package: a flattened copy whose `sdf`, `convexDecomposition` and unauthored mesh colliders become pre-decomposed convex pieces (CoACD, ≤ 32 hulls per collider, ≤ 64 vertices per hull, pieces thinner than 1 mm extruded to 1 mm), parsed back through Newton's importer before it is accepted. Records itself in the package sidecar and changelog (`rlwrld_sidecar.py`). |

## Running

```bash
python -m venv newton15 && newton15/bin/pip install "newton==1.5.2" "newton[importers]" usd-core imageio[ffmpeg] pyglet
# parse + both drops + grasp (line in the stage frame) over a manifest of package USDs (one path per line, relative to --assets-root)
newton15/bin/python newton15_cert.py --assets-root <pool> --manifest packages.txt --out out/
# optional: the grasp with the line on the body, and with stock Newton contact settings
newton15/bin/python newton15_cert.py --assets-root <pool> --manifest packages.txt --out out_bodyline/ --tests grasp_and_lift --line-frame body --no-video
newton15/bin/python newton15_cert.py --assets-root <pool> --manifest packages.txt --out out_defaults/ --tests grasp_and_lift --no-video \
    --contact-timeconst 0.02 --finger-armature 0 --cone pyramidal --impratio 1 --no-multiccd --pad-condim 3
# the report (writes report/results.json and report/summary.md), against the PhysX verdicts of the same packages
newton15/bin/python newton15_cert_report.py --main out/results.json --bodyline out_bodyline/results.json --defaults out_defaults/results.json \
    --physx grasp_verdicts.json --fet003 fet003_results.json --assets-root <pool> --manifest packages.txt --out-dir report/
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

## Relation to the upstream kit (2026.07.1)

SimReady Foundation 2026.07.1's `simready-benchmark` runs FET003 drops and FET005 grasp-and-lift on Newton as
well (`SIMREADY_PHYSICS_RUNTIME=Newton`, Newton runtime variant enabled). Its Newton gripper sets one PD
actuator per jaw and `nconmax`, and leaves every contact parameter at the engine's defaults. Compared on 2026-09-28
(FET005 grasp-and-lift, same packages):

| stack | upstream samples (orange, tool box) | 8 DexBench props |
|---|---|---|
| upstream kit, Isaac Sim 6.1.0-rc.26 / Newton 1.5.2 (DexBench's stack) | fail (pads close through the object) | 0 / 8 |
| upstream kit, Isaac Sim 6.0.1 / Newton 1.2.1 | orange passes | 2 / 3 run |
| this runner, stock Newton contact settings | – | 0 / 8 |
| this runner, tuned settings, line on the body | – | 8 / 8 |

So on Newton 1.5.2 a stock-settings grasp does not hold even upstream's own samples; the kit's Newton verdicts
are meaningful on 6.0.1, not on the stack DexBench runs. This runner's verdicts assume its tuned settings:
per-shape solref time constant 4 ms, elliptic friction cone with `impratio` 10, multi-CCD, `condim 4` on the
pads and 1 kg finger armature. They are not DexBench-Arena's teleop profile settings (MuJoCo-Warp contacts,
`ccd_iterations` 50, multi-CCD, default contact stiffness); treat a pass here as "the asset can be held on
Newton 1.5.2 with contact tuning", not as a SimReady profile result.

