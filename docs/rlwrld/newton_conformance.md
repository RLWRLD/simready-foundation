# SimReady props on Newton 1.5.2: conformance runner, findings and collision variants

RLWRLD addition (2026-09-16, rescoped 2026-09-28), tools in `nv_core/testing_tools/rlwrld-newton-conformance/`.

## Why

When this was written the SimReady runtime tests only existed for Isaac Sim / PhysX; 2026.07.1 added a Newton path,
which does not hold grasps on the Newton 1.5.2 stack DexBench runs (see "Upstream kit on Newton" below). DexBench runs its scenes on Newton as well,
and a package that passes `Prop-Robotics-Neutral 2.0.0` and the PhysX grasp-and-lift can still be
unusable there: MuJoCo-Warp collides every mesh as one 64-vertex convex hull, so a bowl is a lid, a
`convexDecomposition` depends on CoACD at every load, and a planar piece is refused. We needed the same
three runtime tests on Newton, for all 171 handled props, to know which packages hold and what the
engine needs that PhysX did not.

| `godmiddag_bowl_16`, package colliders (one hull): the pads close inside the "lid" and tip the bowl | the same bowl with its Newton variant: pinched at the rim, lifted, shaken, set down |
|---|---|
| ![](newton/godmiddag_bowl_16_hull_fail.png) | ![](newton/godmiddag_bowl_16_variant_pass.png) |

## What

`newton15_cert.py` runs a procedure similar to NVIDIA's FET003 drops and FET005 grasp-and-lift, with
different parameters, on the standalone Newton 1.5.2 wheel: parse with Newton's USD importer, drop 1 cm onto
a floor, drop onto a 15° walled slope, and grasp-and-lift on the authored `grasp_identifier_01` with a two-pad
gantry (close, lift by max(0.3 m, 2 × longest edge), hold 1 s, shake 1 cm along X at 2 Hz for 1.5 s, hold,
open). Its failure messages reuse the names of NVIDIA's phases ("pads touched (no object)", "did not rise",
"dropped"), but the thresholds behind them differ, so a verdict here is not an NVIDIA verdict and PhysX and
Newton results are comparable only in kind. Every asset runs in its own subprocess; a crash is recorded, not
fatal. `newton15_cert_report.py` writes the comparison report. `make_newton_variant.py` authors the
per-package Newton collision layer described below.

Where the runner differs from NVIDIA's documented procedure (`simready_benchmark_kit_suite/docs/`
`fet003/ground-drop.md` 3.1.0, `fet003/slope-drop.md` 3.0.0, `fet005/grasp-and-lift.md` 1.3.0):

| | NVIDIA simready-benchmark | this runner |
|---|---|---|
| physics rate | 240 Hz | 1 kHz, control at 100 Hz |
| ground drop height | 2 × the asset's bounding-box height | 1 cm above the floor |
| ground drop, at rest | bounding-box centre and corners within 2 cm for 2 s, within 8 s of first contact (10 s cap) | touching the floor (lowest collision point within 5 mm) with speed < 2 cm/s and < 0.5 rad/s for 0.5 s (6 s cap) |
| penetration | bounding-box bottom more than 0.1 m below the floor | collision points more than 1 cm below the surface |
| slope | 45°, open onto a flat floor, friction 0.5 / 0.4; passes on a 1 cm slide without penetration | 15°, walled on four sides, friction 0.5; passes when the asset comes to rest on it |
| grasp lines | every `grasp_identifier_*`; the asset passes if any one does | `grasp_identifier_01` only |
| settle before the grasp | centroid within 2 mm for 1 s, up to 3 s | speed < 2 cm/s and < 0.5 rad/s for 0.5 s, up to 3 s |
| closing | at most 0.04 m/s with 5 mm preload per jaw, then 0.3 s settle | position target at the point where the pads meet, force capped at 5 × weight |
| lift | 2 × longest edge of the grasped body, clamped to 0.3–0.5 m, smoothstep over 1 s | max(0.3 m, 2 × longest edge of the asset's bounding box), no upper cap, linear over 1 s |
| hold | fails if the centroid drops > 0.10 m or comes within 0.02 m of the floor | fails if the grasped point slips > 3 cm from the pad centre |
| shake | circular horizontal orbit, 1 cm radius, 2 Hz, 1.5 s, 0.25 s envelope; fails at 5 cm separation | 1 cm sine along X, 2 Hz, 1.5 s, no envelope; fails at 3 cm slip |
| release | open over 0.5 s, pad collision disabled; the body must fall ≥ max(5 cm, its bounding-box height) within 3 s | open for 1 s; the body must fall at least half the height it was lifted |
| contact settings | engine defaults, one PD actuator per jaw on Newton | tuned: 4 ms solref, elliptic cone, `impratio` 10, multi-CCD, pad `condim 4`, 1 kg finger armature |

## Results on the 171 handled props

| test | pass | fail |
|---|---|---|
| parse (Newton importer) | 171 | 0 |
| ground drop | 171 | 0 |
| slope drop (15°, walled) | 170 | 1 |
| grasp-and-lift, line on the body, tuned contact settings | 152 | 19 |
| grasp-and-lift, line in the stage frame | 131 | 40 |
| grasp-and-lift, stock Newton contact settings | 0 | 171 |

Twenty-six packages disagree with PhysX. The ones PhysX passes and Newton fails are the `sdf` tableware
(five bowls and plates), thin cutlery, a spark plug and a roundline cylinder that tip or roll off their
hull, and a chips bag. The ones Newton passes and PhysX fails are props whose floor-standing PhysX verdict
was a rig limit (the split shaft collar, several thin parts).

| `dragon_spoon_19cm` on its hull: the pads close on nothing | with the variant: held through the shake |
|---|---|
| ![](newton/dragon_spoon_19cm_hull_fail.png) | ![](newton/dragon_spoon_19cm_variant_pass.png) |

| `harmynta_deep_plate_22` on its hull: dropped at lift | with the variant: held |
|---|---|
| ![](newton/harmynta_deep_plate_22_hull_fail.png) | ![](newton/harmynta_deep_plate_22_variant_pass.png) |

## What Newton needs from an asset that PhysX did not

1. Every collider becomes convex. `sdf` and unauthored mesh colliders silently turn into one 64-vertex hull.
2. No zero-thickness or sliver pieces: 1002 pieces in 16 packages had to be thickened to 1 mm by the runner.
3. CoACD is a hard dependency for `convexDecomposition`, and it runs on every load.
4. Contact stiffness is mass-scaled and the stock settings cannot hold a pinch grasp (0 of 171); the runner uses solref 4 ms, an elliptic cone, `impratio` 10, `condim 4` on the pads.
5. Torsional friction is off by default; without `condim 4` on the gripper every off-centre pinch spins out.
6. Soft contacts creep: props on the 15° slope slide about 1 cm/s and held objects creep millimetres in the jaws, so rest and hold checks need tolerances.
7. Authored poses are not always rest poses (seven packages tip when dropped 1 cm), so the grasp line has to ride with the body.
8. Mass, inertia and bound physics materials are honoured; multi-body vendor packages import as articulations.
9. Hulls are capped at 64 vertices, coarser than PhysX: cylinders facet, rims flatten.
10. `UsdPhysics.LoadUsdPhysicsFromRange` races on a body with many collision prims (heap corruption in about one load in four); `PXR_WORK_THREAD_LIMIT=1` before opening the stage avoids it.

## Newton collision variants

`make_newton_variant.py` writes `usd/variants/<name>_newton.usd` inside every package whose colliders the
importer cannot honour: a flattened, self-contained copy in which each such collider is replaced by
pre-decomposed convex pieces (CoACD with the importer's own light search, merge on, at most 32 hulls per
collider and 64 vertices per hull, pieces thinner than 1 mm extruded to 1 mm), placed under the rigid body
in the body's frame with the original physics material and a volume-split mass. The original mesh keeps
rendering. Each variant is parsed back through Newton's importer in a fresh interpreter before it is
accepted; the sidecar records the pieces, the CoACD settings and the certification verdict.

| | package USD | Newton variant |
|---|---:|---:|
| colliders the importer replaced / thickened | 87 / 1002 | 0 / 0 |
| parse time, all 85 | 95 s | 43 s |
| ground drop pass | 85 | 85 |
| slope drop pass | 85 | 83 |
| grasp holds, line on the body | 74 | 79 |

| `ikea_365_bowl_rounded_16` hull vs variant | `hex_nut_m20` variant (32 pieces) | `sus304_flat_bar`: a 3 mm sheet the pads cannot pinch on any engine |
|---|---|---|
| ![](newton/ikea_365_bowl_rounded_16_hull_fail.png) ![](newton/ikea_365_bowl_rounded_16_variant_pass.png) | ![](newton/hex_nut_m20_variant_pass.png) | ![](newton/sus304_flat_bar_fail.png) |

## Upstream kit on Newton (2026-09-28)

`simready-benchmark` 2026.07.1 runs FET005 grasp-and-lift on Newton with the package's Newton runtime variant
enabled, one PD actuator per jaw and every contact parameter at the engine default. On Isaac Sim 6.1.0-rc.26 /
Newton 1.5.2 it failed upstream's own orange and tool box samples and 8 of 8 DexBench props: the pads close
through the object (on the R2 can, from 94 mm to 6 mm at the same 19 N grip). On Isaac Sim 6.0.1 / Newton 1.2.1
the same kit passed the orange, the can and a box, and lifted but dropped a bowl. This runner with stock settings
fails the same 8 of 8; with its tuned settings it holds all 8 (line on the body). The kit's Newton verdicts are
therefore not a substitute on Newton 1.5.2. Two setup notes from that run: upstream's documented engines.toml
(`isaac-sim.sh` with `experience = "isaacsim.exp.full.newton"`) started the PhysX app on standalone Isaac Sim
while reporting a Newton run; pointing the Newton engine at `isaac-sim.newton.sh` started Newton.

## What it is not

Not an engine plugin for `simready-benchmark`, and not the same test: a standalone runner with a similar
procedure and different parameters (table above), so its verdicts sit beside NVIDIA's rather than replace
them. Not a SimReady requirement: a package's profile pass stays a PhysX/Kit statement,
and the Newton verdicts are recorded beside it. Its grasp verdicts assume its tuned contact settings (solref
4 ms, elliptic cone, `impratio` 10, multi-CCD, pad `condim 4`), which differ from DexBench-Arena's teleop
profiles (MuJoCo-Warp contacts, `ccd_iterations` 50, multi-CCD, default stiffness).
