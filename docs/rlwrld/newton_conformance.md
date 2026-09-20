# SimReady props on Newton 1.5.2: conformance runner, findings and collision variants

RLWRLD addition (2026-09-16), tools in `nv_core/testing_tools/rlwrld-newton-conformance/`.

## Why

The SimReady runtime tests only exist for Isaac Sim / PhysX. DexBench runs its scenes on Newton as well,
and a package that passes `Prop-Robotics-Neutral 2.0.0` and the PhysX grasp-and-lift can still be
unusable there: MuJoCo-Warp collides every mesh as one 64-vertex convex hull, so a bowl is a lid, a
`convexDecomposition` depends on CoACD at every load, and a planar piece is refused. We needed the same
three runtime tests on Newton, for all 171 handled props, to know which packages hold and what the
engine needs that PhysX did not.

| `godmiddag_bowl_16`, package colliders (one hull): the pads close inside the "lid" and tip the bowl | the same bowl with its Newton variant: pinched at the rim, lifted, shaken, set down |
|---|---|
| ![](newton/godmiddag_bowl_16_hull_fail.png) | ![](newton/godmiddag_bowl_16_variant_pass.png) |

## What

`newton15_cert.py` mirrors the three NVIDIA runtime tests on the standalone Newton 1.5.2 wheel: parse
with Newton's USD importer, drop 1 cm onto a floor, drop onto a 15° walled slope, and grasp-and-lift on the
authored `grasp_identifier_01` with a two-pad gantry (close, lift by max(0.3 m, 2 × longest edge), hold 1 s,
shake 1 cm at 2 Hz for 1.5 s, hold, open). Verdict messages follow NVIDIA's wording so PhysX and Newton
results can be compared line by line. Every asset runs in its own subprocess; a crash is recorded, not fatal.
`newton15_cert_report.py` writes the comparison report. `make_newton_variant.py` authors the per-package
Newton collision layer described below.

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

## Deformables (examples)

SimReady has no runtime test for deformable packages. `examples/newton15_soft_grasp.py` (tetrahedral fruit)
and `examples/newton15_polybag_grasp.py` (cloth film with seal springs) run the same pad gantry on Newton's
VBD soft-body solver: settle, close by a set squeeze, lift, hold, shake, open. The SpaceAI apple, plum and
strawberry and the two polybags hold and deform plausibly.

| apple | plum | polybag with cotton |
|---|---|---|
| ![](newton/deformable_apple_strip.png) | ![](newton/deformable_plum_strip.png) | ![](newton/deformable_polybag_cotton_strip.png) |

## What it is not

Not an engine plugin for `simready-benchmark`; a standalone runner that mirrors the tests' procedure so
verdicts can be compared. Not a SimReady requirement: a package's profile pass stays a PhysX/Kit statement,
and the Newton verdicts are recorded beside it.
