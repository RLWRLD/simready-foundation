# Free-space variant of the FET005 grasp-and-lift benchmark

RLWRLD change to `simready_benchmark_kit_suite/fet005_grasp` (2026-09-15). Config flag
`free_space` (default `False`, the standard test is untouched) and `free_space_height`
(0.5 m). Every result of the mode is stamped `variant: free-space` in its metrics and messages,
so it can never be read as the standard floor-standing verdict.

## Why

DexBench keeps 171 handled props, every one with an authored SimReady grasp line, and runs
NVIDIA's grasp-and-lift on all of them. 27 failed the floor-standing test. For 22 of them the
failure was the rig, not the asset: the two pads cannot sit lower than half their edge above the
floor, so on a 1.5-3 mm sheet (cutlery, sheet-metal parts) they close above the metal
("Grasping failed: pads touched (no object)"), or they drag over a flat object instead of pinching
it ("Lifting failed"). Such a verdict says nothing about the mass, the friction, the collider or
the line. We needed a test that closes the pads on the authored line without the floor in the way
and then runs the identical lift, hold, shake and release.

| standard, `fork_big` line 01: the pads rise, the fork stays | free-space, `fork_big` vertical line 02: pinched and carried through the shake |
|---|---|
| ![](free_space/fork_big_standard_fail.png) | ![](free_space/fork_big_freespace_pass.png) |

## What the mode does

* No settle drop: the asset is placed with its bbox bottom `free_space_height` above the floor at
  its authored orientation; the placement is verified and repeated until the bottom really is there
  (one frame of physics can move it).
* Physics starts at zero gravity with the asset's `sleepThreshold` at 0.
* The pads are built at the two line endpoints and closed exactly as in the standard test
  (ramp + `close_settle`).
* When Grasping completes, gravity returns to 9.81 m/s² live (`physics:gravityMagnitude`), and the
  object's shift, rotation and pad gap during the zero-g closure are logged
  (`grasp_<id>_free_space_closure_shift_m`, `_rotation_deg`, `_pad_gap_after_close_mm`).
* Lift, hold, shake, hold, open and drop run unchanged. After Opening the pad cubes stop colliding:
  on a vertical line the retracted lower pad is otherwise a shelf under the object and the drop
  check reports a 2 cm "fall" for a grasp that held through everything.
* The Grasping-phase floor gate (`is_grasp_line_reachable`) is skipped so vertical lines are allowed
  (it is dead code in the standard flow: it sits under `if phases.tracking_active`, which is False
  once the phase is Grasping).
* Two multi-identifier fixes that the standard flow also benefits from: the second identifier no
  longer reads a stale asset pose, and the previous gripper is removed before the asset is placed.

| standard, `dragon_spoon_19cm` vertical line: dropped | free-space, same line: held |
|---|---|
| ![](free_space/dragon_spoon_standard_fail.png) | ![](free_space/dragon_spoon_freespace_pass.png) |

| standard, `plate_small` centre line: gripper positioning timeout | free-space: pinched through the centre and lifted |
|---|---|
| ![](free_space/plate_small_standard_fail.png) | ![](free_space/plate_small_freespace_pass.png) |

## What it showed (2026-09-15, Isaac Sim 6.0.1, kit-suite 2026.6.1 + this change)

29 props: the 27 floor failures and two passing controls. **24 pass in free space**: both controls,
12 of the 27 with their existing horizontal line (the floor alone had defeated them), 10 more only
with a vertical pinch line added for the experiment. **5 still fail**, and each failure now names the
asset: a 3 mm flat bar (below the 4 mm pad-gap rule of the rig), a split shaft collar whose SDF halves
had no convex pieces, a strawberry proxy that slips, a serving bowl with no physics material, a power
drill that crashes the engine session. Standard mode on the same copies: 0 of the 22 pass.

| free-space passes with the existing line | |
|---|---|
| `dragon_fork_19cm` ![](free_space/dragon_fork_freespace_pass.png) | `hex_nut_m20` ![](free_space/hex_nut_m20_freespace_pass.png) |
| `prop_bowl_16` (a through-centre line at 50 % height) ![](free_space/bowl_16_freespace_pass.png) | control `005_tomato_soup_can` ![](free_space/soup_can_control_freespace_pass.png) |

| free-space passes only with a vertical line | free-space failures that name the asset |
|---|---|
| `ntn_6206` bearing, pinch of the solid annulus ![](free_space/bearing_6206_freespace_pass.png) | `sus304_flat_bar` 3 mm sheet ![](free_space/flat_bar_freespace_fail.png) |
| | strawberry proxy slipping ![](free_space/strawberry_fail_strip.png) |
| | split shaft collar dropping in the shake ![](free_space/collar_fail_strip.png) |

The full per-prop, per-identifier table, the 43 videos and the remote recipe are in the DexBench-Arena
repository under `docs/assets/grasp_benchmark_free_space/`.

## How to use it

Apply this branch's kit suite (editable install of
`nv_core/testing_tools/simready-benchmark-kit-suite/packages/simready_benchmark_kit_suite`) and set
`free_space: true` in the grasp_and_lift test config (`free_space_height` defaults to 0.5 m). With
the flag off the suite behaves exactly as upstream 2026.06.0.

## What it is not

It does not replace the floor-standing test and it is not a SimReady requirement. A pass in free
space says the line, the mass, the friction and the collider hold a pinch, lift and shake; whether a
real gripper can reach that line on a shelf is the floor test's question. Results are recorded
separately (`variant: free-space`).
