# SPDX-License-Identifier: Apache-2.0
"""Let the asset go above the floor and see whether it falls, lands and settles.

For a rigid asset this is NVIDIA's own FET_003 ground drop. For a deformable one it is ours,
because NVIDIA's refuses an asset without `UsdPhysics.RigidBodyAPI`; the rules it is judged by are
in `native/drop_shape.py`, which imports no engine and takes no engine parameter.
"""
NAME = "drop"
SUMMARY = "drop the asset on the floor: does it fall, land, stay out of the floor and settle?"
KINDS = {"rigid", "deformable"}
SECONDS = 2.0
RIGID = ("simready_benchmark_kit_suite.fet003_physics.ground_drop", "ground_drop")
DEFORMABLE = {"newton": "newton_drop.py", "physx": "physx_drop.py"}
# PR #2's own criteria apply, and this experiment expects the asset to come to rest
PR2_CRITERIA = "rest"
