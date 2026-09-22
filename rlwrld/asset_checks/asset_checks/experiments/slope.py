# SPDX-License-Identifier: Apache-2.0
"""Set the asset on an incline and see whether it slides without falling through.

Rigid only: the deformable path has no slope runner, and declaring the kind without a driver is
refused when this package loads rather than after a Kit launch.
"""
NAME = "slope"
SUMMARY = "drop the asset on a 45-degree slope: does it slide, and stay out of the ground?"
KINDS = {"rigid"}
SECONDS = None
RIGID = ("simready_benchmark_kit_suite.fet003_physics.slope_drop", "slope_drop")
DEFORMABLE = {}
