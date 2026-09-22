# SPDX-License-Identifier: Apache-2.0
"""Close two pads on the asset's own grasp annotation and lift it.

Rigid only, for the same reason as the slope: NVIDIA's test reads rigid-body transforms, and we
have not written a deformable equivalent.
"""
NAME = "grasp"
SUMMARY = "close the gripper on the asset's grasp annotation and lift: does it hold?"
KINDS = {"rigid"}
SECONDS = None
RIGID = ("simready_benchmark_kit_suite.fet005_grasp.grasp_and_lift", "grasp_and_lift")
DEFORMABLE = {}
# PR #2's criteria are about an asset falling; they say nothing here.
PR2_CRITERIA = None
