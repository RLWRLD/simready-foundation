# SPDX-License-Identifier: Apache-2.0
"""Press a flat plate into the settled asset and take it away again.

Deformable only: pressing a rigid body measures the contact solver, not the asset. The schedule,
the depth, the plate's geometry and the pass criteria are in `native/press_shape.py`, which
imports no engine -- the experiment states the condition and each engine arranges to represent it.
"""
NAME = "press"
SUMMARY = "press a plate into the asset: how far does it give, and does it come back?"
KINDS = {"deformable"}
SECONDS = 4.0
RIGID = None
DEFORMABLE = {"newton": "newton_press.py", "physx": "physx_press.py"}
# Pressing measures how much a body gives and how much of that it gets back. A surface has no
# thickness to give, so for a cloth the experiment has no answer, and the pipeline refuses it
# rather than reporting a verdict about nothing.
BODIES = {"volume"}
# PR #2's criteria are about an asset falling; they say nothing here.
PR2_CRITERIA = None
