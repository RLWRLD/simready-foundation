# SPDX-License-Identifier: Apache-2.0
"""How finely a recorded frame is resolved. One number, for every engine and every solver.

A recorded frame is 1/fps of simulated time -- `drop_shape.free_fall` is the guard that holds
every runner to that -- and this is how many physics steps are taken inside it. It is here, on its
own, because it was three numbers before: 10 for Newton's VBD, 32 for its XPBD, 4 for PhysX, each
read off that solver's own examples. Three numbers is three experiments. An asset that failed on
one engine and passed on another was then telling you about the numbers as much as the engines.

**The rule is the finest any of them asks for**, not the one that makes an asset pass. Newton's
`example_rigid_soft_contact.py` takes 32 substeps and its softbody examples take 10; PhysX ships
no count for a deformable drop at all. 32 is the largest of what the examples ask, and it is a
rule that keeps working on a solver nobody has added yet: whatever the finest is, everything runs
at it.

Measured, on the apple pressed 20% of its height (the numbers that made this one constant):

    PhysX   240 Hz (4)    below floor 4.0 mm   compressed 10.8 mm   pushed-through-floor
    PhysX   480 Hz (8)                2.6 mm              11.8 mm   pass
    PhysX   960 Hz (16)               1.4 mm              13.3 mm   pass
    PhysX  1920 Hz (32)               1.4 mm              13.7 mm   pass
    Newton 1.5 VBD, 600 Hz (10)       0.0 mm              15.2 mm   pass

PhysX walks towards Newton's answer as the step shrinks, so `pushed-through-floor` at 240 Hz was a
statement about 240 Hz. Cost is not a reason to keep them apart either: that cell took 39 s at
1920 Hz and 32 s at 240 Hz -- a run is bound by what surrounds the step, not by the step.

This is about the *step*, and nothing else. How each solver is set up is still its own -- VBD's
graph colouring, XPBD's compliance, PhysX's collision offsets all come from that engine's own
documentation and examples, and are not touched by this.
"""
SUBSTEPS = 32
# The recorded frame rate, for every runner: a frame is 1/FPS of simulated time and one video
# frame. It was the default of four separate `--fps` arguments.
FPS = 60.0
