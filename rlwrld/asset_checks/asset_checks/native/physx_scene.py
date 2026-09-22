# SPDX-License-Identifier: Apache-2.0
"""The PhysX world both deformable runners simulate in: the scene, its step rate, and the floor.

One owner, because the two runners had two copies and the copies drifted. What drifted was not a
constant anyone could see -- it was the *absence* of one. Neither copy set the scene's step rate,
so PhysX kept its own default of 60 steps per second while each runner called `app.update()` once
per substep. Isaac's documentation says how many physics steps one update is worth: "PhysX
determines how many physics steps to calculate based on the TimeStepsPerSecond and the variable
elapsed time since the last call", so four updates of a 60 Hz timeline at 60 steps per second are
four steps of 1/60 s -- four times the simulated time the frame is recorded as, at four times the
step Newton takes.

Measured, on the apple: 27.245 mm of fall in the first recorded frame, where four steps of 1/60 s
from rest give g*dt^2*n(n+1)/2 = 27.25 mm and the 1/60 s the frame claims would give 2.7 mm. The
videos played four times fast and the contacts were solved on a step four times coarser than
Newton's, which is most of why one engine held an apple on the floor and the other let it 5 mm
through. `drop_shape.free_fall` is the guard that now measures this every run.

So: the step rate is `fps * substeps` -- the same `1/(fps*substeps)` Newton integrates with -- and
a recorded frame is one `app.update()`, which is `1/fps` of timeline and `substeps` steps of
physics. The rigid tests have run at 240 Hz all along (`add_physics(fps=240)`), so 60 fps and four
substeps is not a new number, it is that one.
"""
GRAVITY = 9.81
# How many physics steps PhysX takes inside one recorded frame, the way `newton_drop.SUBSTEPS`
# says it for each Newton solver. 4 at 60 fps is 240 Hz, which is the rate the rigid tests have
# always run at (`add_physics(fps=240)` in `kit/nvidia_test.py`), so this is that number and not
# a new one. Both runners read it from here; it was written out twice before.
SUBSTEPS = 4
GROUND_DEPTH = 0.5     # how far the floor reaches below z = 0; it reaches recording.GROUND_HALF out


def world(stage, fps, substeps, path="/World/PhysicsScene", ground_path="/World/Ground"):
    """Define the scene and the floor. Both live on the stage at known paths; nothing is returned.

    Raises if PhysX does not take the step rate: a rate that was asked for and not applied is the
    bug this module exists to stop, and a silent default is how it hid the first time.
    """
    # USD is imported here rather than at the top so that a runner can read SUBSTEPS while
    # building its command line: importing pxr before Kit starts leaves Kit unable to register
    # its own schema wrappers, and the run dies during startup.
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    import recording   # the floor drawn in the video and the floor simulated are one floor

    scene = UsdPhysics.Scene.Define(stage, path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(GRAVITY)

    rate = float(fps) * int(substeps)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateTimeStepsPerSecondAttr(rate)
    read_back = physx.GetTimeStepsPerSecondAttr().Get()
    if read_back is None or abs(float(read_back) - rate) > 1e-6:
        raise SystemExit(f"[physx] asked for {rate:g} physics steps per second; the scene holds "
                         f"{read_back}. Without it a frame is {substeps} steps of PhysX's own "
                         f"default, not {substeps} steps of 1/{rate:g} s")
    print(f"[physx] scene: {rate:g} physics steps per second -- {substeps} steps of "
          f"{1.0 / rate * 1000:.3f} ms in each {1.0 / fps * 1000:.2f} ms frame, "
          f"the step Newton integrates with", flush=True)

    # A solid floor, not a sheet. Newton's `add_ground_plane` is a half-space -- infinitely thick
    # -- and a zero-thickness triangle mesh is a weaker collider by construction: measured, a
    # pressed banana was squeezed 11 mm through the sheet, which says nothing about PhysX and
    # everything about the floor it was given. The box's top face is at z = 0, so the two engines'
    # floors are in the same place.
    half = recording.GROUND_HALF
    ground = UsdGeom.Cube.Define(stage, ground_path)
    ground.CreateSizeAttr(2.0)
    UsdGeom.XformCommonAPI(ground).SetScale(Gf.Vec3f(half, half, GROUND_DEPTH / 2.0))
    UsdGeom.XformCommonAPI(ground).SetTranslate(Gf.Vec3d(0.0, 0.0, -GROUND_DEPTH / 2.0))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
