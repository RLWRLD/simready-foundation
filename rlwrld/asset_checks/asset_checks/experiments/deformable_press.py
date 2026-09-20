"""Press a deformable asset against the floor with a descending plate, then lift the plate away.

The drop test asks whether an asset falls and settles. This asks what it does when something pushes
into it: how far it compresses under a plate driven down by a set depth, and how much of that it
gets back when the plate leaves. A rigid body would simply resist or slide; a deformable should
flatten and then recover, and how much of each is the thing worth comparing across engines.

The plate is kinematic -- its height is prescribed, not simulated -- so the same motion is applied
whatever the engine, and the asset's response is the only variable. It descends to `press_to` of
the asset's height, holds, then rises.

Verdict:
  - the asset must compress: its height has to fall by at least `min_compression` of itself;
  - it must not be crushed through the floor: no point below it by more than `tunnel_depth`;
  - it must recover: after the plate leaves, its height must return to at least `min_recovery` of
    where it started;
  - and every point must stay finite.
"""
import numpy as np
from simready_benchmark.core.decorator import test

from asset_checks.experiments.deformable_drop import (
    deformable_state,
    lift_to,
    mesh_points,
    write_points_back,
)

PLATE = "/World/PressPlate"


def add_plate(stage, centre_xy, half_width, thickness, z):
    """A kinematic box above the asset. It is driven by its transform, so no engine simulates it and
    every engine sees the same motion."""
    from pxr import Gf, UsdGeom, UsdPhysics

    plate = UsdGeom.Cube.Define(stage, PLATE)
    plate.CreateSizeAttr(2.0)
    UsdGeom.XformCommonAPI(plate).SetScale(Gf.Vec3f(half_width, half_width, thickness / 2.0))
    UsdGeom.XformCommonAPI(plate).SetTranslate(Gf.Vec3d(centre_xy[0], centre_xy[1], z))
    plate.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.45, 0.85)])
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(plate.GetPrim())
    body.CreateKinematicEnabledAttr(True)
    return plate


def move_plate(stage, z):
    from pxr import Gf, Usd, UsdGeom

    api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(PLATE))
    translate = api.GetXformVectors(Usd.TimeCode.Default())[0]
    api.SetTranslate(Gf.Vec3d(translate[0], translate[1], float(z)))


@test(
    features=[{"id": "FET_003_STANDARD", "version": ">=0.1.0"}],
    name="deformable_press",
    description=(
        "Presses a deformable asset against the floor with a kinematic plate driven to a set depth, "
        "holds, then lifts the plate, and measures how much the asset compressed and how much of "
        "its height it recovered."
    ),
    expected_video=(
        "The asset rests on the floor. A blue plate descends onto it, flattens it, holds, then "
        "rises. The asset should spring back towards its original shape. One that does not deform "
        "at all, is pushed through the floor, or stays flat afterwards is the interesting case."
    ),
    version="0.1.0",
    engine={"tags": ["kit"], "version": ">=2024.2.0"},
    config_defaults={
        "simulation_seconds": 8.0,
        "physics_fps": 240,
        "capture_fps": 15,
        "settle_frames": 5,
        "asset_load_timeout": 30,
        "floor_level": 0.0,
        "settle_seconds": 1.0,      # let the asset rest on the floor before the plate moves
        "press_seconds": 2.0,       # how long the plate takes to reach its depth
        "hold_seconds": 1.0,        # how long it stays there
        "release_seconds": 1.5,     # how long it takes to lift away, and how long to watch after
        "press_to": 0.5,            # plate stops at this much of the asset's starting height
        "min_compression": 0.05,    # it has to lose at least this much of its height
        "min_recovery": 0.5,        # and get back at least this much of what it lost
        "tunnel_depth": 0.01,
        "import_grace_seconds": 1.0,
    },
    max_duration=300,
)
async def deformable_press(ctx):
    from asset_checks.kit.scene import ASSET_PRIM

    config = ctx.config
    physics_fps = int(config["physics_fps"])
    floor = float(config["floor_level"])
    capture_interval = max(1, physics_fps // int(config["capture_fps"]))
    grace_frames = int(float(config["import_grace_seconds"]) * physics_fps)
    settle_f = int(float(config["settle_seconds"]) * physics_fps)
    press_f = int(float(config["press_seconds"]) * physics_fps)
    hold_f = int(float(config["hold_seconds"]) * physics_fps)
    release_f = int(float(config["release_seconds"]) * physics_fps)
    total_frames = min(int(float(config["simulation_seconds"]) * physics_fps),
                       settle_f + press_f + hold_f + 2 * release_f)

    ctx.set_settle_frames(config["settle_frames"])
    ctx.scene.load_asset(ctx.asset_path, timeout=config["asset_load_timeout"])
    room = ctx.scene.add_room()
    room.auto_size(ctx.scene.asset)
    room.show_ground()
    physics = ctx.scene.add_physics(fps=physics_fps)
    _, height = lift_to(ctx, floor + 0.002)  # resting on the floor, not dropped

    authored = mesh_points(ctx.scene._stage, ASSET_PRIM)
    if authored is None:
        ctx.fail("the asset has no deformable geometry to press")
        ctx.add_metric("deformable_press_passed", 0)
        return
    centre = (float(authored[:, 0].mean()), float(authored[:, 1].mean()))
    top = float(authored[:, 2].max())
    plate_thickness = max(0.01, height * 0.2)
    add_plate(ctx.scene._stage, centre, max(0.05, height * 2.0), plate_thickness, top + plate_thickness)
    ctx.scene.setup_camera_follow()
    ctx.scene.update_camera_follow()
    await ctx.capture_frame(label="deformable_press")

    physics.stop()
    physics.play()

    start_top = plate_start = top + plate_thickness / 2.0
    plate_end = floor + float(config["press_to"]) * height + plate_thickness / 2.0
    lowest_top, deepest_below, recovered_top = None, 0.0, None
    previous, source = None, None
    for frame in range(total_frames):
        await ctx.physics_step()
        positions, _, source = deformable_state(ctx, previous, 1.0 / physics_fps)
        if positions is None and frame < grace_frames:
            continue
        if positions is None:
            ctx.fail("nothing deformable to measure: this engine did not build the asset as a deformable")
            ctx.add_metric("deformable_press_passed", 0)
            return
        if not np.isfinite(positions).all():
            ctx.fail(f"the asset's points stopped being finite at {frame / physics_fps:.2f}s: physics diverged")
            ctx.add_metric("deformable_press_passed", 0)
            return
        previous = positions
        if lowest_top is None:
            lowest_top = start_top = float(positions[:, 2].max())

        # Where the plate should be this frame: down, hold, up, then out of the way.
        if frame < settle_f:
            plate_z = plate_start
        elif frame < settle_f + press_f:
            plate_z = plate_start + (plate_end - plate_start) * (frame - settle_f) / press_f
        elif frame < settle_f + press_f + hold_f:
            plate_z = plate_end
        else:
            rising = min(1.0, (frame - settle_f - press_f - hold_f) / release_f)
            plate_z = plate_end + (plate_start - plate_end) * rising
        move_plate(ctx.scene._stage, plate_z)

        top_now = float(positions[:, 2].max())
        lowest_top = min(lowest_top, top_now)
        deepest_below = max(deepest_below, floor - float(positions[:, 2].min()))
        if frame >= settle_f + press_f + hold_f + release_f:
            recovered_top = top_now
        if frame % capture_interval == 0:
            if source == "newton particles":
                write_points_back(ctx.scene._stage, ASSET_PRIM, positions)
            ctx.scene.update_camera_follow()
            await ctx.capture_frame(label="deformable_press")
        await ctx.physics_advance()

    compressed = start_top - lowest_top
    recovery = 0.0 if recovered_top is None or compressed <= 0 else (recovered_top - lowest_top) / compressed
    ctx.add_metric("deformable_press_points", int(len(positions)))
    ctx.add_metric("deformable_press_start_top", round(start_top, 4))
    ctx.add_metric("deformable_press_lowest_top", round(lowest_top, 4))
    ctx.add_metric("deformable_press_compressed_m", round(compressed, 4))
    ctx.add_metric("deformable_press_compressed_frac", round(compressed / height if height else 0.0, 3))
    ctx.add_metric("deformable_press_recovered_frac", round(recovery, 3))
    ctx.add_metric("deformable_press_below_floor_m", round(deepest_below, 4))
    ctx.log(f"[deformable] pressed with a plate from {plate_start:.3f} to {plate_end:.3f}, measured from {source}")

    if compressed < float(config["min_compression"]) * height:
        failure = f"did not deform: its top moved {compressed * 1000:.1f} mm under the plate"
    elif deepest_below > float(config["tunnel_depth"]):
        failure = f"was pushed through the floor: {deepest_below * 1000:.1f} mm below it"
    elif recovery < float(config["min_recovery"]):
        failure = f"did not spring back: it recovered {recovery * 100:.0f}% of what it lost"
    else:
        failure = None
    ctx.add_metric("deformable_press_passed", 0 if failure else 1)
    if failure:
        ctx.fail(f"Deformable press FAILED: the asset {failure}.")
