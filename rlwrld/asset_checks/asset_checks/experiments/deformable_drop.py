"""Drop a deformable asset on the floor and see whether it falls, lands and settles.

NVIDIA's Foundation 7.1 tests are rigid-only by construction: ground_drop, slope_drop and
grasp_and_lift all begin by refusing an asset without `UsdPhysics.RigidBodyAPI`, and read the
asset's motion from rigid-body transforms, of which a deformable has none. A cloth or a soft body
is particles, so this reads the particle state Newton owns instead.

It is registered with NVIDIA's own `@test` decorator and takes NVIDIA's RunContext, so the runner
treats it exactly like theirs -- `--experiments deformable_drop` -- and it is also the worked
example for adding an experiment (see the README).

Verdict:
  - the asset must fall: its lowest particle has to drop by at least `min_fall`;
  - it must reach the floor: the lowest particle has to come within `floor_margin` of it;
  - it must not go through: no particle may stay below the floor by more than `tunnel_depth` ;
  - it must settle: the fastest particle has to stay under `rest_speed` for `rest_hold_seconds`;
  - and every particle must stay finite -- a diverged solver is a failure, not a verdict.
"""
import numpy as np
from simready_benchmark.core.decorator import test


def newton_particles():
    """(positions, velocities) of Newton's particles this step, or (None, None) without Newton."""
    try:
        import isaacsim.physics.newton as isaac_newton
    except ImportError:
        return None, None
    state = getattr(isaac_newton.acquire_stage(), "state_0", None)
    if state is None or getattr(state, "particle_q", None) is None:
        return None, None
    positions = np.asarray(state.particle_q.numpy())
    velocities = np.asarray(state.particle_qd.numpy()) if getattr(state, "particle_qd", None) is not None else None
    return (positions if len(positions) else None), velocities


def mesh_points(stage, root_path):
    """Every point of the deformable geometry under root_path, in world space, from the stage.

    Newton owns its particles and the state above is the direct reading. PhysX has no such array
    here, and writes the deformed geometry back to the mesh instead, so for any other engine the
    points are the measurement. Velocity is not read back this way; the caller differences positions.
    """
    from pxr import Usd, UsdGeom

    from asset_checks.kit.reading import raw_api_schemas
    from asset_checks.kit.scene import DEFORMABLE_SIM_API

    deformable = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies())
                  if any(name.endswith(api) for name in raw_api_schemas(prim) for api in DEFORMABLE_SIM_API)]
    out = []
    for prim in deformable:
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        out.extend([to_world.Transform(p) for p in points])
    return np.asarray([[p[0], p[1], p[2]] for p in out]) if out else None


def deformable_state(ctx, previous, dt):
    """(positions, speeds, where they came from) this step, whichever engine is running."""
    positions, velocities = newton_particles()
    source = "newton particles"
    if positions is None:
        from asset_checks.kit.scene import ASSET_PRIM

        positions = mesh_points(ctx.scene._stage, ASSET_PRIM)
        source = "mesh points"
        velocities = None if positions is None or previous is None or previous.shape != positions.shape \
            else (positions - previous) / dt
    if positions is None:
        return None, None, source
    speeds = np.abs(velocities).max() if velocities is not None and len(velocities) else 0.0
    return positions, float(speeds), source


@test(
    features=[{"id": "FET_003_STANDARD", "version": ">=0.1.0"}],
    name="deformable_drop",
    description=(
        "Drops a deformable asset (cloth, soft body or cable) onto a flat floor and checks that it "
        "falls, reaches the floor, does not pass through it, and settles. Reads the particle state "
        "of the solver rather than rigid-body transforms, which a deformable does not have."
    ),
    expected_video=(
        "The asset starts above a flat floor, falls under gravity, lands and comes to rest. Cloth "
        "should drape; a soft body should deform on impact and stop. An asset that falls through "
        "the floor, keeps sliding, or never moves at all indicates a broken deformable setup."
    ),
    version="0.1.0",
    engine={"tags": ["kit"], "version": ">=2024.2.0"},
    config_defaults={
        "simulation_seconds": 6.0,
        "physics_fps": 240,
        "capture_fps": 15,
        "settle_frames": 5,
        "asset_load_timeout": 30,
        "floor_level": 0.0,
        "floor_margin": 0.02,      # how close the lowest particle must come to the floor
        "tunnel_depth": 0.01,      # how far below the floor a particle may sit
        "min_fall": 0.01,          # how far the lowest particle must drop to count as falling
        "rest_speed": 0.05,        # m/s, under which the asset counts as still
        "rest_hold_seconds": 0.5,
    },
    max_duration=300,
)
async def deformable_drop(ctx):
    config = ctx.config
    physics_fps = int(config["physics_fps"])
    floor, margin = float(config["floor_level"]), float(config["floor_margin"])
    capture_interval = max(1, physics_fps // int(config["capture_fps"]))
    total_frames = int(float(config["simulation_seconds"]) * physics_fps)
    rest_frames = max(1, int(float(config["rest_hold_seconds"]) * physics_fps))

    ctx.set_settle_frames(config["settle_frames"])
    ctx.scene.load_asset(ctx.asset_path, timeout=config["asset_load_timeout"])
    room = ctx.scene.add_room()
    room.auto_size(ctx.scene.asset)
    room.show_ground()
    ctx.scene.setup_camera_follow()  # where the runner places its camera and floor cues

    physics = ctx.scene.add_physics(fps=physics_fps)
    physics.stop()
    physics.play()

    start_z, rest_run = None, 0
    lowest_seen, deepest_below, fastest = None, 0.0, 0.0
    settled_at, frames, previous, source = None, 0, None, None
    for frame in range(total_frames):
        await ctx.physics_step()
        positions, speed, source = deformable_state(ctx, previous, 1.0 / physics_fps)
        if positions is None:
            ctx.fail("nothing deformable to measure: the asset did not import as a deformable in "
                     "this engine, or this solver does not simulate it")
            ctx.add_metric("deformable_drop_passed", 0)
            return
        if not np.isfinite(positions).all():
            ctx.fail(f"particle positions stopped being finite at {frame / physics_fps:.2f}s: physics diverged")
            ctx.add_metric("deformable_drop_passed", 0)
            return
        frames = frame + 1
        low = float(positions[:, 2].min())
        if lowest_seen is None:
            start_z = lowest_seen = low
        lowest_seen = min(lowest_seen, low)
        deepest_below = max(deepest_below, floor - low)
        previous = positions
        fastest = max(fastest, speed)
        rest_run = rest_run + 1 if speed < float(config["rest_speed"]) else 0
        if settled_at is None and rest_run >= rest_frames:
            settled_at = (frame - rest_frames + 1) / physics_fps
        if frame % capture_interval == 0:
            ctx.scene.update_camera_follow()
            await ctx.capture_frame(label="deformable_drop")
        await ctx.physics_advance()
        if settled_at is not None and frame > rest_frames * 2:
            break

    fell = start_z - lowest_seen
    ctx.add_metric("deformable_drop_points", int(len(positions)))
    ctx.log(f"[deformable] measured from {source}")
    ctx.add_metric("deformable_drop_fall_m", round(fell, 4))
    ctx.add_metric("deformable_drop_lowest_z", round(lowest_seen, 4))
    ctx.add_metric("deformable_drop_below_floor_m", round(deepest_below, 4))
    ctx.add_metric("deformable_drop_max_speed", round(fastest, 3))
    ctx.add_metric("deformable_drop_settled_s", -1.0 if settled_at is None else round(settled_at, 3))
    ctx.add_metric("deformable_drop_seconds", round(frames / physics_fps, 3))

    if fell < float(config["min_fall"]):
        failure = f"never fell: its lowest point moved {fell * 1000:.1f} mm"
    elif deepest_below > float(config["tunnel_depth"]):
        failure = f"went through the floor: {deepest_below * 1000:.1f} mm below it"
    elif lowest_seen > floor + margin:
        failure = f"never reached the floor: stopped {(lowest_seen - floor) * 1000:.1f} mm above it"
    elif settled_at is None:
        failure = f"never settled: still moving at {fastest:.2f} m/s after {frames / physics_fps:.1f}s"
    else:
        failure = None
    ctx.add_metric("deformable_drop_passed", 0 if failure else 1)
    if failure:
        ctx.fail(f"Deformable drop FAILED: the asset {failure}.")
