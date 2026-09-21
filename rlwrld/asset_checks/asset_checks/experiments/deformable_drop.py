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

    # The prims that declare themselves simulated, or, when none of those moved, whatever geometry
    # the asset has: PhysX updates a surface deformable's own mesh, but for a volume deformable it is
    # the render mesh that follows the simulation tetmesh, and that is what anyone watching sees.
    prims = list(Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()))
    declared = [prim for prim in prims
                if any(name.endswith(api) for name in raw_api_schemas(prim) for api in DEFORMABLE_SIM_API)]
    geometry = [prim for prim in prims if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.TetMesh)]
    out = []
    for prim in (declared or geometry):
        points = UsdGeom.PointBased(prim).GetPointsAttr().Get()
        if not points:
            continue
        to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        out.extend([to_world.Transform(p) for p in points])
    return np.asarray([[p[0], p[1], p[2]] for p in out]) if out else None


_SKIN = {}  # asset root -> (sim rest points, [(prim path, rest points, nearest sim index)])


def _skin_setup(stage, root_path, sim_prim_path, sim_rest):
    """Tie the asset's other geometry to the simulated points, once.

    A soft body is simulated as a tetrahedral mesh and drawn as a separate render mesh -- NVIDIA's
    and SpaceAI's assets both ship that pair -- so writing the solver's points into the simulation
    mesh leaves the visible one hanging in mid-air, which is what every banana video showed. Each
    render vertex is bound to its nearest simulation vertex at rest and then carries that vertex's
    displacement: not skinning weights, but enough that the picture is the simulation.
    """
    from pxr import Usd, UsdGeom

    bound = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        if str(prim.GetPath()) == sim_prim_path or not prim.IsA(UsdGeom.PointBased):
            continue
        attr = UsdGeom.PointBased(prim).GetPointsAttr()
        points = attr.Get() if attr else None
        if not points:
            continue
        to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rest = np.asarray([[*to_world.Transform(p)] for p in points], dtype=np.float64)
        nearest = np.empty(len(rest), dtype=np.int64)
        for start in range(0, len(rest), 512):  # chunked: the product of both point counts is large
            chunk = rest[start:start + 512]
            nearest[start:start + 512] = np.argmin(((chunk[:, None, :] - sim_rest[None, :, :]) ** 2).sum(-1), axis=1)
        bound.append((str(prim.GetPath()), rest, nearest))
    return bound


def _write_points(stage, prim_path, world_points):
    from pxr import Gf, Usd, UsdGeom, Vt

    prim = stage.GetPrimAtPath(prim_path)
    to_local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    local = [to_local.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in world_points]
    UsdGeom.PointBased(prim).GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in local]))


def write_points_back(stage, root_path, positions):
    """Put the solver's particles into the asset's geometry, so a capture shows the simulation.

    Isaac's Newton stage syncs rigid body transforms to Fabric and nothing else: a cloth or a soft
    body simulates, but the mesh a renderer draws stays where it was authored, and every video of a
    deformable comes out still. The particles are that geometry's points, in import order, so
    writing them back is what makes the picture true. Returns the prim written, or None.
    """
    from pxr import Gf, Usd, UsdGeom, Vt

    from asset_checks.kit.reading import raw_api_schemas
    from asset_checks.kit.scene import DEFORMABLE_SIM_API

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        points_attr = UsdGeom.PointBased(prim).GetPointsAttr() if prim.IsA(UsdGeom.PointBased) else None
        if points_attr is None or not points_attr.HasAuthoredValue():
            continue
        declared = any(name.endswith(api) for name in raw_api_schemas(prim) for api in DEFORMABLE_SIM_API)
        if not declared or len(points_attr.Get() or []) != len(positions):
            continue
        path = str(prim.GetPath())
        if root_path not in _SKIN:
            _SKIN[root_path] = (np.asarray(positions, dtype=np.float64),
                                _skin_setup(stage, root_path, path, np.asarray(positions, dtype=np.float64)))
        sim_rest, bound = _SKIN[root_path]
        _write_points(stage, path, positions)
        displacement = np.asarray(positions, dtype=np.float64) - sim_rest
        for other_path, rest, nearest in bound:
            _write_points(stage, other_path, rest + displacement[nearest])
        return path + (f" (+{len(bound)} carried along)" if bound else "")
    return None


def deformable_state(ctx, previous, dt):
    """(positions, speeds, where they came from) this step, whichever engine is running."""
    from simready_benchmark_engine_kit.physics_utils import active_physics_engine

    from asset_checks.kit.scene import ASSET_PRIM

    # Whether the engine built anything to simulate, and where that thing is, are two questions.
    # Newton's particle array answers the first: no particles means the importer did not read the
    # asset as a deformable, which no amount of geometry reading would reveal. The geometry answers
    # the second for every engine alike -- PhysX has no particle array, and a Newton solver's
    # particle frame is its own (VBD reports a cloth authored at 5 cm as starting at 75 cm) -- and it
    # is also what a renderer and a person see.
    if active_physics_engine() == "newton":
        positions, velocities = newton_particles()
        if positions is None:
            return None, None, "newton, no particles"
        source = "newton particles"
    else:
        positions, source = mesh_points(ctx.scene._stage, ASSET_PRIM), "deformable geometry"
        if positions is None:
            return None, None, "no deformable geometry"
        velocities = None if previous is None or previous.shape != positions.shape else (positions - previous) / dt
    speeds = np.abs(velocities).max() if velocities is not None and len(velocities) else 0.0
    return positions, float(speeds), source


def particle_radius_now():
    """The radius the engine gave the asset's particles, or 0 where there are none (PhysX)."""
    try:
        import isaacsim.physics.newton as isaac_newton
    except ImportError:
        return 0.0
    model = getattr(isaac_newton.acquire_stage(), "model", None)
    if model is None or not getattr(model, "particle_count", 0):
        return 0.0
    import numpy as np

    return float(np.asarray(model.particle_radius.numpy()).max())


def asset_height(ctx):
    """How tall the asset is, from its own geometry: every length in these tests is a fraction of it."""
    from pxr import Usd, UsdGeom

    from asset_checks.kit.scene import ASSET_PRIM

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeWorldBound(
        ctx.scene._stage.GetPrimAtPath(ASSET_PRIM)).ComputeAlignedRange()
    return float(box.GetMax()[2] - box.GetMin()[2])


def lift_to(ctx, target_z):
    """Put the asset's lowest point at target_z by moving its points, not its transform.

    NVIDIA's room helper places a rigid body by its bounding box and refuses an asset with no rigid
    body, so a deformable has to be placed here. It is placed by editing the geometry rather than by
    authoring a transform because the two do not agree: with a transform, Newton's particles came up
    6 cm from where the same prim sits in USD, and every measurement against the floor was wrong by
    that much. The points are what both Newton and PhysX read, so moving them leaves one frame.
    """
    from pxr import Gf, Usd, UsdGeom, Vt

    from asset_checks.kit.scene import ASSET_PRIM

    prim = ctx.scene._stage.GetPrimAtPath(ASSET_PRIM)
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeWorldBound(prim)
    span = box.ComputeAlignedRange()
    shift = float(target_z - span.GetMin()[2])
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.PointBased):
            continue
        attr = UsdGeom.PointBased(child).GetPointsAttr()
        points = attr.Get() if attr else None
        if not points:
            continue
        to_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        to_local = to_world.GetInverse()
        moved = [to_local.Transform(to_world.Transform(point) + Gf.Vec3d(0.0, 0.0, shift)) for point in points]
        attr.Set(Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in moved]))
    return shift, float(span.GetMax()[2] - span.GetMin()[2])


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
        # Lengths are fractions of the asset's own height, so the same test suits a 2 cm bead and a
        # 2 m sheet. Each has an absolute floor for a degenerate asset -- a flat cloth has no height.
        "drop_height_of_height": 1.0,   # the asset starts this much of its own height above the floor
        "drop_height_min": 0.02,
        "floor_margin_of_height": 0.5,  # how close its lowest point must come to count as landed
        "floor_margin_min": 0.005,
        "tunnel_depth_of_height": 0.2,  # how far below the floor it may sit
        "tunnel_depth_min": 0.005,
        "min_fall_of_height": 0.2,      # how far it must drop to count as having fallen
        "min_fall_min": 0.005,
        "rest_speed": 0.05,        # m/s, under which the asset counts as still
        "rest_hold_seconds": 0.5,
        "import_grace_seconds": 1.0,   # how long an engine may take to build its model before this gives up
    },
    max_duration=300,
)
async def deformable_drop(ctx):
    config = ctx.config
    physics_fps = int(config["physics_fps"])
    floor = float(config["floor_level"])
    capture_interval = max(1, physics_fps // int(config["capture_fps"]))
    total_frames = int(float(config["simulation_seconds"]) * physics_fps)
    rest_frames = max(1, int(float(config["rest_hold_seconds"]) * physics_fps))
    grace_frames = int(float(config["import_grace_seconds"]) * physics_fps)

    ctx.set_settle_frames(config["settle_frames"])
    ctx.scene.load_asset(ctx.asset_path, timeout=config["asset_load_timeout"])
    room = ctx.scene.add_room()
    room.auto_size(ctx.scene.asset)
    room.show_ground()
    ctx.scene.setup_camera_follow()  # where the runner places its camera and floor cues

    # Placed before physics is set up: the engine reads the geometry when it builds its model, and
    # anything moved after that is moved only in USD.
    height = asset_height(ctx)
    scaled = {name: max(float(config[f"{name}_min"]), float(config[f"{name}_of_height"]) * height)
              for name in ("drop_height", "floor_margin", "tunnel_depth", "min_fall")}
    lifted, height = lift_to(ctx, floor + scaled["drop_height"])
    margin = scaled["floor_margin"]
    ctx.log(f"[deformable] the asset is {height * 100:.1f} cm tall, so it drops from "
            f"{scaled['drop_height'] * 100:.1f} cm and has to land within {margin * 1000:.1f} mm")
    physics = ctx.scene.add_physics(fps=physics_fps)

    # Where it starts, and a picture of it, both taken before the timeline runs: a capture costs app
    # updates, and with the timeline playing those step physics, so anything read after the first
    # capture is already centimetres into the fall. The asset's own geometry is the start under every
    # engine, and a run that dies on its first step still has a frame of what it was given.
    from asset_checks.kit.scene import ASSET_PRIM

    authored = mesh_points(ctx.scene._stage, ASSET_PRIM)
    start_z = None if authored is None else float(authored[:, 2].min())
    ctx.scene.update_camera_follow()
    await ctx.capture_frame(label="deformable_drop")

    physics.stop()
    physics.play()

    rest_run = 0
    lowest_seen, deepest_below, fastest = None, 0.0, 0.0
    settled_at, frames, previous, source = None, 0, None, None
    for frame in range(total_frames):
        await ctx.physics_step()
        positions, speed, source = deformable_state(ctx, previous, 1.0 / physics_fps)
        if positions is None and frame < grace_frames:
            continue  # the engine may not have built its model yet; see import_grace_seconds
        if positions is None:
            ctx.fail("nothing deformable to measure: " + (
                "this Newton built no particles from the asset, so its importer did not read it as a "
                "deformable (Newton 1.2.1 has no USD deformable import; 1.5 reads the proposal's "
                "Physics*DeformableSimAPI names)" if source == "newton, no particles" else
                "the asset has no deformable geometry this engine could find"))
            ctx.add_metric("deformable_drop_passed", 0)
            return
        if not np.isfinite(positions).all():
            ctx.fail(f"particle positions stopped being finite at {frame / physics_fps:.2f}s: physics diverged")
            ctx.add_metric("deformable_drop_passed", 0)
            return
        frames = frame + 1
        low = float(positions[:, 2].min())
        if lowest_seen is None:
            # A solver reports its particles in its own frame: Newton VBD puts a cloth authored at
            # 5 cm at 75 cm. One step of falling apart, the first reading is the start, so the offset
            # between the two frames is constant and the floor moves with it. Measured, not assumed:
            # under XPBD it comes out at zero and nothing shifts.
            # Compare the two frames at the same instant: the geometry as it stands on this step,
            # not as it stood before play, because the engine may have moved the asset when the
            # timeline started. Under XPBD this comes out at zero.
            here = mesh_points(ctx.scene._stage, ASSET_PRIM)
            offset = 0.0 if here is None else low - float(here[:, 2].min())
            floor += offset
            start_z, lowest_seen = low, low
            # What is measured is particle centres, and a particle at rest sits one radius above
            # the surface, so "reached the floor" has to allow for that however big the particles are.
            radius = particle_radius_now()
            margin += radius
            ctx.log(f"[deformable] first step: {source} lowest {low:.4f}, the same geometry in USD "
                    f"{'?' if here is None else round(float(here[:, 2].min()), 4)}, so the floor is "
                    f"{floor:.4f} here; particles are {radius * 1000:.2f} mm, so landing means within "
                    f"{margin * 1000:.1f} mm")
        lowest_seen = min(lowest_seen, low)
        deepest_below = max(deepest_below, floor - low)
        previous = positions
        fastest = max(fastest, speed)
        rest_run = rest_run + 1 if speed < float(config["rest_speed"]) else 0
        if settled_at is None and rest_run >= rest_frames:
            settled_at = (frame - rest_frames + 1) / physics_fps
        if frame % capture_interval == 0:
            if source == "newton particles":
                written = write_points_back(ctx.scene._stage, ASSET_PRIM, positions)
                if frame == 0:
                    ctx.log(f"[deformable] writing the solver's points into {written} for the capture")
            ctx.scene.update_camera_follow()
            await ctx.capture_frame(label="deformable_drop")
        await ctx.physics_advance()
        if settled_at is not None and frame > rest_frames * 2:
            break

    fell = start_z - lowest_seen
    ctx.add_metric("deformable_drop_points", int(len(positions)))
    ctx.add_metric("deformable_drop_start_z", round(start_z, 4))
    ctx.add_metric("deformable_drop_lifted_by", round(lifted, 4))
    ctx.add_metric("deformable_drop_floor_used", round(floor, 4))
    ctx.log(f"[deformable] measured from {source}")
    ctx.add_metric("deformable_drop_fall_m", round(fell, 4))
    ctx.add_metric("deformable_drop_lowest_z", round(lowest_seen, 4))
    ctx.add_metric("deformable_drop_below_floor_m", round(deepest_below, 4))
    ctx.add_metric("deformable_drop_max_speed", round(fastest, 3))
    ctx.add_metric("deformable_drop_settled_s", -1.0 if settled_at is None else round(settled_at, 3))
    ctx.add_metric("deformable_drop_seconds", round(frames / physics_fps, 3))

    if fell < scaled["min_fall"]:
        failure = f"never fell: its lowest point moved {fell * 1000:.1f} mm"
    elif deepest_below > scaled["tunnel_depth"]:
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
