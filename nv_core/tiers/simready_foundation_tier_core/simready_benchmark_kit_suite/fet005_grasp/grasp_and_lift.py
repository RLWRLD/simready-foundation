# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FET005 Grasp-and-Lift test.

Validates whether a SimReady asset can be grasped, lifted, held, shaken,
and released by a standardized parallel-jaw gripper at each declared
grasp point (grasp_identifier_* prims).

Skips if no grasp identifiers are found on the asset.
Stops at the first failed phase per identifier.
Fresh scene per identifier for PhysX safety.
"""
import os

from simready_benchmark.core.decorator import test

_RUNTIME_PHYSICS_FEATURES = {
    "newton": "FET_003_NEWTON",
    "mujoco": "FET_003_MUJOCO",
    "physx": "FET_003_PHYSX",
}


def _required_physics_feature(runtime):
    # type: (str) -> str | None
    """Return the exact FET003 dependency for a recognized runtime."""
    return _RUNTIME_PHYSICS_FEATURES.get(runtime.strip().lower())


@test(
    features=[
        {"id": "FET_005_STANDARD", "version": ">=0.1.0"},
    ],
    name="grasp_and_lift",
    description=(
        "Iterates each grasp_identifier_* prim authored on the asset. At "
        "each, positions a synthetic gantry parallel-jaw gripper at the "
        "identifier's pose, closes it onto the asset, lifts the gripper "
        "by a configurable height, holds, optionally shakes. Verifies the "
        "asset stays grasped — doesn't slip out of the jaws or separate "
        "vertically by more than fall_min_delta_z. Validates that each "
        "declared grasp is physically achievable in the active physics runtime. Depends on "
        "the active runtime's FET_003 physics feature: the asset is skipped, "
        "not failed, when its physics setup is not validated, since grasp behavior is only "
        "meaningful once that setup is confirmed."
    ),
    expected_video=(
        "For each grasp_identifier on the asset: a parallel-jaw gripper "
        "moves to the identifier's pose, closes around the asset, lifts "
        "it cleanly off the floor, holds it in mid-air, optionally shakes "
        "side-to-side, then opens to drop. The asset should follow the "
        "gripper's lift motion without sliding out. An asset that slips "
        "during lift or shake fails that grasp."
    ),
    version="1.3.0",
    engine={"tags": ["kit"], "version": ">=2024.2.0"},
    config_defaults={
        "physics_fps": 240,
        "capture_fps": 15,
        "simulation_seconds": 15.0,
        # Preserve the terminal verdict immediately, then keep stepping and
        # recording so the video shows why the object was lost.
        "post_failure_observation_seconds": 1.0,
        "close_duration": 0.5,
        "close_ramp_speed_m_s": 0.04,
        # After the close ramp ends, keep holding the close command for
        # this long so PD-driven gripper joints can physically converge
        # onto the object before Lifting starts.  Without this the
        # gripper is still moving when lift commands rise, and small /
        # tightly-fitted objects slip out.
        "close_settle_seconds": 0.3,
        # Do not begin Lifting while the fingers are still moving or rebounding
        # from first contact. Require a bounded stable-position window after
        # the minimum settle time.
        "close_convergence_hold_seconds": 0.1,
        # Preserve the squeeze that produced a stable bilateral grasp and
        # observe it for another 0.3 seconds before beginning the lift.
        "post_grasp_stability_seconds": 0.3,
        "close_convergence_tolerance": 0.0005,
        "close_object_settle_tolerance": 0.0005,
        "close_contact_travel_tolerance": 0.002,
        "grasp_contact_compression": 0.005,
        "close_max_settle_seconds": 2.0,
        "lift_duration": 1.0,
        "open_duration": 0.5,
        "lift_height_multiplier": 2.0,
        "lift_min_delta_z": 0.02,
        "lift_min_height": 0.3,
        "lift_max_height": 0.5,
        # Minimum release fall used by Dropping. The required distance is the
        # larger of this value and the grasped body's bounding-box height.
        "fall_min_delta_z": 0.05,
        # Hold-phase drop tolerance (separate from fall_min_delta_z so
        # phase_dropping's "did it actually fall when released?" check
        # stays strict). Long thin objects -- dishwand-style tools,
        # handles -- can shift 5-10cm in centre-of-mass z while firmly
        # gripped near one end: the bbox tip rotates in the gripper
        # which drags the centre down without the object falling out.
        # 10cm catches real in-grip drops while tolerating swing.
        "hold_drop_tolerance": 0.10,
        "hold_seconds": 1.0,
        "shake_duration_seconds": 1.5,
        "shake_amplitude": 0.01,
        "shake_frequency_hz": 2.0,
        # Ease the gantry into and out of the circular orbit. Starting the
        # cosine component at full amplitude creates a one-frame position
        # discontinuity and an artificial contact impulse.
        "shake_ramp_seconds": 0.25,
        "shake_grip_separation_tolerance": 0.05,
        "enable_shake_test": True,
        "stability_max_seconds": 3.0,
        "rest_tolerance": 0.002,
        "rest_detection_hold_seconds": 1.0,
        "pad_touching_tolerance": 0.002,
        "floor_level": 0.0,
        # Raise the generated gripper when necessary so the complete oriented
        # pad boxes remain above the room floor.
        "pad_ground_clearance": 0.002,
        # Floor-contact tolerance for shake/hold phases. NEGATIVE values
        # allow the object's bbox-bottom to dip below the floor by this
        # much during a shake without failing the test. Slim assets
        # swinging in the gripper can pass 1-2cm below ground while the
        # object remains firmly held (visible in failing-asset videos:
        # the gripper retains the asset but reports a 6mm bbox dip).
        # -2cm leaves headroom for normal in-grip motion while still
        # catching genuine ground contact when the object truly falls.
        "floor_margin": -0.02,
        "release_check_seconds": 0.5,
        "drop_check_seconds": 3.0,
        "post_drop_observation_seconds": 1.0,
        "asset_load_timeout": 30,
        "settle_frames": 5,
        "show_gripper": True,
        # DexBench free-space variant (2026-09-15). False = the standard
        # floor-standing test, unchanged. True = no settle drop: the asset is
        # floated with its bbox bottom `free_space_height` metres above the
        # floor at its authored orientation, physics starts at zero gravity,
        # the pads are built at the line endpoints and closed as usual
        # (ramp + close_settle), then gravity is restored to 9.81 m/s^2 and
        # the lift / hold / shake / hold / open / drop phases run unchanged.
        # The Grasping-phase floor gate (both endpoints above floor_level)
        # is skipped so vertical closing lines are allowed. Every result of
        # this mode carries `variant: free-space` in its metrics and messages.
        "free_space": False,
        "free_space_height": 0.5,
    },
    max_duration=600,
)
async def test_grasp_and_lift(ctx):
    """Test grasp-and-lift for each grasp_identifier on the asset."""
    # FET005 depends on the active runtime's FET003 physics contract. Running
    # it against an asset that has not validated that contract would exercise
    # physics that is not guaranteed and report a misleading failure, so skip
    # FET005 instead of testing it. asset_validated_features is None for external or
    # forced (--features) runs that carry no workspace validation record; in
    # that case the dependency cannot be checked, so we proceed and run.
    runtime = os.environ.get("SIMREADY_PHYSICS_RUNTIME", "PhysX")
    required_feature = _required_physics_feature(runtime)
    if required_feature is None:
        ctx.skip(
            "Unsupported SIMREADY_PHYSICS_RUNTIME value %r. Expected one of: "
            "%s. FET005 was not run because its runtime-specific FET003 "
            "dependency cannot be determined." % (runtime, ", ".join(sorted(_RUNTIME_PHYSICS_FEATURES)))
        )
        return

    validated = ctx.asset_validated_features
    if validated is not None and required_feature not in validated:
        ctx.skip(
            "Requires %s to be validated first: grasp-and-lift depends on the "
            "active runtime physics setup, and %s is not in this asset's validated "
            "features." % (required_feature, required_feature)
        )
        return

    # Lazy imports: Kit/USD modules are not available in pure-Python test env
    from simready_benchmark_kit_suite.fet005_grasp.grasp_checks import run_pre_checks

    cfg = ctx.config

    # Free-space variant: stamp the result so it can never be read as the
    # standard floor-standing test (metric + step now, message prefix below).
    free_space = bool(cfg.get("free_space", False))
    variant_tag = "[variant: free-space] " if free_space else ""
    if free_space:
        ctx.add_metric("variant", "free-space")
        ctx.add_metric("free_space_height_m", float(cfg.get("free_space_height", 0.5)))
        ctx.step(
            "variant: free-space -- NOT the standard FET005 floor-standing test: asset floated "
            "%.2f m above the floor, jaws closed at zero gravity, gravity restored before lift"
            % float(cfg.get("free_space_height", 0.5))
        )

    # --- Load asset + room (before pre-checks, same as FET003/FET004) ---
    ctx.set_settle_frames(cfg["settle_frames"])
    ctx.scene.load_asset(ctx.asset_path, timeout=cfg["asset_load_timeout"])

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    asset_prim_path = ctx.scene.asset.prim_path

    # --- Pre-checks ---
    ctx.step("Discovering grasp identifiers")
    result_str, identifiers = run_pre_checks(stage, asset_prim_path)
    if result_str is not None:
        msg = result_str.replace("SKIP: ", "")
        if result_str.startswith("SKIP: no grasp_identifier"):
            # No identifiers at all -> structural, skip regardless of validation.
            ctx.skip(msg)
        else:
            ctx.precheck_failure(msg)
        return

    ctx.step("Found %d grasp identifier(s)" % len(identifiers))

    # --- Per-identifier loop ---
    num_failed = 0
    failures = []

    for idx, id_path in enumerate(identifiers):
        id_name = id_path.rsplit("/", 1)[-1]
        safe_name = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in id_name).strip("_") or "grasp"

        ctx.step("Testing identifier %d/%d: %s" % (idx + 1, len(identifiers), id_name))

        # Fresh scene per identifier
        try:
            scene_result = await _test_one_identifier(ctx, cfg, id_path, safe_name, asset_prim_path)
        except Exception as exc:
            scene_result = {
                "phase_name": "Exception",
                "message": str(exc),
                "failed": True,
            }

        # Asset-wide early exits apply equally to every grasp identifier. The
        # helper returns a signal dict so this caller can set the whole-test
        # verdict once and stop without treating None as a phase result.
        if scene_result.get("skip"):
            ctx.skip(scene_result["skip"])
            return
        if scene_result.get("precheck"):
            ctx.precheck_failure(scene_result["precheck"])
            return

        passed = not scene_result.get("failed", False)
        ctx.add_metric("grasp_%s_passed" % safe_name, 1 if passed else 0)

        if not passed:
            num_failed += 1
            failures.append((id_path, scene_result.get("message", "unknown")))
            ctx.log("%sFAILED: %s -- %s" % (variant_tag, id_path, scene_result["message"]))
        else:
            ctx.log("%sPASSED: %s" % (variant_tag, id_path))

    # --- Overall result ---
    total = len(identifiers)
    num_passed = total - num_failed
    ctx.add_metric("grasp_total", total)
    ctx.add_metric("grasp_passed", num_passed)
    ctx.add_metric("grasp_failed", num_failed)

    # Pass if at least one identifier passed; fail only when every
    # identifier failed. Per-identifier failures are still logged above
    # and captured in metrics.
    if num_passed == 0:
        lines = ["%sAll %d grasp identifier(s) failed." % (variant_tag, total)]
        for path, reason in failures:
            lines.append("  %s: %s" % (path, reason))
        lines.append("")
        lines.append("How to fix:")
        lines.append(
            "- The per-identifier reason above identifies which grasp phase failed "
            "(gripper-positioning, grasping, lifting, hold, dropping, shake, "
            "stability, opening). Treat each phase failure independently."
        )
        lines.append(
            "- Verify `physxRigidBody:mass` and `physxRigidBody:diagonalInertia` on "
            "the asset -- objects with zero or NaN mass cannot be grasped."
        )
        lines.append(
            "- Check `physxMaterial:dynamicFriction` / `staticFriction` on the asset "
            "-- low friction prevents the gripper from holding under gravity."
        )
        lines.append(
            "- Confirm the grasp identifier (USD prim path/name) actually points at a "
            "graspable surface; missing or mis-targeted identifiers fail across all "
            "phases consistently."
        )
        lines.append(
            "- Inspect the captured video for each identifier; common visual cues: "
            "gripper passes through asset (collider missing), asset slips out of "
            "fingers (friction too low), asset shoots away on contact (penetration "
            "depth misconfigured)."
        )
        lines.append(
            "- If the asset is intentionally not graspable for some identifiers, "
            "remove those identifiers from the asset's grasp metadata rather than "
            "tuning physics to make them pass."
        )
        ctx.fail("\n".join(lines))
    elif num_failed > 0:
        summary = "%sGrasp passed on %d of %d identifier(s); %d failed." % (variant_tag, num_passed, total, num_failed)
        ctx.log(summary)
        for path, reason in failures:
            ctx.warn("%s%s: %s" % (variant_tag, path, reason))
    elif free_space:
        ctx.step("%sAll %d grasp identifier(s) passed." % (variant_tag, total))


async def _test_one_identifier(ctx, cfg, identifier_path, safe_name, asset_prim_path):
    # type: (...) -> dict
    """Run the full 9-phase test for a single grasp identifier.

    Builds the gripper, runs simulation, encodes video.
    Returns the final result dict.
    """
    import omni.usd
    from isaacsim.core.utils.stage import update_stage_async
    from simready_benchmark_engine_kit.physics_utils import (
        active_physics_engine,
        configure_physx_determinism,
    )
    from simready_benchmark_kit_suite.fet005_grasp.grasp_scene import GraspScene
    from simready_benchmark_kit_suite.fet005_grasp.newton_articulation import (
        apply_temporary_newton_articulations,
        remove_temporary_newton_articulations,
    )

    stage = omni.usd.get_context().get_stage()
    temporary_articulations = []
    physics = None
    grasp_scene = None

    free_space = bool(cfg.get("free_space", False))
    float_height = float(cfg.get("free_space_height", 0.5))

    try:
        # Setup room -- place asset so bbox bottom sits just above ground.
        # This compensates for pivots below the mesh (V1 approach: let gravity
        # settle the object during the stability phase).
        # Free-space variant: same placement code, but the bbox bottom goes
        # `free_space_height` above the floor and nothing settles (zero g).
        room = ctx.scene.add_room()
        room.auto_size(ctx.scene.asset)
        room.set_color(0.3, 0.3, 0.3)
        room.show_ground()
        if free_space:
            # The previous identifier's gripper is only removed at the next
            # "rebuild", i.e. AFTER the asset has been re-placed and its Stability
            # phase run: pads left closed on the old line overlap the re-placed
            # asset and PhysX's depenetration launches it (at zero g it never
            # stops). Remove the stale gripper before placing the asset.
            await _remove_stale_gripper(ctx, stage)
            await _float_asset(ctx, stage, asset_prim_path, float_height)
        else:
            _place_asset_on_ground(stage, asset_prim_path, margin=0.01)

        ctx.scene.lighting.add_dome(intensity=1000.0)

        # Free-space variant: zero gravity until the jaws have closed (restored
        # by the phase hook below); the floating body is kept from sleeping so
        # the gravity change is never ignored by a dormant actor.
        physics = ctx.scene.add_physics(gravity=0.0 if free_space else 9.81, fps=float(cfg["physics_fps"]))
        if free_space:
            n_bodies = _keep_rigid_bodies_awake(stage, asset_prim_path)
            ctx.log(
                "variant: free-space -- asset floated with bbox bottom %.2f m above the floor at its "
                "authored orientation, gravity 0 until the jaws close (%d rigid body(ies) kept awake)"
                % (float_height, n_bodies)
            )

        if active_physics_engine() == "newton":
            temporary_articulations = apply_temporary_newton_articulations(stage, asset_prim_path)
            if temporary_articulations:
                ctx.log(
                    "[newton-articulation] temporary roots: "
                    + ", ".join(mutation.path for mutation in temporary_articulations)
                )

        configure_physx_determinism(stage, float(cfg["physics_fps"]))
        # No fix_mesh_approximations here: run the asset's collision AS AUTHORED,
        # exactly like Isaac. PhysX performs its own runtime convexHull fallback
        # for dynamic triangle-mesh colliders and emits the diagnostic, which the
        # KitLogMonitor captures into kit_logs (instead of us pre-converting and
        # hiding both the error and Isaac's real collision behaviour).

        # DON'T build gripper before physics. Let the object settle
        # naturally without interference (like FET003). The gripper
        # will be built after stability in the rebuild step.
        grasp_scene = GraspScene(ctx, identifier_path, asset_prim_path)
        grasp_scene.init_tracking()

        # Camera + settle + start physics (no gripper on stage yet)
        ctx.scene.setup_camera_follow()
        await ctx.settle(count=3)

        # Cook dynamic mesh colliders (and author missing mass) OFF the timeline
        # path, time-boxed, BEFORE play() -- a cold SDF cook triggered by play()
        # can freeze the run. Reports a clean failure if a collider cannot cook.
        from simready_benchmark_engine_kit.physics_utils import cook_skip_message

        cook_skip = cook_skip_message(await ctx.scene.prepare_physics())
        if cook_skip is not None:
            return {"precheck": cook_skip[5:].strip()}

        physics.stop()
        physics.play()

        if active_physics_engine() == "newton":
            # Newton builds its simulation asynchronously from the composed
            # stage. Pump updates before querying the sim view, then skip
            # unsupported loose/cyclic joint topologies without stepping a
            # broken simulation.
            await ctx.settle(count=3)
            from simready_benchmark_kit_suite.engine_guard import (
                NEWTON_SCENE_SKIP,
                newton_scene_initialized,
            )

            if not newton_scene_initialized():
                return {"skip": NEWTON_SCENE_SKIP}

        ctx.step("Running 9-phase grasp simulation")
        if free_space:
            return await grasp_scene.run_simulation(
                cfg,
                safe_name,
                physics,
                on_phase_complete=_make_free_space_hook(ctx, physics, stage, asset_prim_path, safe_name),
            )
        return await grasp_scene.run_simulation(cfg, safe_name, physics)
    finally:
        robot = getattr(grasp_scene, "robot", None) if grasp_scene is not None else None
        if robot is not None:
            robot.teardown()
        if physics is not None:
            physics.stop()
        if free_space:
            # Let PhysX's reset-on-stop land before the next identifier is placed:
            # the pose write-back is deferred by a frame, and the placement of the
            # next identifier would otherwise read this identifier's final pose.
            for _ in range(3):
                await update_stage_async()
        if temporary_articulations:
            remove_temporary_newton_articulations(stage, temporary_articulations)
            await update_stage_async()


def _place_asset_on_ground(stage, asset_prim_path, margin=0.01):
    # type: (Any, str, float) -> None
    """Move asset so its bounding-box bottom is *margin* above Z=0.

    Compensates for pivots that are below the mesh bottom. Without this,
    objects whose local origin is inside/below the mesh would start
    partially underground.
    """
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(asset_prim_path)
    if not prim or not prim.IsValid():
        return
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedBox()
    bbox_min_z = float(aligned.GetMin()[2])

    # Offset so bbox bottom = margin above ground
    z_offset = -bbox_min_z + margin

    # Apply to the asset root (parent of the asset prim)
    root_path = asset_prim_path.rsplit("/", 1)[0]
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        root_prim = prim
    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_offset))


# ----------------------------------------------------------------------
# Free-space variant helpers (only reached when cfg["free_space"] is True)
# ----------------------------------------------------------------------


async def _remove_stale_gripper(ctx, stage):
    # type: (Any, Any) -> None
    """Delete the gripper articulation left by the previous identifier, if any."""
    from isaacsim.core.utils.stage import update_stage_async
    from simready_benchmark_kit_suite.fet005_grasp.grasp_robot import GraspRobot

    old = stage.GetPrimAtPath(GraspRobot.ROBOT_PATH)
    if old and old.IsValid():
        stage.RemovePrim(GraspRobot.ROBOT_PATH)
        await update_stage_async()
        ctx.log("variant: free-space -- removed the previous identifier's gripper before placing the asset")


def _bbox_bottom(stage, asset_prim_path):
    # type: (Any, str) -> float
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(asset_prim_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    return float(bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox().GetMin()[2])


async def _float_asset(ctx, stage, asset_prim_path, float_height, tolerance=0.005):
    # type: (Any, Any, str, float, float) -> float
    """Place the asset with its bbox bottom `float_height` above the floor and verify it.

    `_place_asset_on_ground` derives the root offset from the CURRENT world
    bbox. Between two identifiers that bbox can still hold the previous run's
    simulated pose (PhysX restores the authored pose one frame after stop), so
    the offset would be computed against a stale pose. Place, pump a frame,
    measure, and repeat until the bottom really sits at float_height.
    """
    from isaacsim.core.utils.stage import update_stage_async

    bottom = float("nan")
    for attempt in range(1, 6):
        _place_asset_on_ground(stage, asset_prim_path, margin=float_height)
        await update_stage_async()
        bottom = _bbox_bottom(stage, asset_prim_path)
        if abs(bottom - float_height) <= tolerance:
            break
    ctx.log(
        "variant: free-space -- asset bbox bottom at z=%.3f m (target %.2f m) after %d placement pass(es)"
        % (bottom, float_height, attempt)
    )
    if abs(bottom - float_height) > tolerance:
        ctx.warn("variant: free-space -- asset could not be floated: bbox bottom z=%.3f m, target %.2f m" % (bottom, float_height))
    return bottom


def _keep_rigid_bodies_awake(stage, asset_prim_path):
    # type: (Any, str) -> int
    """Set physxRigidBody:sleepThreshold = 0 on every rigid body under the asset.

    A body floating at zero gravity is perfectly still, so PhysX would put it
    to sleep before the pads arrive; a sleeping actor does not feel a later
    gravity change. Returns the number of bodies touched.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(asset_prim_path)
    if not root or not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.CreateSleepThresholdAttr().Set(0.0)
            count += 1
    return count


def _wake_rigid_bodies(stage, asset_prim_path):
    # type: (Any, str) -> str
    """Ask PhysX to wake every rigid body under the asset; returns a short status."""
    try:
        import omni.physx
        import omni.usd
        from pxr import Usd, UsdPhysics

        physx = omni.physx.get_physx_interface()
        if not hasattr(physx, "wake_up"):  # not in every PhysX binding (absent in Kit 110 / Isaac 6.0.1)
            return "kept awake by sleepThreshold=0"
        stage_id = omni.usd.get_context().get_stage_id()
        root = stage.GetPrimAtPath(asset_prim_path)
        woken = 0
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                physx.wake_up(stage_id, str(prim.GetPath()))
                woken += 1
        return "woke %d body(ies)" % woken
    except Exception as exc:  # the sleepThreshold=0 authored before play() is the real guarantee
        return "wake_up failed (%s); kept awake by sleepThreshold=0" % exc


def _asset_pose(stage, asset_prim_path):
    # type: (Any, str) -> Optional[Tuple[Any, Any]]
    """(world translation Gf.Vec3d, world rotation Gf.Quatd) of the asset body."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(asset_prim_path)
    if not prim or not prim.IsValid():
        return None
    xf = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim).RemoveScaleShear()
    return (xf.ExtractTranslation(), xf.ExtractRotationQuat())


def _pose_delta(before, after):
    # type: (Any, Any) -> Tuple[float, float]
    """(translation shift in metres, rotation angle in degrees) between two poses."""
    import math

    if before is None or after is None:
        return (0.0, 0.0)
    shift = float((after[0] - before[0]).GetLength())
    q = after[1] * before[1].GetInverse()
    w = max(-1.0, min(1.0, abs(float(q.GetReal()))))
    return (shift, math.degrees(2.0 * math.acos(w)))


def _make_free_space_hook(ctx, physics, stage, asset_prim_path, safe_name):
    # type: (...) -> Any
    """Phase-completion hook for the free-space variant.

    GripperPositioning done  -> remember the asset pose before the jaws move.
    Grasping done (closed)   -> restore gravity to 9.81 m/s^2, wake the asset,
                                report how far the zero-g closure pushed / spun it.
    Opening done (released)  -> switch the pad cubes' collision off, so the
                                Dropping check measures the release itself: on a
                                vertical closing line the lower pad sits under
                                the object and would otherwise catch it (the
                                object rests on the retracted pad and "falls"
                                only by the joint travel).
    """
    state = {"pose_before_close": None}

    def pad_gap_mm(scene, tracker):
        # distance left between the two pad faces = what the jaws are pinching
        try:
            props = scene.scene_properties
            grasp_dist = float(props["gripper_position_info"]["grasp_distance"])
            pad = float(props["gripper_pad_properties"]["scale"])
            most_closed = float(tracker.get_most_closed_joint_position())
            return (grasp_dist - pad - 2.0 * abs(most_closed)) * 1000.0
        except Exception:
            return float("nan")

    def hook(phase_result, scene, tracker):
        name = phase_result.get("phase_name")
        if phase_result.get("failed", False):
            ctx.log(
                "[variant: free-space] %s failed at frame %d; pad faces %.1f mm apart at the most-closed point"
                % (name, int(phase_result.get("frame", -1)), pad_gap_mm(scene, tracker))
            )
            return
        if name == "GripperPositioning":
            state["pose_before_close"] = _asset_pose(stage, asset_prim_path)
        elif name == "Grasping":
            physics.set_gravity(9.81)
            wake = _wake_rigid_bodies(stage, asset_prim_path)
            shift, rot = _pose_delta(state["pose_before_close"], _asset_pose(stage, asset_prim_path))
            gap = pad_gap_mm(scene, tracker)
            ctx.add_metric("grasp_%s_free_space_closure_shift_m" % safe_name, round(shift, 4))
            ctx.add_metric("grasp_%s_free_space_closure_rotation_deg" % safe_name, round(rot, 2))
            ctx.add_metric("grasp_%s_free_space_pad_gap_after_close_mm" % safe_name, round(gap, 1))
            note = (
                "variant: free-space -- jaws closed at zero g: the asset shifted %.1f mm and rotated "
                "%.1f deg during closure, pad faces %.1f mm apart (estimated from joint travel); gravity "
                "restored to 9.81 m/s^2 at frame %d (%s)" % (shift * 1000.0, rot, gap, int(phase_result.get("frame", -1)), wake)
            )
            ctx.log(note)  # ctx.log is what result.json keeps (and it reaches the event stream too)
        elif name == "Opening":
            try:
                scene._set_pad_collision_enabled(False)
                ctx.log("variant: free-space -- gripper opened at frame %d; pad collision switched off so the drop "
                        "check measures the release, not the lower pad acting as a shelf" % int(phase_result.get("frame", -1)))
            except Exception as exc:
                ctx.log("[variant: free-space] could not disable pad collision after opening: %s" % exc)

    return hook
