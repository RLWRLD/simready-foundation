# SPDX-License-Identifier: Apache-2.0
"""Ground drop in any engine: NVIDIA's FET003 `ground_drop` procedure, parameters and verdict.

The scene, stepping, capture, parameters (the registered test's config defaults) and stability
checks are NVIDIA's (engine-kit, simready_benchmark_kit_suite); only the pose reads are this
package's (reading.py), because NVIDIA's reads USD, which Newton does not update. The settle /
tunnel criteria of RLWRLD's standalone Newton runner (PR #2, newton15_cert.py) are computed on the
same trajectory for comparison; PR #2 drops from 1 cm and uses mesh vertices, this drop is NVIDIA's
(twice the bounding-box height) and its heights come from the bounding box.
"""

import time

# newton15_cert.py (branch rlwrld/newton-conformance): SETTLE_LIN, SETTLE_ANG, SETTLE_HOLD,
# TUNNEL_DEPTH, EXPLODE_SPEED; "more than 5 frames at 100 Hz" below the support is 0.05 s here.
PR2_CRITERIA = {
    "settle_lin_m_s": 2e-2,
    "settle_ang_rad_s": 5e-1,
    "settle_hold_s": 0.5,
    "tunnel_depth_m": 0.01,
    "tunnel_hold_s": 0.05,
    "explode_speed_m_s": 10.0,
    "soft_penetration_m": 0.003,
    "tilt_deg": 20.0,
}


class _Bounds:
    def __init__(self, bound):
        self.min, self.max = bound


def nvidia_config():
    """Config defaults of NVIDIA's registered ground_drop test."""
    from simready_benchmark_kit_suite.fet003_physics import ground_drop as nvidia_test
    from simready_benchmark.core.decorator import get_registered_tests

    found = [d for d in get_registered_tests() if d.name == "ground_drop" and d.func.__module__ == nvidia_test.__name__]
    if len(found) != 1:
        raise RuntimeError(f"expected NVIDIA's ground_drop registered once, found {len(found)}")
    return dict(found[0].config_defaults)


async def run(req):
    import math

    from simready_benchmark_engine_kit.kit_engine_proxy import KitEngineProxy
    from simready_benchmark_engine_kit.scene_handle import KitSceneHandle
    from simready_benchmark_kit_suite.fet003_physics import ground_drop as nvidia_test
    from simready_benchmark_kit_suite.fet003_physics.stability import bounds_to_history_entry, check_rest_window

    from asset_checks.kit import reading
    from asset_checks.kit import scene as scene_mod

    engine, asset, out = req["engine"], req["asset"], req["out_dir"]
    cfg = nvidia_config()
    unknown = set(req.get("config", {})) - set(cfg)
    if unknown:
        raise ValueError(f"unknown ground_drop config keys {sorted(unknown)}; known: {sorted(cfg)}")
    cfg.update(req.get("config", {}))
    fps = int(cfg["physics_fps"])
    dt = 1.0 / fps
    touch_threshold = float(cfg["floor_level"]) + float(cfg["floor_margin"])
    pen_threshold = float(cfg["floor_level"]) - float(cfg["floor_margin"])
    pen_check_frames = int(float(cfg["penetration_check_seconds"]) * fps)
    rest_tol = float(cfg["rest_tolerance"])
    hold_frames = int(float(cfg["rest_detection_hold_seconds"]) * fps)
    max_rest_frames = int(float(cfg["rest_detection_max_seconds"]) * fps)
    after_result_frames = int(float(cfg["simulate_after_result_seconds"]) * fps)
    capture_interval = max(1, fps // int(cfg["capture_fps"]))
    total_frames = int(float(cfg["simulation_seconds"]) * fps)

    # --- scene: NVIDIA's ground_drop setup, in its order (minus its PhysX-only precheck) ---
    stage = await scene_mod.new_stage()
    handle = KitSceneHandle(stage)
    proxy = KitEngineProxy(out + "/frames", width=int(req.get("capture_px", 512)), height=int(req.get("capture_px", 512)))
    proxy._scene_handle = handle  # as engine-kit's execution.py wires it
    handle.load_asset(asset, timeout=cfg["asset_load_timeout"])
    variant = scene_mod.select_runtime_variant(stage, asset, engine)
    room = handle.add_room()
    room.auto_size(handle.asset)
    room.set_color(0.3, 0.4, 0.7)
    room.show_ground(color=(0.25, 0.35, 0.6))
    room.place_asset_above_ground(height_factor=cfg["drop_height_factor"])
    handle.lighting.add_dome(intensity=1000.0)
    physics = handle.add_physics(gravity=9.81, fps=float(fps))
    handle.enable_ground_plane(friction=0.5)
    nvidia_test._apply_ground_physics_material()
    handle.setup_camera_follow()
    await proxy.settle(int(cfg["settle_frames"]) * 3)
    bodies = [str(p.GetPath()) for p in reading.rigid_bodies(stage, scene_mod.ASSET_PRIM)]
    if not bodies:
        raise RuntimeError(f"no rigid body under {scene_mod.ASSET_PRIM}: nothing to drop")

    physics.play()
    mats, source = reading.body_matrices(stage, bodies, engine)
    init = reading.world_bound(stage, scene_mod.ASSET_PRIM, mats)
    init_mats = mats
    traj = {"t": [], "z_min": [], "center_z": [], "lin": [], "ang": [], "source": []}
    frames, history = [], []
    touched, touch_frame, penetrated = False, -1, False
    rest_detected, rest_frame, result_frame = False, -1, -1
    prev_z, prev_mats = None, mats
    engine_observed = None
    deadline = time.monotonic() + 120.0  # NVIDIA's watchdog
    hung = False
    for frame in range(total_frames):
        await proxy.physics_step()
        if engine_observed is None:
            engine_observed = reading.active_engine()
        if time.monotonic() > deadline:
            hung = True
            break
        mats, source = reading.body_matrices(stage, bodies, engine)
        bound = reading.world_bound(stage, scene_mod.ASSET_PRIM, mats)
        z_min = bound[0][2]
        history.append(bounds_to_history_entry(_Bounds(bound)))
        lin = max(math.dist(mats[b][3][:3], prev_mats[b][3][:3]) for b in bodies) / dt
        ang = max(reading.rotation_angle(prev_mats[b], mats[b]) for b in bodies) / dt
        prev_mats = mats
        traj["t"].append(round((frame + 1) * dt, 5))
        traj["z_min"].append(round(z_min, 5))
        traj["center_z"].append(round((bound[0][2] + bound[1][2]) / 2.0, 5))
        traj["lin"].append(round(lin, 5))
        traj["ang"].append(round(ang, 5))
        traj["source"].append(source)

        if not touched and prev_z is not None and prev_z > touch_threshold and z_min <= touch_threshold:
            touched, touch_frame = True, frame
        if touched and not penetrated and z_min < pen_threshold:
            penetrated = True
            if result_frame < 0:
                result_frame = frame
        if touched and not rest_detected and not penetrated and frame - touch_frame >= pen_check_frames:
            if check_rest_window(history, hold_frames, rest_tol, rest_tol):
                rest_detected, rest_frame = True, frame
                if result_frame < 0:
                    result_frame = frame
            elif frame - touch_frame >= max_rest_frames and result_frame < 0:
                result_frame = frame
        handle.update_camera_follow()
        prev_z = z_min
        if frame % capture_interval == 0:
            frames.append(await proxy.capture_frame(label="drop"))
        if result_frame >= 0 and frame - result_frame >= after_result_frames:
            break
        await proxy.physics_advance()

    # --- NVIDIA's verdict (ground_drop._report_result), same order and wording ---
    passed = touched and not penetrated and rest_detected and not hung
    if hung:
        message = "Physics simulation hung (exceeded 120s watchdog)."
    elif passed:
        message = "Ground drop + stability PASSED: touched at %.2fs, settled at %.2fs" % (touch_frame * dt, rest_frame * dt)
    elif not touched:
        message = "Ground drop FAILED: Object never touched the ground within %.0f seconds." % float(cfg["simulation_seconds"])
    elif penetrated:
        message = "Ground drop FAILED: Object penetrated the ground plane."
    else:
        message = "Ground drop FAILED: Object touched the ground but never came to rest within %.1f seconds after touch." % float(cfg["rest_detection_max_seconds"])
    nvidia = {
        "passed": passed, "message": message, "touched": touched, "penetrated": penetrated, "rest_detected": rest_detected,
        "touch_time_s": round(touch_frame * dt, 3) if touch_frame >= 0 else None,
        "rest_time_s": round(rest_frame * dt, 3) if rest_frame >= 0 else None,
        "initial_z_min": round(init[0][2], 4), "bbox_height": round(init[1][2] - init[0][2], 4),
        "touch_threshold": touch_threshold, "penetration_threshold": pen_threshold,
    }

    # --- PR #2's criteria on the same trajectory ---
    c, floor = PR2_CRITERIA, float(cfg["floor_level"])
    depth = [floor - z for z in traj["z_min"]]
    hold = max(1, int(round(c["tunnel_hold_s"] * fps)))
    run_below, tunnel = 0, False
    for d in depth:
        run_below = run_below + 1 if d > c["tunnel_depth_m"] else 0
        tunnel = tunnel or run_below > hold
    settle_n = max(1, int(round(c["settle_hold_s"] * fps)))
    quiet, settle_time = 0, None
    for i, (v, w) in enumerate(zip(traj["lin"], traj["ang"])):
        quiet = quiet + 1 if (v < c["settle_lin_m_s"] and w < c["settle_ang_rad_s"]) else 0
        if touched and i >= touch_frame and quiet >= settle_n:
            settle_time = round(traj["t"][i - settle_n + 1], 3)
            break
    tilt = max(reading.rotation_angle(init_mats[b], mats[b]) for b in bodies) * 180.0 / math.pi
    exploded = max(traj["lin"], default=0.0) > c["explode_speed_m_s"]
    max_pen = max(depth, default=0.0)
    final_depth = depth[-1] if depth else 0.0
    pr2_fail = "flew away / exploded" if exploded else ("tunnelled into the support" if tunnel or final_depth > c["tunnel_depth_m"] else ("did not come to rest" if settle_time is None else None))
    notes = []
    if max_pen > c["soft_penetration_m"]:
        notes.append(f"soft contact: dips {max_pen * 1000:.1f} mm below the floor")
    if tilt > c["tilt_deg"]:
        notes.append(f"tipped over ({tilt:.0f} deg from the drop pose)")
    pr2 = {"passed": pr2_fail is None, "message": pr2_fail or "; ".join(notes), "settle_time_s": settle_time,
           "max_penetration_m": round(max_pen, 4), "final_depth_m": round(final_depth, 4), "tilt_deg": round(tilt, 1),
           "max_speed_m_s": round(max(traj["lin"], default=0.0), 3), "criteria": c}

    return {
        "experiment": "drop", "engine_observed": engine_observed, "pose_source": sorted(set(traj["source"])),
        "variant": variant, "bodies": bodies, "config": cfg, "nvidia": nvidia, "pr2_criteria": pr2,
        "frames": frames, "trajectory": traj,
    }
