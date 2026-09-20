# SPDX-License-Identifier: Apache-2.0
"""Run one of NVIDIA's registered Foundation 7.1 runtime tests, unmodified, in this Kit.

This is what engine-kit's execute_single_test does to prepare and call one test (fresh stage,
KitSceneHandle, KitEngineProxy, RunContext), with those two engine-kit objects replaced by
subclasses:
  Scene(KitSceneHandle): load_asset also selects the asset's runtime physics variant for the engine
    (scene.select_runtime_variant); under Newton, prepare_physics leaves collider cooking to the
    engine ("Any runtime-specific cooking or fallback behavior is owned by the selected engine",
    7.1 docs/fet005/grasp-and-lift.md) instead of waiting on PhysX property queries nothing answers.
    Under Newton, update_camera_follow gives engine-kit's camera follow the live bound.
  Proxy(KitEngineProxy): under Newton, get_asset_bounds places the asset's rigid bodies with their
    Fabric poses (reading.py); every physics_step records the asset's trajectory, and the first
    one records which simulation actually runs.
"""
import math

TESTS = {
    "drop": ("simready_benchmark_kit_suite.fet003_physics.ground_drop", "ground_drop"),
    "slope": ("simready_benchmark_kit_suite.fet003_physics.slope_drop", "slope_drop"),
    "grasp": ("simready_benchmark_kit_suite.fet005_grasp.grasp_and_lift", "grasp_and_lift"),
    # ours, because NVIDIA's tests all refuse an asset without UsdPhysics.RigidBodyAPI
    "deformable_drop": ("asset_checks.experiments.deformable_drop", "deformable_drop"),
    "deformable_press": ("asset_checks.experiments.deformable_press", "deformable_press"),
}


def registered(experiment):
    import importlib

    from simready_benchmark.core.decorator import get_registered_tests

    module_name, test_name = TESTS[experiment]
    module = importlib.import_module(module_name)
    found = [d for d in get_registered_tests() if d.name == test_name and d.func.__module__ == module.__name__]
    if len(found) != 1:
        raise RuntimeError(f"expected NVIDIA's {test_name} registered once by {module_name}, found {len(found)}")
    return found[0]


class Recorder:
    """The asset's rigid bodies once per physics step: bound, lowest collider vertex, speeds, tilt.
    `t` counts steps at the test's fps; `timeline_t` is the timeline's own time and, under Newton,
    `sim_t` the time Isaac's Newton stage has simulated -- where they drift from `t`, a step did not
    advance physics by one frame."""

    def __init__(self, engine, dt):
        self.engine, self.dt = engine, dt
        self.engine_observed = None
        self.bodies = None
        self.traj = {"t": [], "timeline_t": [], "sim_t": [], "z_min": [], "lowest_vertex_z": [], "lin": [], "ang": [],
                     "tilt_deg": [], "source": []}

    def sample(self, stage):
        from asset_checks.kit import reading
        from asset_checks.kit.scene import ASSET_PRIM

        if self.engine_observed is None:
            self.engine_observed = reading.active_engine()
        if self.bodies is None:
            self.bodies = [str(p.GetPath()) for p in reading.rigid_bodies(stage, ASSET_PRIM)]
            self.points = reading.collider_points(stage, self.bodies) if self.bodies else {}
            self.init = self.prev = None
        if not self.bodies:
            return
        mats, source = reading.body_matrices(stage, self.bodies, self.engine)
        if self.init is None:
            self.init = self.prev = mats
        bound = reading.world_bound(stage, ASSET_PRIM, mats)
        t = self.traj
        t["t"].append(round(len(t["t"]) * self.dt + self.dt, 5))
        import omni.timeline

        t["timeline_t"].append(round(omni.timeline.get_timeline_interface().get_current_time(), 5))
        if self.engine == "newton":
            import isaacsim.physics.newton as isaac_newton

            t["sim_t"].append(round(float(isaac_newton.acquire_stage().sim_time), 5))
        t["z_min"].append(round(bound[0][2], 5))
        t["lowest_vertex_z"].append(round(reading.lowest_point(self.points, mats), 5))
        t["lin"].append(round(max(math.dist(mats[b][3][:3], self.prev[b][3][:3]) for b in self.bodies) / self.dt, 5))
        t["ang"].append(round(max(reading.rotation_angle(self.prev[b], mats[b]) for b in self.bodies) / self.dt, 5))
        t["tilt_deg"].append(round(max(reading.rotation_angle(self.init[b], mats[b]) for b in self.bodies) * 180.0 / math.pi, 2))
        t["source"].append(source)
        self.prev = mats


def _classes(engine, asset_path, recorder, contact_profile, dump_dir=None, trace=None, test_config=None, camera_mode="fixed",
             visual_cues=True, solver="physx"):
    from simready_benchmark_engine_kit.kit_engine_proxy import BoundsResult, KitEngineProxy
    from simready_benchmark_engine_kit.scene_handle import KitSceneHandle

    from asset_checks.kit import reading
    from asset_checks.kit import scene as scene_mod

    class Scene(KitSceneHandle):
        variant = None
        contact_applied = []  # one report per play() while a contact profile is set
        plays = 0

        def add_physics(self, gravity=9.81, fps=240.0):
            physics = super().add_physics(gravity=gravity, fps=fps)
            from asset_checks.kit import contact

            play = physics.play

            def play_counted():  # the profile is authored before every play, so a rebuilt gripper gets it too
                if Scene.solver is None:  # the scene exists by the first play; a rebuild keeps the schema
                    Scene.solver = scene_mod.select_solver(self._stage, engine, solver)
                if contact_profile is not None:
                    Scene.contact_applied.append(contact.apply(self._stage, contact_profile))
                Scene.plays += 1
                return play()

            physics.play = play_counted
            return physics

        def load_asset(self, *args, **kwargs):
            handle = super().load_asset(*args, **kwargs)
            Scene.variant = scene_mod.select_runtime_variant(self._stage, asset_path, engine)
            return handle

        camera = None  # how the video camera was placed, recorded in result.json
        solver = None  # which solver was asked for and how, recorded in result.json
        look = None  # the floor grid and key light added for the videos, recorded in result.json

        def setup_camera_follow(self, config=None):
            # every test sets its camera up once, after building the room and placing the asset
            if Scene.look is None:
                if visual_cues:
                    bodies = [str(p.GetPath()) for p in reading.rigid_bodies(self._stage, scene_mod.ASSET_PRIM)]
                    lo, hi = reading.world_bound(self._stage, scene_mod.ASSET_PRIM, reading.body_matrices(self._stage, bodies, engine)[0] if bodies else {})
                    Scene.look = scene_mod.add_visual_cues(self._stage, ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2))
                else:
                    Scene.look = {"added": False, "reason": "requested plain scene"}
            return super().setup_camera_follow(config)

        def _place_fixed_camera(self, config):
            """One camera pose for the whole test, framing where its objects can be: the asset and the
            gripper as they are at the first camera update (the gripper exists by then), down to the floor,
            and up by the test's largest lift (lift_max_height). engine-kit's own framing math places it;
            only keys the test config sets are passed, the rest are that function's defaults. A slope test
            does not bound how far the asset travels, so it keeps engine-kit's follow."""
            import math

            from simready_benchmark_engine_kit import camera_follow
            from simready_benchmark_kit_suite.fet005_grasp.grasp_robot import GraspRobot

            cfg = dict(test_config or {})
            if "slope_angle_deg" in cfg:
                return {"mode": "follow", "reason": "slope_drop does not bound how far the asset travels"}
            state = getattr(self, "_camera_follow_state", None)
            if state is None:
                return {"mode": "none", "reason": "the test set up no camera"}
            lo, hi = [math.inf] * 3, [-math.inf] * 3
            for root in (scene_mod.ASSET_PRIM, GraspRobot.ROBOT_PATH):
                if not self._stage.GetPrimAtPath(root).IsValid():
                    continue
                bodies = [str(p.GetPath()) for p in reading.rigid_bodies(self._stage, root)]
                box = reading.world_bound(self._stage, root, reading.body_matrices(self._stage, bodies, engine)[0] if bodies else {})
                lo, hi = [min(a, b) for a, b in zip(lo, box[0])], [max(a, b) for a, b in zip(hi, box[1])]
            lo[2] = min(lo[2], float(cfg.get("floor_level", 0.0)))
            hi[2] += float(cfg.get("lift_max_height", 0.0))
            keys = {"camera_fit_mode": "fit_mode", "camera_margin_factor": "margin_factor", "camera_direction": "direction"}
            kwargs = {arg: (config or {})[key] for key, arg in keys.items() if key in (config or {})}
            params = camera_follow.compute_target_from_bbox(center=tuple((a + b) / 2 for a, b in zip(lo, hi)),
                                                            size=tuple(b - a for a, b in zip(lo, hi)), **kwargs)
            camera_follow.apply_camera_params(self._stage, state.camera_prim_path, params)
            return {"mode": "fixed", "box": [[round(v, 4) for v in lo], [round(v, 4) for v in hi]]}

        def update_camera_follow(self, config=None, update_history=True):
            """camera_mode "fixed": the first call places one camera for the whole test, later calls do
            nothing. Otherwise engine-kit's follow, where one fix applies: its follow bounds the asset from
            USD, which Newton does not update, so under Newton that one read gets the live bound; the rest
            of engine-kit's follow logic (history, smoothing, zoom-out) is unchanged."""
            import omni.timeline

            if Scene.camera is None:
                Scene.camera = self._place_fixed_camera(config) if camera_mode == "fixed" else {"mode": "follow", "reason": "requested"}
            if Scene.camera["mode"] != "follow":
                return None

            if engine == "physx" or omni.timeline.get_timeline_interface().is_stopped():
                return super().update_camera_follow(config, update_history)
            from unittest import mock

            from simready_benchmark_engine_kit import camera_follow

            bodies = [str(p.GetPath()) for p in reading.rigid_bodies(self._stage, scene_mod.ASSET_PRIM)]
            if not bodies:  # nothing simulated: USD is current
                return super().update_camera_follow(config, update_history)
            lo, hi = reading.world_bound(self._stage, scene_mod.ASSET_PRIM, reading.body_matrices(self._stage, bodies, engine)[0])
            live = ([float(v) for v in lo], [float(v) for v in hi])
            with mock.patch.object(camera_follow, "_compute_asset_bbox", lambda stage, path: live):
                return super().update_camera_follow(config, update_history)

        async def prepare_physics(self, timeout_ms=None):
            if engine == "physx":
                return await super().prepare_physics(timeout_ms=timeout_ms)
            return {}

    class Proxy(KitEngineProxy):
        def get_asset_bounds(self):
            import omni.timeline

            if engine == "physx" or omni.timeline.get_timeline_interface().is_stopped():
                return super().get_asset_bounds()
            stage = self._scene_handle._stage
            path = scene_mod.ASSET_PRIM if stage.GetPrimAtPath(scene_mod.ASSET_PRIM).IsValid() else "/World/AssetRoot"
            bodies = [str(p.GetPath()) for p in reading.rigid_bodies(stage, path)]
            if not bodies:  # nothing simulated: the authored USD bound is current
                return super().get_asset_bounds()
            mn, mx = reading.world_bound(stage, path, reading.body_matrices(stage, bodies, engine)[0])
            return BoundsResult(tuple(mn), tuple(mx))

        solver_seen = []  # what MuJoCo compiled, read on the first step after each play (Newton)
        notes = []  # what this engine did to the asset that its USD did not ask for
        stepped_plays = 0  # plays whose first step re-armed engine-kit's pause
        physics_dumps = []  # contact.dump() of the same moment, when the run asks for it
        measured_plays = 0

        async def physics_step(self):
            # engine-kit pauses the timeline after the first update of a play so that each step is one
            # frame (its docstring), but sets _physics_paused once and never clears it: after a second
            # play (the grasp test plays twice) the timeline kept playing and every step advanced two
            # frames, more on capture steps. Clearing it on the first step of each play restores the
            # documented behavior; run.py's frames-per-step gate checks the result.
            if Proxy.stepped_plays < Scene.plays:
                Proxy.stepped_plays = Scene.plays
                self._physics_paused = False
            await super().physics_step()
            recorder.sample(self._scene_handle._stage)
            if trace is not None and Scene.plays:
                trace.sample(recorder.traj["t"][-1] if recorder.traj["t"] else 0.0)
            if engine == "newton" and Proxy.measured_plays < Scene.plays:
                from asset_checks.kit import contact

                Proxy.measured_plays = Scene.plays
                Proxy.solver_seen.append(contact.measure())
                if not Proxy.notes:  # the model exists once the first play has built it
                    Proxy.notes = reading.stack_notes(self._scene_handle._stage, engine, scene_mod.ASSET_PRIM)
                if dump_dir is not None:
                    Proxy.physics_dumps.append(contact.dump(f"{dump_dir}/physics_play{Scene.plays}.npz"))

    return Scene, Proxy


async def run(req):
    import traceback

    from simready_benchmark.runner.test_context import PrecheckAborted, RunContext
    from simready_benchmark_engine_kit.kit_runner import _build_media_list, _get_kit_version

    from asset_checks.kit import nvidia_api, pr2
    from asset_checks.kit import scene as scene_mod

    engine, asset, out = req["engine"], req["asset"], req["out_dir"]
    supplied = nvidia_api.install(engine)
    defn = registered(req["experiment"])
    config = dict(defn.config_defaults)
    unknown = set(req.get("config", {})) - set(config)
    if unknown:
        raise ValueError(f"unknown {defn.name} config keys {sorted(unknown)}; known: {sorted(config)}")
    config.update(req.get("config", {}))
    from asset_checks.kit import contact

    profile_name = req.get("contact_profile", "stock")
    if profile_name not in contact.PROFILES:
        raise ValueError(f"unknown contact profile {profile_name!r}; known: {sorted(contact.PROFILES)}")
    # PR #2's settings are MuJoCo's: they are authored as mjc:/newton: contact attributes that only
    # SolverMuJoCo reads. A PhysX run ignores the flag, as it ignores every Newton-only option; a
    # Newton run on another solver would author values nothing consumes, so that is refused.
    if profile_name != "stock" and engine == "newton" and req["solver"] != "mujoco":
        raise ValueError(f"contact profile {profile_name!r} is MuJoCo's; this Newton run uses {req['solver']!r}")
    profile = contact.PROFILES[profile_name] if engine == "newton" and req["solver"] == "mujoco" else None
    recorder = Recorder(engine, 1.0 / float(config.get("physics_fps", 240)))
    trace = contact.Trace(req["trace_contacts"]) if engine == "newton" and req.get("trace_contacts") else None
    Scene, Proxy = _classes(engine, asset, recorder, profile, out if req.get("dump_physics") else None, trace,
                            config, req.get("camera", "fixed"), req.get("visual_cues", True), req["solver"])

    stage = await scene_mod.new_stage()
    handle = Scene(stage)
    px = req.get("capture_px")  # None: engine-kit's own capture size
    proxy = Proxy(out) if px is None else Proxy(out, width=int(px), height=int(px))
    proxy._scene_handle = handle
    ctx = RunContext(
        asset_path=asset, asset_rel_path=asset, output_dir=out, test_name=defn.name,
        test_version=getattr(defn, "version", "") or "", feature_id="", engine_name="kit",
        engine_version=_get_kit_version(), session_id="asset_checks", scene=handle, config=config,
        engine_session=proxy, test_feature_names=[], asset_validated_features=req["validated_features"],
    )
    error = None
    try:
        await defn.func(ctx)
    except PrecheckAborted:
        pass
    except Exception as exc:  # noqa: BLE001 - as execute_single_test: the test failed
        error = traceback.format_exc()
        if not ctx.failed and not ctx.skipped:
            ctx.fail("Test raised exception: %s" % exc)
    status = "fail" if ctx.failed else ("skipped" if ctx.skipped else "pass")
    result = {
        "test": defn.name, "test_version": getattr(defn, "version", ""), "verdict": status,
        "message": ctx.failure_message or ctx.skip_reason or "", "metrics": ctx.metrics,
        "warnings": ctx.warnings, "logs": ctx.log_messages, "media": _build_media_list(ctx, out),
        "test_exception": error, "nvidia_api_supplied": supplied,
        "physx_cook_skips_withheld": list(nvidia_api.WITHHELD_COOK_SKIPS), "variant": Scene.variant,
        "contact_profile": profile_name if profile is not None else "stock", "contact_applied": Scene.contact_applied,
        "solver_seen": Proxy.solver_seen,
        "physics_dumps": Proxy.physics_dumps,
        "camera": Scene.camera,
        "solver": Scene.solver,
        "notes": Proxy.notes,
        "look": Scene.look,
        "contact_trace": trace.rows if trace is not None else None,
        "engine_observed": recorder.engine_observed, "pose_source": sorted(set(recorder.traj["source"])),
        "rigid_bodies": recorder.bodies, "trajectory": recorder.traj,
    }
    if req["experiment"] in ("drop", "slope") and recorder.traj["t"]:
        # NVIDIA's slope (45 deg) expects the asset to keep sliding: PR #2's rest criterion belongs to its
        # own walled 15 deg slope, so on this slope only its tunnel / explode checks apply
        result["pr2_criteria"] = pr2.evaluate(recorder.traj, float(config.get("floor_level", 0.0)), expect_rest=req["experiment"] == "drop")
    return result
