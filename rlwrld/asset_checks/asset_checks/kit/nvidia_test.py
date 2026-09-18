# SPDX-License-Identifier: Apache-2.0
"""Run one of NVIDIA's registered Foundation 7.1 runtime tests, unmodified, in this Kit.

This is what engine-kit's execute_single_test does to prepare and call one test (fresh stage,
KitSceneHandle, KitEngineProxy, RunContext), with those two engine-kit objects replaced by
subclasses:
  Scene(KitSceneHandle): load_asset also selects the asset's runtime physics variant for the engine
    (scene.select_runtime_variant); under Newton, prepare_physics leaves collider cooking to the
    engine ("Any runtime-specific cooking or fallback behavior is owned by the selected engine",
    7.1 docs/fet005/grasp-and-lift.md) instead of waiting on PhysX property queries nothing answers.
  Proxy(KitEngineProxy): under Newton, get_asset_bounds places the asset's rigid bodies with their
    Fabric poses (reading.py); every physics_step records the asset's trajectory, and the first
    one records which simulation actually runs.
"""
import math

TESTS = {
    "drop": ("simready_benchmark_kit_suite.fet003_physics.ground_drop", "ground_drop"),
    "slope": ("simready_benchmark_kit_suite.fet003_physics.slope_drop", "slope_drop"),
    "grasp": ("simready_benchmark_kit_suite.fet005_grasp.grasp_and_lift", "grasp_and_lift"),
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
    """The asset's rigid bodies once per physics step: bound, lowest collider vertex, speeds, tilt."""

    def __init__(self, engine, dt):
        self.engine, self.dt = engine, dt
        self.engine_observed = None
        self.bodies = None
        self.traj = {"t": [], "z_min": [], "lowest_vertex_z": [], "lin": [], "ang": [], "tilt_deg": [], "source": []}

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
        t["z_min"].append(round(bound[0][2], 5))
        t["lowest_vertex_z"].append(round(reading.lowest_point(self.points, mats), 5))
        t["lin"].append(round(max(math.dist(mats[b][3][:3], self.prev[b][3][:3]) for b in self.bodies) / self.dt, 5))
        t["ang"].append(round(max(reading.rotation_angle(self.prev[b], mats[b]) for b in self.bodies) / self.dt, 5))
        t["tilt_deg"].append(round(max(reading.rotation_angle(self.init[b], mats[b]) for b in self.bodies) * 180.0 / math.pi, 2))
        t["source"].append(source)
        self.prev = mats


def _classes(engine, asset_path, recorder):
    from simready_benchmark_engine_kit.kit_engine_proxy import BoundsResult, KitEngineProxy
    from simready_benchmark_engine_kit.scene_handle import KitSceneHandle

    from asset_checks.kit import reading
    from asset_checks.kit import scene as scene_mod

    class Scene(KitSceneHandle):
        variant = None

        def load_asset(self, *args, **kwargs):
            handle = super().load_asset(*args, **kwargs)
            Scene.variant = scene_mod.select_runtime_variant(self._stage, asset_path, engine)
            return handle

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

        async def physics_step(self):
            await super().physics_step()
            recorder.sample(self._scene_handle._stage)

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
    recorder = Recorder(engine, 1.0 / float(config.get("physics_fps", 240)))
    Scene, Proxy = _classes(engine, asset, recorder)

    stage = await scene_mod.new_stage()
    handle = Scene(stage)
    proxy = Proxy(out, width=int(req.get("capture_px", 512)), height=int(req.get("capture_px", 512)))
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
        "test_exception": error, "nvidia_api_supplied": supplied, "variant": Scene.variant,
        "engine_observed": recorder.engine_observed, "pose_source": sorted(set(recorder.traj["source"])),
        "rigid_bodies": recorder.bodies, "trajectory": recorder.traj,
    }
    if req["experiment"] in ("drop", "slope") and recorder.traj["t"]:
        result["pr2_criteria"] = pr2.evaluate(recorder.traj, float(config.get("floor_level", 0.0)))
    return result
