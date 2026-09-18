# SPDX-License-Identifier: Apache-2.0
"""Two engine-kit names NVIDIA's Foundation 7.1 tests import that the public engine-kit 2026.6.5 lacks.

The 7.1 tests do `from simready_benchmark_engine_kit.physics_utils import active_physics_engine` and
`from simready_benchmark_engine_kit import fabric_utils` inside their functions, under PhysX as well
as Newton; the release they were written against (>= 2026.6.6) is unpublished. Each name is supplied
only if engine-kit does not provide it, only in this Kit process, and `install()` reports what it
supplied so result.json records it.
"""
import sys
import types

_ENGINE = None


def active_physics_engine() -> str:
    """The engine this run was launched with (request.json). The first physics step checks it
    against the simulation that actually runs (nvidia_test.Proxy.physics_step)."""
    if _ENGINE is None:
        raise RuntimeError("asset_checks: engine not set before a test asked for it")
    return _ENGINE


def _fabric_utils():
    from asset_checks.kit import reading

    module = types.ModuleType("simready_benchmark_engine_kit.fabric_utils")
    module.__doc__ = "Supplied by asset_checks (engine-kit 2026.6.5 has no fabric_utils)."

    def _stopped():
        import omni.timeline

        return omni.timeline.get_timeline_interface().is_stopped()

    def live_world_matrix(stage, prim_path):
        # None makes the 7.1 callers read the authored USD pose, which is the current one when stopped
        return None if _stopped() else reading._fabric_matrices(stage, [str(prim_path)])[str(prim_path)]

    def is_rigid_body(prim):
        from pxr import UsdPhysics

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return False
        enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        return enabled is None or bool(enabled)

    def live_world_aabb(stage, prim_path, predicate=is_rigid_body):
        from pxr import Usd

        if _stopped():
            return None
        bodies = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(str(prim_path)), Usd.TraverseInstanceProxies()) if predicate(p)]
        if not bodies:
            return None
        return reading.world_bound(stage, str(prim_path), reading._fabric_matrices(stage, bodies), purposes=("default",))

    module.live_world_matrix, module.live_world_aabb, module.is_rigid_body = live_world_matrix, live_world_aabb, is_rigid_body
    return module


def install(engine: str) -> dict:
    """Set the run's engine; supply whichever of the two names engine-kit lacks. Returns what was supplied."""
    global _ENGINE
    import importlib.util

    from simready_benchmark_engine_kit import physics_utils

    _ENGINE = engine
    supplied = {}
    if not hasattr(physics_utils, "active_physics_engine"):
        physics_utils.active_physics_engine = active_physics_engine
        supplied["physics_utils.active_physics_engine"] = "asset_checks.kit.nvidia_api"
    name = "simready_benchmark_engine_kit.fabric_utils"
    if importlib.util.find_spec(name) is None:
        module = _fabric_utils()
        sys.modules[name] = module
        import simready_benchmark_engine_kit

        simready_benchmark_engine_kit.fabric_utils = module
        supplied["fabric_utils"] = "asset_checks.kit.nvidia_api"
    return supplied
