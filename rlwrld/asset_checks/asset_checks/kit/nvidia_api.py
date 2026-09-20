# SPDX-License-Identifier: Apache-2.0
"""What NVIDIA's Foundation 7.1 tests need from engine-kit that the public 2026.6.5 does not give them.

The 7.1 tests do `from simready_benchmark_engine_kit.physics_utils import active_physics_engine` and
`from simready_benchmark_engine_kit import fabric_utils` inside their functions, under PhysX as well
as Newton; the release they were written against (>= 2026.6.6) is unpublished. Each name is supplied
only if engine-kit does not provide it, only in this Kit process, and `install()` reports what it
supplied so result.json records it.

`install()` also makes one existing name engine-aware. `physics_utils.check_physics_cookable` is, by
its own docstring, a crash guard for PhysX cooking ("colliders [that] would crash PhysX during
cooking"): it refuses a dynamic collider whose approximation PhysX cannot cook, such as `none`.
Newton does not cook with PhysX and has no such restriction, yet the drop, slope and grasp tests run
that guard under every engine, so three of NVIDIA's own sample props (toaster, dishwand, toolbox)
skip on Newton instead of being tested. Under Newton the guard is replaced with one that keeps
NVIDIA's other half, `check_has_colliders`, and drops the PhysX cooking half; the replacement records
every skip it withholds, and the run's result.json says the swap happened.
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


WITHHELD_COOK_SKIPS = []  # what the PhysX cooking guard would have skipped under Newton


def _engine_aware_prechecks(physics_utils) -> dict:
    """Replace the PhysX cooking guard with one that does nothing under a non-PhysX engine."""
    cookable, ready = physics_utils.check_physics_cookable, physics_utils.check_physics_ready

    def check_physics_cookable(stage, asset_root_path):
        message = cookable(stage, asset_root_path)
        if message is None or active_physics_engine() == "physx":
            return message
        WITHHELD_COOK_SKIPS.append({"asset_root": asset_root_path, "physx_would_skip": message})
        return None

    def check_physics_ready(stage, asset_root_path):
        message = check_physics_cookable(stage, asset_root_path)
        return message if message is not None else physics_utils.check_has_colliders(stage, asset_root_path)

    physics_utils.check_physics_cookable = check_physics_cookable
    physics_utils.check_physics_ready = check_physics_ready
    return {"physics_utils.check_physics_cookable": "asset_checks.kit.nvidia_api (PhysX-only guard)",
            "physics_utils.check_physics_ready": "asset_checks.kit.nvidia_api (PhysX-only guard)"}


def install(engine: str) -> dict:
    """Set the run's engine, supply whichever names engine-kit lacks, and make the PhysX cooking
    guard engine-aware. Returns what was supplied or replaced."""
    global _ENGINE
    import importlib.util

    from simready_benchmark_engine_kit import physics_utils

    _ENGINE = engine
    supplied = {}
    if engine != "physx":
        supplied.update(_engine_aware_prechecks(physics_utils))
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
