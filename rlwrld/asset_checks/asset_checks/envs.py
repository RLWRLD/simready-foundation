# SPDX-License-Identifier: Apache-2.0
"""The three environments and the Kit launch settings. The single source for both."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    """One thing an asset can be checked in: an Isaac Sim build, a physics engine, and a solver.

    The three are independent and a result only means something with all three named: Newton's
    version moves with Isaac's (isaacsim-core pins `newton[sim]==`), so a Newton 1.2-vs-1.5
    difference is an Isaac+Newton difference until an environment holds one of them still.
    """

    name: str  # label in results and reports
    venv: str  # simready-bench venv tag passed to isaac-run
    experience: str  # Kit experience the isaacsim launcher starts
    engine: str  # physics engine that experience simulates with: "physx" or "newton"
    newton: str  # Newton version prefix the venv must report ("" when the engine is not Newton)
    solver: str  # what integrates the scene: "physx", or a Newton solver ("mujoco", "vbd", "xpbd")
    note: str

    @property
    def isaac(self) -> str:
        return {"isaac601": "6.0.1", "isaac610": "6.1.0"}[self.venv]


# A Newton solver other than the default is selected by applying its scene API schema to the
# PhysicsScene (Isaac 6.1.0 `impl/utils.py newton_solver_to_api_schema`); Isaac 6.0.1 has no such
# mapping and its `_get_solver` accepts only mujoco and xpbd.
NEWTON_SOLVER_SCENE_API = {"mujoco": "MjcSceneAPI", "xpbd": "NewtonXpbdSceneAPI", "vbd": "NewtonVbdSceneAPI"}

ENVIRONMENTS = {
    env.name: env
    for env in (
        Environment("physx", "isaac610", "isaacsim.exp.full", "physx", "", "physx", "Isaac Sim 6.1.0, PhysX"),
        Environment("physx601", "isaac601", "isaacsim.exp.full", "physx", "", "physx", "Isaac Sim 6.0.1, PhysX"),
        Environment("newton12", "isaac601", "isaacsim.exp.full.newton", "newton", "1.2.", "mujoco", "Isaac Sim 6.0.1, Newton 1.2.1, MuJoCo"),
        Environment("newton15", "isaac610", "isaacsim.exp.full.newton", "newton", "1.5.", "mujoco", "Isaac Sim 6.1.0, Newton 1.5.0, MuJoCo"),
        Environment("newton12_xpbd", "isaac601", "isaacsim.exp.full.newton", "newton", "1.2.", "xpbd", "Isaac Sim 6.0.1, Newton 1.2.1, XPBD"),
        Environment("newton15_xpbd", "isaac610", "isaacsim.exp.full.newton", "newton", "1.5.", "xpbd", "Isaac Sim 6.1.0, Newton 1.5.0, XPBD"),
        Environment("newton12_vbd", "isaac601", "isaacsim.exp.full.newton", "newton", "1.2.", "vbd", "Isaac Sim 6.0.1, Newton 1.2.1, VBD"),
        Environment("newton15_vbd", "isaac610", "isaacsim.exp.full.newton", "newton", "1.5.", "vbd", "Isaac Sim 6.1.0, Newton 1.5.0, VBD"),
    )
}

DEFAULT_ENVIRONMENTS = ("physx", "newton12", "newton15")  # the rigid comparison; the rest are opt-in


# SIMREADY_PHYSICS_RUNTIME per engine: NVIDIA's grasp_and_lift reads it (default "PhysX") to pick the
# FET_003 feature the asset must have validated; the official runner is expected to set it.
PHYSICS_RUNTIME = {"physx": "PhysX", "newton": "Newton"}


def gpu_settings(gpu: int) -> dict:
    """Kit settings that keep Kit on the workspace GPU.

    isaac-run exposes only that GPU to CUDA, so PhysX's device 0 is it; the renderer enumerates
    GPUs through Vulkan and is pinned to the same physical index. With Kit's defaults on a two-GPU
    machine (multi-GPU rendering, PhysX device -1) Kit crashed whenever a stage that ran GPU
    physics was replaced (measured 2026-09-18 on Isaac Sim 6.0.1).
    """
    return {"/physics/cudaDevice": 0, "/renderer/multiGpu/enabled": False, "/renderer/activeGpu": gpu}


def kit_flags(gpu: int) -> list:
    """Command-line flags for every Kit launch: synchronous rendering so a captured frame is the fully
    rendered one (as NVIDIA's engine-kit launches Kit for captures), no ROS bridge, capture enabled."""
    flags = [
        "--no-window",
        "--/app/window/hideUi=1",
        "--/app/useFabricSceneDelegate=true",
        "--/app/asyncRendering=false",
        "--/app/asyncRenderingLowLatency=false",
        "--/exts/omni.kit.renderer.core/threads/syncMainToPresent=true",
        "--/app/runLoopsGlobal/syncToPresent=true",
        "--/isaac/startup/ros_bridge_extension=",
        "--enable",
        "omni.kit.renderer.capture",
    ]
    for key, value in gpu_settings(gpu).items():
        flags.append(f"--{key}={str(value).lower() if isinstance(value, bool) else value}")
    return flags
