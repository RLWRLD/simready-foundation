# SPDX-License-Identifier: Apache-2.0
"""The environments, where simready-bench is, and the Kit launch settings. The single source for all three."""

import os
import pathlib
import subprocess
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

def engine_name(env):
    """What a person types for this environment's engine: `physx`, `newton1.2`, `newton1.5`.

    The PhysX on the newest Isaac is plainly `physx`; a PhysX on an older Isaac carries its
    Isaac version so it is never mistaken for it. That is the rule, rather than the version
    literal that was written in two files.
    """
    if env.engine == "physx":
        newest = max((e.isaac for e in ENVIRONMENTS.values() if e.engine == "physx"),
                     key=lambda v: tuple(int(x) for x in v.split(".")))
        return "physx" if env.isaac == newest else f"physx{env.isaac}"
    return "newton" + env.newton.rstrip(".")


DEFAULT_ENVIRONMENTS = ("physx", "newton12", "newton15")  # the rigid comparison; the rest are opt-in

# The one environment deformable assets are checked in (by `native/`, never through Kit). Newton 1.5
# VBD only, by the user's scope decision of 2026-09-29: Newton 1.2, XPBD and PhysX deformables were
# removed. `check.py` refuses a deformable anywhere else.
DEFORMABLE = "newton15_vbd"

BENCH_VARIABLE = "SIMREADY_BENCH"


def bench(given=None):
    """simready-bench -- the venvs, isaac-run and the GPU pin: `given`, else $SIMREADY_BENCH. There
    is no default: a path written into the code is somebody else's machine."""
    where = given or os.environ.get(BENCH_VARIABLE)
    if not where:
        raise SystemExit(f"where is simready-bench? pass --bench or set {BENCH_VARIABLE}")
    path = pathlib.Path(where).resolve()
    if not (path / "isaac-run").exists():
        raise SystemExit(f"{path} is not simready-bench: it has no isaac-run")
    return path


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]


def code_version():
    """The commit this package runs from, its branch, and whether its tree has uncommitted changes --
    written into every result, so a number can always be traced to the code that made it."""
    def git(*args):
        return subprocess.run(["git", "-C", str(PACKAGE_ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "."))}


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
