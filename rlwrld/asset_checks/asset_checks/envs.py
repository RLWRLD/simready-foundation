# SPDX-License-Identifier: Apache-2.0
"""The three environments and the Kit launch settings. The single source for both."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    name: str  # label in results and reports
    venv: str  # simready-bench venv tag passed to isaac-run
    experience: str  # Kit experience the isaacsim launcher starts
    engine: str  # physics engine that experience simulates with
    newton: str  # Newton version prefix the venv must report ("" when the engine is not Newton)
    note: str


ENVIRONMENTS = {
    env.name: env
    for env in (
        Environment("physx", "isaac610", "isaacsim.exp.full", "physx", "", "Isaac Sim 6.1.0, PhysX"),
        Environment("newton12", "isaac601", "isaacsim.exp.full.newton", "newton", "1.2.", "Isaac Sim 6.0.1, Newton 1.2.1"),
        Environment("newton15", "isaac610", "isaacsim.exp.full.newton", "newton", "1.5.", "Isaac Sim 6.1.0, Newton 1.5.0"),
    )
}


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
