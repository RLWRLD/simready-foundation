# SPDX-License-Identifier: Apache-2.0
"""PR #2's MuJoCo contact settings for Newton, authored as USD attributes Newton's importer reads.

newton15_cert.py (branch rlwrld/newton-conformance) found that Newton's stock contact settings hold no
pinch grasp (0 of 171 props): contact stiffness scales with the effective mass of the pair, so light
pads pushed hard sink through the asset, and without torsional friction (condim 4) an off-centre
pinch spins out. It uses a 4 ms solref time constant on every shape, condim 4 and a 0.05 torsional
coefficient on the pads, an elliptic cone and impratio 10. Only these are applied; NVIDIA's gripper,
masses, gains and verdicts are unchanged.

Where Newton reads them (newton/_src/usd/schemas.py, solvers/mujoco/solver_mujoco.py, 1.2.1 and 1.5.0):
newton:contact_ke / newton:contact_kd on each collider, mjc:condim on each collider,
newton:torsionalFriction on the bound physics material, mjc:option:impratio / mjc:option:cone on
the PhysicsScene.
"""

PROFILES = {
    "stock": None,
    "pr2": {  # newton15_cert.py: CONTACT_TIMECONST, CONTACT_DAMPRATIO, --pad-condim, PAD_MU_TORSIONAL, --impratio, --cone
        "timeconst_s": 0.004,
        "dampratio": 1.0,
        "pad_condim": 4,
        "pad_torsional_friction": 0.05,
        "impratio": 10.0,
        "cone": "elliptic",
    },
}


def gains(timeconst, dampratio):
    """(ke, kd) that Newton's MuJoCo export turns into solref (timeconst, dampratio) (newton15_cert.contact_gains)."""
    kd = 2.0 / timeconst
    return (kd / (2.0 * dampratio)) ** 2, kd


def apply(stage, profile):
    """Author the profile on what is on the stage now; returns what was authored."""
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    from simready_benchmark_kit_suite.fet005_grasp.grasp_robot import GraspRobot

    ke, kd = gains(profile["timeconst_s"], profile["dampratio"])
    report = {"contact_ke": ke, "contact_kd": kd, "shapes": 0, "instance_proxies_skipped": 0, "pads": [], "pad_materials": [], "scenes": []}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if prim.IsInstanceProxy():
                report["instance_proxies_skipped"] += 1
                continue
            prim.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(ke)
            prim.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(kd)
            report["shapes"] += 1
        if prim.IsA(UsdPhysics.Scene):
            prim.CreateAttribute("mjc:option:impratio", Sdf.ValueTypeNames.Float).Set(profile["impratio"])
            prim.CreateAttribute("mjc:option:cone", Sdf.ValueTypeNames.Token).Set(profile["cone"])
            report["scenes"].append(str(prim.GetPath()))
    # Isaac Sim hands its own MuJoCo solver config to SolverMuJoCo. Where that config carries these
    # options (Isaac Sim 6.0.1: cone and impratio; 6.1.0 has neither) it overrides the USD values:
    # measured on 6.0.1, impratio stayed 1.0 with mjc:option:impratio = 10 alone. So they are set on
    # Isaac's config as well; measure() shows what the compiled model got.
    import isaacsim.physics.newton as isaac_newton

    solver_cfg = isaac_newton.acquire_stage().cfg.solver_cfg
    report["isaac_solver_cfg"] = {}
    for key in ("impratio", "cone"):
        if hasattr(solver_cfg, key):
            setattr(solver_cfg, key, profile[key])
            report["isaac_solver_cfg"][key] = profile[key]
    for pad_root in (GraspRobot.LEFT_PAD, GraspRobot.RIGHT_PAD):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(pad_root)) if stage.GetPrimAtPath(pad_root) else []:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(profile["pad_condim"])
            report["pads"].append(str(prim.GetPath()))
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if material:
                material.GetPrim().CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(profile["pad_torsional_friction"])
                report["pad_materials"].append(str(material.GetPath()))
    return report


def measure():
    """What MuJoCo actually got: the live solver's compiled model (not what was authored in USD)."""
    import numpy as np

    import isaacsim.physics.newton as isaac_newton

    solver = getattr(isaac_newton.acquire_stage(), "solver", None)
    out = {"solver": type(solver).__name__ if solver is not None else None}
    mj = getattr(solver, "mj_model", None)
    if mj is None:
        return out
    out["timestep_s"] = float(mj.opt.timestep)
    import mujoco

    # with refsafe on (MuJoCo's default) a solref time constant below 2 x timestep is raised to it
    out["refsafe"] = not bool(int(mj.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
    out["impratio"] = float(mj.opt.impratio)
    out["cone"] = {0: "pyramidal", 1: "elliptic"}.get(int(mj.opt.cone), int(mj.opt.cone))
    values, counts = np.unique(np.asarray(mj.geom_condim), return_counts=True)
    out["geom_condim"] = {int(v): int(c) for v, c in zip(values, counts)}
    values, counts = np.unique(np.asarray(mj.geom_solref).round(5), axis=0, return_counts=True)
    out["geom_solref"] = {f"{v[0]:g},{v[1]:g}": int(c) for v, c in zip(values, counts)}
    # Joint dynamics: Isaac's Newton stage gives every joint that authors no armature its own
    # default (cfg.armature), which on a light articulated prop outweighs the links' inertia.
    out["isaac_default_joint_armature"] = getattr(getattr(isaac_newton.acquire_stage(), "cfg", None), "armature", None)
    for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
        values, counts = np.unique(np.asarray(getattr(mj, field)).round(6), return_counts=True)
        out[field] = {f"{v:g}": int(c) for v, c in zip(values, counts)}
    mjw = getattr(solver, "mjw_model", None)
    try:  # the GPU copy the steps run on
        out["mjw_impratio"] = [round(float(x), 4) for x in np.asarray(mjw.opt.impratio.numpy()).ravel()[:4]]
        out["mjw_cone"] = int(np.asarray(mjw.opt.cone.numpy()).ravel()[0]) if hasattr(mjw.opt.cone, "numpy") else int(mjw.opt.cone)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal: the compiled model above is the main reading
        out["mjw_read_error"] = repr(exc)[:200]
    return out
