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


def dump(path):
    """Write what this run's Newton physics was built from to `path` (.npz), to diff two environments
    offline. Arrays: the compiled MuJoCo model on the CPU (`mj.*`) and the copy the GPU steps run on
    (`mjw.*`), with their option structs; the Newton model (`newton.*`), whose shapes feed Newton's own
    collision pipeline (Isaac runs MuJoCo with use_mujoco_contacts=False, so contacts come from
    there); the solver's and the collision pipeline's own arrays. `__meta__` (JSON): every scalar
    setting of the same objects, object names (MuJoCo ids and Newton shapes), Isaac's Newton config,
    and each attribute that could not be read, with the reason."""
    import dataclasses
    import json

    import mujoco
    import numpy as np

    import isaacsim.physics.newton as isaac_newton

    stage = isaac_newton.acquire_stage()
    solver = stage.solver
    arrays, scalars, unread = {}, {}, {}

    def put(prefix, obj):
        for name in dir(obj):
            if name.startswith("__"):
                continue
            key = f"{prefix}.{name}"
            try:
                value = getattr(obj, name)
                if callable(value) and not hasattr(value, "shape"):
                    continue
                if isinstance(value, np.ndarray):
                    arrays[key] = value
                elif hasattr(value, "numpy") and hasattr(value, "shape") and hasattr(value, "dtype"):  # warp array
                    arrays[key] = np.asarray(value.numpy())
                elif isinstance(value, (bool, int, float, str)) or value is None:
                    scalars[key] = value
            except Exception as exc:  # noqa: BLE001 - recorded in __meta__.unread, the dump is diagnostic
                unread[key] = repr(exc)[:160]

    mj, mjw = solver.mj_model, solver.mjw_model
    for prefix, obj in (("mj", mj), ("mj.opt", mj.opt), ("mjw", mjw), ("mjw.opt", mjw.opt),
                        ("newton", stage.model), ("solver", solver),
                        ("pipeline", getattr(stage, "collision_pipeline", None))):
        if obj is not None:
            put(prefix, obj)
    names = {}
    for kind, count in (("BODY", mj.nbody), ("JOINT", mj.njnt), ("GEOM", mj.ngeom), ("SITE", mj.nsite),
                        ("MESH", mj.nmesh), ("ACTUATOR", mj.nu), ("EQUALITY", mj.neq), ("TENDON", mj.ntendon)):
        obj = getattr(mujoco.mjtObj, f"mjOBJ_{kind}")
        names[kind.lower()] = [mujoco.mj_id2name(mj, obj, i) or f"#{i}" for i in range(count)]
    labels = getattr(stage.model, "shape_label", None) or getattr(stage.model, "shape_key", None) or []
    names["newton_shape"] = [str(v) for v in labels]
    cfg = stage.cfg
    meta = {"scalars": scalars, "names": names, "unread": unread,
            "isaac_cfg": dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)}
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta, default=repr)), **arrays)
    return {"path": str(path), "arrays": len(arrays), "scalars": len(scalars), "unread": len(unread)}


class Trace:
    """The contacts the MuJoCo solver actually used, every `every` physics steps: how many join each
    gripper pad to the asset, their summed normal force, the deepest penetration, and how many hold
    the asset on the floor. Read from the GPU data the steps run on (Newton writes its own contacts
    there: use_mujoco_contacts=False)."""

    def __init__(self, every):
        self.every, self.steps, self.rows, self.groups, self.solver_id = int(every), 0, [], None, None

    def _group_geoms(self, mj):
        import mujoco

        from simready_benchmark_kit_suite.fet005_grasp.grasp_robot import GraspRobot

        from asset_checks.kit.scene import ASSET_PRIM

        prefixes = {"left": GraspRobot.LEFT_PAD + "/", "right": GraspRobot.RIGHT_PAD + "/", "asset": ASSET_PRIM + "/"}
        groups = {k: set() for k in prefixes}
        for i in range(mj.ngeom):
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            for k, prefix in prefixes.items():
                if name.startswith(prefix):
                    groups[k].add(i)
        empty = [k for k, v in groups.items() if not v]
        if empty:  # a trace of zeros would read as "no contact"; it must not pass
            raise RuntimeError(f"contact trace found no geoms for {empty} under {[prefixes[k] for k in empty]}")
        return groups

    def sample(self, t):
        import numpy as np

        import isaacsim.physics.newton as isaac_newton

        self.steps += 1
        if self.steps % self.every:
            return
        solver = isaac_newton.acquire_stage().solver
        mj, d = solver.mj_model, solver.mjw_data
        if self.solver_id != id(solver):  # a play after a rebuild compiles a new model: regroup its geoms
            self.groups, self.solver_id = self._group_geoms(mj), id(solver)
        n = int(d.nacon.numpy()[0])
        geom = d.contact.geom.numpy()[:n]
        dist = d.contact.dist.numpy()[:n]
        adr = d.contact.efc_address.numpy()[:n]
        force = d.efc.force.numpy()[0]
        elliptic = int(mj.opt.cone) == 1
        normal = np.array([force[a[0]] if elliptic else force[a[a >= 0]].sum() for a in adr]) if n else np.zeros(0)
        normal[adr[:, 0] < 0] = 0.0  # detected but not in the constraint set
        g = self.groups
        in_a = np.isin(geom, list(g["asset"]))
        row = {"t": round(t, 4), "nacon": n}
        for pad in ("left", "right"):
            m = np.isin(geom, list(g[pad])).any(axis=1) & in_a.any(axis=1)
            row[f"{pad}_n"] = int(m.sum())
            row[f"{pad}_fn"] = round(float(normal[m].sum()), 4)
            row[f"{pad}_depth_mm"] = round(float(-dist[m].min() * 1000.0), 3) if m.any() else None
        others = ~(np.isin(geom, list(g["left"] | g["right"])).any(axis=1)) & in_a.any(axis=1) & ~in_a.all(axis=1)
        row["asset_other_n"] = int(others.sum())
        row["asset_other_fn"] = round(float(normal[others].sum()), 4)
        self.rows.append(row)
