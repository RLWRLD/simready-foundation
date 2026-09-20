# SPDX-License-Identifier: Apache-2.0
"""Where a simulated body is, under either engine.

PhysX writes simulated poses back to USD; Newton writes them to Fabric only, so under Newton the
USD pose stays where the body started (measured 2026-09-18: in a 0.3 m drop USD stayed at 0.3374 m
while Fabric went to 0.0025 m). Body poses are read from USD under PhysX, as NVIDIA's engine-kit
does, and from Fabric under Newton, the way isaacsim.core.experimental.prims.XformPrim's fabric
backend reads them. While the timeline is stopped the authored USD pose is the current one.
Bounds are computed the same way for both engines: each rigid body's own geometry, bounded in the
body's frame from USD, placed with the body's current world matrix.
"""


def active_engine() -> str:
    """Lower-case name of the physics simulation that is running ("physx", "newton")."""
    import omni.physics.core

    physics = omni.physics.core.get_physics_interface()
    names = [str(physics.get_simulation_name(i)).lower() for i in physics.get_simulation_ids() if physics.is_simulation_active(i)]
    if len(names) != 1:
        raise RuntimeError(f"expected exactly one active physics simulation, found {names}")
    return names[0]


def rigid_bodies(stage, root_path):
    """Prims under root_path that simulate as rigid bodies (RigidBodyAPI, not disabled)."""
    from pxr import Usd, UsdPhysics

    bodies = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            if enabled is None or enabled:
                bodies.append(prim)
    return bodies


def _usdrt_stage_id(stage) -> int:
    from pxr import UsdUtils

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    if stage_id >= 0:
        return stage_id
    import omni.usd

    ctx = omni.usd.get_context()
    if ctx.get_stage() is not None and ctx.get_stage().GetRootLayer().identifier == stage.GetRootLayer().identifier:
        return ctx.get_stage_id()
    raise LookupError("stage is neither in the USD stage cache nor the active omni.usd context stage")


def _fabric_matrices(stage, paths):
    """Fabric world matrices, read with XformPrim's fabric-backend steps: get each prim on the Fabric
    stage, create its Fabric hierarchy local/world matrix attributes if missing, update world
    transforms, read. get_world_xform alone returned identity for a body Newton had not written yet
    (the first read after play(), measured)."""
    import usdrt

    rt_stage = usdrt.Usd.Stage.Attach(_usdrt_stage_id(stage))
    for path in paths:
        prim = rt_stage.GetPrimAtPath(str(path))
        if not prim or not prim.IsValid():
            raise LookupError(f"{path} is not on the Fabric stage")
        xformable = usdrt.Rt.Xformable(prim)
        if not xformable.GetFabricHierarchyLocalMatrixAttr():
            xformable.CreateFabricHierarchyLocalMatrixAttr()
        if not xformable.GetFabricHierarchyWorldMatrixAttr():
            xformable.CreateFabricHierarchyWorldMatrixAttr()
    hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId())
    hierarchy.update_world_xforms()
    out = {}
    for path in paths:
        m = hierarchy.get_world_xform(usdrt.Sdf.Path(str(path)))
        out[str(path)] = [[float(m.GetRow(r)[c]) for c in range(4)] for r in range(4)]
    return out


def _usd_matrices(stage, paths):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for path in paths:
        m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(str(path)))
        out[str(path)] = [[float(m[r][c]) for c in range(4)] for r in range(4)]
    return out


def body_matrices(stage, paths, engine):
    """({path: world matrix rows}, source) for the given rigid bodies under the given engine.

    A pose that is not finite means the solver diverged. That raises here, the one place every
    reading goes through; left alone, NaN drops out of a Gf.Range3d union silently and reappears
    later as a misleading "no bounded geometry" or a zero speed."""
    import math

    import omni.timeline

    if engine == "physx" or omni.timeline.get_timeline_interface().is_stopped():
        mats, source = _usd_matrices(stage, paths), "usd"
    elif engine == "newton":
        mats, source = _fabric_matrices(stage, paths), "fabric"
    else:
        raise ValueError(f"unknown physics engine {engine!r}")
    for path, rows in mats.items():
        if not all(math.isfinite(v) for row in rows for v in row):
            raise RuntimeError(f"non-finite pose for {path} (read from {source}): physics diverged")
    return mats, source


def world_bound(stage, root_path, matrices, purposes=("default", "render", "proxy")):
    """World-aligned (min, max) of the geometry under root_path; geometry owned by a rigid body in
    `matrices` is bounded in that body's frame and placed with its matrix, other geometry is static."""
    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    owners = set(matrices)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), list(purposes))
    world = Gf.Range3d()
    local = {path: Gf.Range3d() for path in owners}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        owner = prim
        while owner.IsValid() and str(owner.GetPath()) not in owners and owner != root:
            owner = owner.GetParent()
        if str(owner.GetPath()) in owners:
            own = cache.ComputeUntransformedBound(prim) if owner == prim else cache.ComputeRelativeBound(prim, owner)
            local[str(owner.GetPath())].UnionWith(own.ComputeAlignedRange())
        else:
            world.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedRange())
    for path, box in local.items():
        if box.IsEmpty():
            continue
        rows = matrices[path]
        lo, hi = box.GetMin(), box.GetMax()
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    world.UnionWith(Gf.Vec3d(*(x * rows[0][i] + y * rows[1][i] + z * rows[2][i] + rows[3][i] for i in range(3))))
    if world.IsEmpty():
        raise LookupError(f"no bounded geometry under {root_path} for purposes {list(purposes)}")
    lo, hi = world.GetMin(), world.GetMax()
    return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])


def rotation_angle(rows_a, rows_b) -> float:
    """Angle in radians between the rotations of two world matrices (scale removed)."""
    from pxr import Gf

    qa = Gf.Matrix4d(*[v for row in rows_a for v in row]).RemoveScaleShear().ExtractRotationQuat()
    qb = Gf.Matrix4d(*[v for row in rows_b for v in row]).RemoveScaleShear().ExtractRotationQuat()
    dot = abs(qa.GetReal() * qb.GetReal() + Gf.Dot(qa.GetImaginary(), qb.GetImaginary()))
    import math

    return 2.0 * math.acos(min(1.0, dot))


def collider_points(stage, bodies):
    """{body: Nx3 points in the body's frame} of the collision geometry each rigid body owns: mesh
    vertices of mesh colliders, bounding-box corners of other collider shapes."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    owners = set(bodies)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for body in bodies:
        body_prim = stage.GetPrimAtPath(body)
        chunks = []
        for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
            if prim != body_prim and str(prim.GetPath()) in owners:
                continue  # a nested body owns its own geometry
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Gprim):
                continue
            if prim.IsA(UsdGeom.Mesh):
                pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
            else:
                ext = UsdGeom.Boundable(prim).ComputeExtent(Usd.TimeCode.Default())
                pts = np.array([[x, y, z] for x in (ext[0][0], ext[1][0]) for y in (ext[0][1], ext[1][1]) for z in (ext[0][2], ext[1][2])], dtype=float)
            if not len(pts):
                continue
            rel, _ = cache.ComputeRelativeTransform(prim, body_prim)
            m = np.array([[rel[r][c] for c in range(4)] for r in range(4)])
            chunks.append(pts @ m[:3, :3] + m[3, :3])
        if chunks:
            out[body] = np.vstack(chunks)
    if not out:
        raise LookupError(f"no collision geometry under the rigid bodies {bodies}")
    return out


def lowest_point(points, matrices):
    """Lowest world z of the collider points placed with the bodies' current matrices."""
    import numpy as np

    lows = []
    for body, pts in points.items():
        m = np.asarray(matrices[body])
        lows.append(float((pts @ m[:3, :3] + m[3, :3])[:, 2].min()))
    return min(lows)


LEGACY_MASS_ATTR = "pxr:usd:physics_mass"  # an older encoding UsdPhysics.MassAPI does not resolve


def authored_mass(stage, root_path):
    """(canonical mass, mass in the older namespace). The first is UsdPhysics.MassAPI on each rigid
    body, else on the colliders under it. The second is `pxr:usd:physics_mass`, which some exporters
    write as well or instead -- NVIDIA's lamp writes only that. Which one an engine ends up using is
    not assumed here: `stack_notes` reports both against the mass the engine actually simulated."""
    from pxr import Usd, UsdPhysics

    read = legacy = 0.0
    for body in rigid_bodies(stage, root_path):
        for prim in Usd.PrimRange(body):
            attr = prim.GetAttribute(LEGACY_MASS_ATTR)
            if attr and attr.HasAuthoredValue():
                legacy += float(attr.Get() or 0.0)
        own = UsdPhysics.MassAPI(body).GetMassAttr() if body.HasAPI(UsdPhysics.MassAPI) else None
        if own is not None and own.HasAuthoredValue():
            read += float(own.Get() or 0.0)
            continue
        for prim in Usd.PrimRange(body):
            attr = UsdPhysics.MassAPI(prim).GetMassAttr() if prim.HasAPI(UsdPhysics.MassAPI) else None
            if attr is not None and attr.HasAuthoredValue():
                read += float(attr.Get() or 0.0)
    return read, legacy


def simulated_mass(stage, root_path):
    """The mass Newton's model ended up with for the asset's bodies, or None outside Newton."""
    import numpy as np

    import isaacsim.physics.newton as isaac_newton

    model = getattr(isaac_newton.acquire_stage(), "model", None)
    if model is None:
        return None
    labels = list(getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])
    mass = np.asarray(model.body_mass.numpy())
    return float(sum(m for label, m in zip(labels, mass) if str(label).startswith(root_path)))


def stack_notes(stage, engine, root_path):
    """What this engine did to the asset that its USD did not ask for. These do not change a verdict
    -- they change how one should be read -- so they travel with the result instead of being left in
    a log for someone to find: both were first noticed only by reading dumps by hand."""
    from pxr import Usd, UsdPhysics

    notes = []
    if engine != "newton":
        return notes
    joints = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies())
              if p.IsA(UsdPhysics.Joint)]
    ARMATURE = ("newton:armature", "mjc:armature", "physxJoint:armature")
    unauthored = [str(j.GetPath()) for j in joints
                  if not any(j.GetAttribute(n).HasAuthoredValue() for n in ARMATURE)]
    if unauthored:
        notes.append({"note": "joint_armature_default",
                      "detail": f"{len(unauthored)} of the asset's {len(joints)} joints author no armature, so Isaac's "
                                f"Newton stage gives each its default (cfg.armature, 0.1 kg m^2); PhysX uses 0. On light "
                                f"links this can hold a joint still that PhysX lets move.",
                      "joints": unauthored[:8]})
    canonical, legacy = authored_mass(stage, root_path)
    simulated = simulated_mass(stage, root_path)
    if simulated is not None and not any(m > 0 and abs(simulated - m) <= 0.01 * m for m in (canonical, legacy)):
        declared = ", ".join(filter(None, [f"{canonical:.4f} kg as UsdPhysics.MassAPI" if canonical > 0 else "",
                                           f"{legacy:.4f} kg as {LEGACY_MASS_ATTR}" if legacy > 0 else ""])) or "no mass"
        notes.append({"note": "mass_differs_from_usd",
                      "detail": f"the asset declares {declared} and this Newton simulated {simulated:.4f} kg. "
                                f"Newton 1.2.1 reads a collider's MassAPI only when its rigid body has one too, and "
                                f"otherwise recomputes the mass from density.",
                      "authored_kg": round(canonical, 5), "legacy_kg": round(legacy, 5),
                      "simulated_kg": round(simulated, 5),
                      "ratio_to_authored": round(simulated / canonical, 3) if canonical > 0 else None})
    return notes
