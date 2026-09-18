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
    """({path: world matrix rows}, source) for the given rigid bodies under the given engine."""
    import omni.timeline

    if engine == "physx" or omni.timeline.get_timeline_interface().is_stopped():
        return _usd_matrices(stage, paths), "usd"
    if engine == "newton":
        return _fabric_matrices(stage, paths), "fabric"
    raise ValueError(f"unknown physics engine {engine!r}")


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
