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


def raw_api_schemas(prim):
    """The prim's applied API schemas as authored. Usd.Prim.GetAppliedSchemas() drops names this USD
    build does not register, and the deformable proposal's public names are registered in neither
    Isaac venv, so the authored list is the only place they appear -- which is what Newton's own
    importer falls back to reading."""
    listop = prim.GetMetadata("apiSchemas")
    return list(listop.GetAddedOrExplicitItems()) if listop else []


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


# The shape we draw where the asset ships no collision geometry of its own and declares no
# approximation. It is ours, not the asset's, so it is one shape used for every engine of that
# asset: the picture then differs only by what the physics did, which is the whole point. A single
# convex hull is the one thing every engine here can collide with -- PhysX refuses a triangle mesh
# on a dynamic body and MuJoCo convexifies -- and it is the hull of the collider's own points, not
# a re-cooking of any engine's (PhysX simplifies its to `physxConvexHullCollision:hullVertexLimit`,
# 64 by default; ours is the exact hull).
OURS = "convexHull"


def collider_shape_choice(prim):
    """-> (approximation, whose, why) for one collider.

    `approximation` is UsdPhysics' name for the shape to draw, where "none" means this prim's own
    geometry. `whose` is "asset" when the USD says it and "ours" when the USD is silent.

    OpenUSD states the convention for collision geometry that is not the render mesh: "Collision
    meshes may be specified explicitly ... by adding the custom collider mesh as a sibling to the
    original graphics mesh, UsdGeomImageable purpose to 'guide' so it does not render." So a
    collider the viewer cannot see is one the asset authored on purpose, and it is drawn as it is.
    """
    from pxr import UsdGeom, UsdPhysics

    purpose = UsdGeom.Imageable(prim).ComputePurpose()
    if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
        return "none", "asset", f"a collider authored beside the render mesh, purpose {purpose}"
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        declared = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none")
        return declared, "asset", "UsdPhysicsMeshCollisionAPI declares it"
    if "PhysxSDFMeshCollisionAPI" in prim.GetAppliedSchemas():
        return "sdf", "asset", "PhysxSDFMeshCollisionAPI is applied"
    orphan = prim.GetAttribute("physics:approximation")
    if orphan and orphan.HasAuthoredValue():
        return OURS, "ours", (f"physics:approximation={orphan.Get()!r} is authored without "
                              f"UsdPhysicsMeshCollisionAPI, so it is not the schema's and engines "
                              f"disagree about it")
    return OURS, "ours", "the asset ships no collision mesh of its own and declares no approximation"


def collider_shape(points, approximation):
    """The points of the shape the engine collides with, and its triangles.

    `convexHull` is the hull of the collider's own points, and the bounding shapes are built from
    the same points. `none` is the prim's own geometry -- what the asset means when it says so, and
    what a collider authored beside the render mesh already is. A decomposition or an SDF is more
    than a copy of geometry can say, so there the caller is told (None) to keep the mesh and print
    the approximation's name beside it.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if approximation in ("none", "convexDecomposition", "sdf", "meshSimplification"):
        return None
    if approximation == "boundingCube":
        lo, hi = points.min(axis=0), points.max(axis=0)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        return corners, [list(f) for f in faces]
    if approximation == "boundingSphere":
        centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
        radius = float(np.linalg.norm(points - centre, axis=1).max())
        return _sphere(centre, radius)
    if approximation == "convexHull":
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        used = sorted(set(int(i) for face in hull.simplices for i in face))
        index = {old: new for new, old in enumerate(used)}
        faces = []
        # `simplices` are not consistently wound, and a mesh whose triangles disagree has normals
        # pointing both ways: it renders inside out in patches. `equations` carries each facet's
        # outward normal, so each triangle is turned to agree with it.
        for simplex, equation in zip(hull.simplices, hull.equations):
            a, b, c = (points[int(i)] for i in simplex)
            face = [index[int(i)] for i in simplex]
            if float(np.dot(np.cross(b - a, c - a), equation[:3])) < 0.0:
                face.reverse()
            faces.append(face)
        return points[used], faces
    raise ValueError(f"unknown collision approximation {approximation!r}")


def _sphere(centre, radius, rings=16, segments=24):
    import numpy as np

    lat = np.linspace(0.0, np.pi, rings + 1)
    lon = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    points = [centre + radius * np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
              for a in lat for b in lon]
    faces = []
    for i in range(rings):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            faces.append([a, b, b + segments, a + segments])
    return np.asarray(points), faces


def fixture_gprims(stage, asset_root, scenery):
    """The visible geometry in the scene that is neither the asset nor scenery: the slope, the
    gripper's pads, whatever a test builds around the asset. Each is a leaf, so it can be copied
    and placed by its own world matrix without any hierarchy to reproduce."""
    from pxr import Usd, UsdGeom

    out = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        path = str(prim.GetPath())
        if path == asset_root or path.startswith(asset_root + "/"):
            continue
        if any(path == s or path.startswith(s + "/") for s in scenery):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.inherited:
            continue
        out.append(path)
    return out


def matrices(stage, paths, engine):
    """World matrices for any prims. A prim that is, or sits under, a rigid body is placed by that
    body's matrix read the way `body_matrices` reads it (Fabric under Newton) and its own USD offset
    from the body -- a collider does not move relative to its body, and only bodies are on the
    Fabric stage. Anything else is read from USD, where a static fixture sits."""
    from pxr import Usd, UsdGeom, UsdPhysics

    def owner(path):
        prim = stage.GetPrimAtPath(path)
        while prim and prim.IsValid() and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
            prim = prim.GetParent()
        return None

    owners = {p: owner(p) for p in paths}
    bodies = sorted({b for b in owners.values() if b})
    body_world = body_matrices(stage, bodies, engine)[0] if bodies else {}
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for path in paths:
        body = owners[path]
        if body is None:
            out.update(_usd_matrices(stage, [path]))
            continue
        rel, _ = cache.ComputeRelativeTransform(stage.GetPrimAtPath(path), stage.GetPrimAtPath(body))
        world = rel * type(rel)(*[v for row in body_world[body] for v in row])
        out[path] = [[float(world[r][c]) for c in range(4)] for r in range(4)]
    return out


# Attribute namespaces that place a prim or run physics on it. A copy gets neither: a recording
# places its copies itself, and physics never runs on a recording.
NOT_COPIED = ("xformOp", "physics", "physxCollision", "physxRigidBody", "physxConvexHullCollision",
              "physxConvexDecompositionCollision", "physxTriangleMeshCollision", "physxSDFMeshCollision")


def copy_gprim(source, stage, path):
    """A leaf copy of a gprim's composed attributes -- its type, points, size, colour -- with no
    transform and no physics on it. The one way geometry is copied out of a stage, used for a
    fixture the test built and for the colliders an asset declares."""
    from pxr import UsdGeom

    copy = stage.DefinePrim(path, source.GetTypeName())
    for attr in source.GetAttributes():
        if attr.GetNamespace() in NOT_COPIED or attr.GetName() == "xformOpOrder":
            continue
        value = attr.Get()
        if value is None:
            continue
        copy.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom()).Set(value)
    colour = bound_colour(source)
    if colour is not None:
        # A material is a binding, not an attribute, and a copy has none. The flat colour NVIDIA's
        # room gives a slope (`apply_flat_color`: a UsdPreviewSurface with a diffuseColor) is what
        # a viewer sees, so it travels with the copy as its displayColor.
        from pxr import Gf, Vt

        UsdGeom.Gprim(copy).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
    return copy


def bound_colour(prim):
    """The diffuse colour of the material bound to `prim`, or None when there is none or it is
    not a plain colour."""
    from pxr import UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return None
    for shader in (UsdShade.Shader(child) for child in material.GetPrim().GetChildren()):
        if not shader:
            continue
        colour = shader.GetInput("diffuseColor")
        if colour and colour.Get() is not None and not colour.GetConnectedSource():
            return tuple(float(v) for v in colour.Get())
    return None


def export_fixtures(stage, paths, target):
    """Copy the fixtures' geometry -- every authored attribute of each gprim, its type and its
    material-free colour -- into their own layer, one leaf each under /fixtures, so a recording can
    reference it and place each one by its recorded world matrix. Returns {live path: copy path}."""
    from pxr import Sdf, Usd, UsdGeom

    layer = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(layer, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(layer, UsdGeom.GetStageMetersPerUnit(stage))
    root = UsdGeom.Xform.Define(layer, "/fixtures")
    layer.SetDefaultPrim(root.GetPrim())
    where = {}
    for path in paths:
        source = stage.GetPrimAtPath(path)
        name = Sdf.Path(path).name
        copy_path = f"/fixtures/{name}"
        n = 1
        while layer.GetPrimAtPath(copy_path):
            n += 1
            copy_path = f"/fixtures/{name}_{n}"
        copy_gprim(source, layer, copy_path)
        where[path] = copy_path
    layer.GetRootLayer().Save()
    return where
