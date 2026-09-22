"""Build a declared deformable when this Newton's USD importer will not.

Newton 1.2.1 simulates cloth perfectly well -- it ships eight cloth examples, and they open a USD
stage and hand the mesh's points and indices to `ModelBuilder.add_cloth_mesh`. What it has no path
for is the *schema*: `PhysicsSurfaceDeformableSimAPI` is unknown to its importer, so a garment
authored the proposal's way imports as nothing at all. That is a gap in one importer, not a limit
of the engine, and the fix is the one the engine's own examples use.

So: read what the asset declares, convert it exactly the way Newton 1.5.0's importer converts it
(`import_usd_deformable_cloth.py`, quoted below), and call the same constructor. Both versions
then receive the same numbers from the same USD, which is the only way a comparison between them
means anything.

This runs only when the importer produced nothing. Where an importer works, it wins -- it also
resolves transforms, mass models and collision gating that are not re-derived here.
"""
import numpy as np
from pxr import Usd, UsdGeom

SURFACE_SIM = "PhysicsSurfaceDeformableSimAPI"
VOLUME_SIM = "PhysicsVolumeDeformableSimAPI"
SURFACE_MATERIAL = "PhysicsSurfaceDeformableMaterialAPI"
VOLUME_MATERIAL = "PhysicsVolumeDeformableMaterialAPI"
# Newton 1.5.0's own fallback when volumetric values are authored without a thickness.
DEFAULT_CLOTH_THICKNESS = 0.001


def _schemas(prim):
    listop = prim.GetMetadata("apiSchemas")
    return list(listop.GetAddedOrExplicitItems()) if listop else list(prim.GetAppliedSchemas())


def _value(prim, name):
    attr = prim.GetAttribute(name)
    return float(attr.Get()) if attr and attr.HasAuthoredValue() else None


def _bound_material(stage, prim, wanted):
    """The material carrying `wanted`, preferring one this prim binds."""
    from pxr import UsdShade

    binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics").GetMaterial()
    if binding and wanted in _schemas(binding.GetPrim()):
        return binding.GetPrim()
    return next((p for p in stage.Traverse() if wanted in _schemas(p)), None)


def is_simulated(prim):
    """Does this prim carry geometry a solver simulates?

    Answering by schema rather than by prim path is what lets one file's geometry be recognised
    inside another's reference -- the same asset is /World/banana in the original and /Asset/Body
    in the PhysX copy, and a name carried across files matches neither.
    """
    schemas = _schemas(prim)
    return (SURFACE_SIM in schemas or VOLUME_SIM in schemas
            or any(s.startswith("OmniPhysics") and "DeformableSimAPI" in s for s in schemas)
            or prim.IsA(UsdGeom.TetMesh))


def find(stage):
    """The prims this asset declares as deformable, and which kind each is.

    Instance proxies are traversed: `Stage.Traverse()` skips them, and a mesh brought in through
    an instanced reference is exactly the mesh a packaged asset has.
    """
    found = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        schemas = _schemas(prim)
        if SURFACE_SIM in schemas:
            found.append(("surface", prim))
        elif VOLUME_SIM in schemas or prim.IsA(UsdGeom.TetMesh):
            found.append(("volume", prim))
    return found


def _points(prim):
    value = UsdGeom.PointBased(prim).GetPointsAttr().Get()
    return 0 if value is None else len(value)


def why_not_one_body(stage, asset_name=""):
    """-> the reason this asset cannot be run as it is, or None.

    An asset may declare several simulated meshes -- a loaded polybag declares its film as a
    surface and its contents as a volume, and they are one object that has to be solved together
    and collide with each other. Both runners drive one mesh and a recording draws one, so taking
    the first would simulate the film alone and report it under the loaded bag's name.

    Separated from the raising so that `check` can refuse before it launches anything, and the
    runners can refuse if they are called directly: one rule, asked in two places.
    """
    name = asset_name or stage.GetRootLayer().identifier
    found = find(stage)
    if not found:
        return f"{name} declares no simulated mesh; there is nothing here to deform"
    if len(found) > 1:
        listed = "; ".join(f"{prim.GetPath()} ({kind}, {_points(prim)} points)"
                           for kind, prim in found)
        return (f"{name} declares {len(found)} simulated meshes and this pipeline drives one: "
                f"{listed}. They are one object, so running it would simulate the first and report "
                f"it under the whole asset's name. Two coupled bodies is the feature this asset "
                f"needs; it is refused rather than answered wrongly")
    return None


def one_body(stage, asset_name=""):
    """The single mesh this asset asks to be simulated; -> (kind, prim), or SystemExit saying why
    not."""
    reason = why_not_one_body(stage, asset_name)
    if reason:
        raise SystemExit(reason)
    return find(stage)[0]


def add_surface(builder, stage, prim, report=None):
    """Add a declared surface deformable to `builder`, the way Newton 1.5.0 would.

    The conversion is quoted from `newton/_src/utils/import_usd_deformable_cloth.py`:

        tri_ke  = stretchStiffness * thickness     (membrane stiffness ~ E*h)
        edge_ke = bendStiffness * thickness**3     (bending ~ E*h^3)
        tri_ka  = 0                                (the proposal authors no Poisson term)
        density = volumetric density * thickness   (Newton's cloth density is areal)
        particle_radius = 0.5 * thickness          (the shell's physical half-thickness)

    `shearStiffness` is deliberately dropped: Newton's isotropic membrane makes stretch and shear
    share one modulus, and 1.5.0 warns rather than folding it in. Dropping it here too is what
    keeps the two versions comparable.
    """
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if points.size == 0 or counts.size == 0:
        raise SystemExit(f"{prim.GetPath()}: declares {SURFACE_SIM} but carries no mesh")

    # Fan-triangulate whatever polygons the asset authored; add_cloth_mesh takes triangles.
    triangles, at = [], 0
    for count in counts:
        face = indices[at:at + count]
        triangles.extend([face[0], face[i], face[i + 1]] for i in range(1, count - 1))
        at += count
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1)

    material = _bound_material(stage, prim, SURFACE_MATERIAL)
    if material is None:
        raise SystemExit(f"{prim.GetPath()}: declares {SURFACE_SIM} but binds no {SURFACE_MATERIAL}; "
                         f"refusing to invent the stiffness of a cloth")
    thickness = _value(material, "physics:thickness")
    stretch = _value(material, "physics:stretchStiffness")
    bend = _value(material, "physics:bendStiffness")
    shear = _value(material, "physics:shearStiffness")
    density = _value(material, "physics:density")
    if thickness is None:
        thickness = DEFAULT_CLOTH_THICKNESS
        if report:
            report["thickness"] = (thickness, f"{material.GetPath()} authors none; Newton 1.5.0's "
                                              f"own default for volumetric values without one")
    tri_ke = stretch * thickness if stretch is not None else None
    edge_ke = bend * thickness ** 3 if bend is not None else None
    areal_density = density * thickness if density is not None else None
    radius = 0.5 * thickness
    if shear is not None and report is not None:
        report["shearStiffness"] = (shear, "dropped: Newton's isotropic membrane shares one "
                                           "modulus between stretch and shear, as 1.5.0 warns")

    import warp as wp
    builder.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0,
                           vel=wp.vec3(0.0, 0.0, 0.0), vertices=points.tolist(),
                           indices=triangles.tolist(), density=areal_density,
                           tri_ke=tri_ke, tri_ka=0.0, edge_ke=edge_ke, particle_radius=radius)
    return {"kind": "surface", "prim": str(prim.GetPath()), "thickness": thickness,
            "tri_ke": tri_ke, "edge_ke": edge_ke, "density": areal_density, "particle_radius": radius}


def add_missing(builder, asset, report=None):
    """Whatever the importer left out, built the way the engine's own examples build it.

    Returns a list of what was added; empty means the importer had already covered everything.
    """
    if builder.particle_count:
        return []
    stage = Usd.Stage.Open(str(asset))
    added = []
    for kind, prim in find(stage):
        if kind == "surface":
            added.append(add_surface(builder, stage, prim, report))
    return added
