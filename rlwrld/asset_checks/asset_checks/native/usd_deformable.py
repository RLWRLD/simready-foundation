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


def simulated_kind(prim):
    """"surface", "volume", or None: what a solver would simulate this prim as.

    Answering by schema rather than by prim path is what lets one file's geometry be recognised
    inside another's reference -- the same asset is /World/banana in the original and /Asset/Body
    in the PhysX copy, and a name carried across files matches neither.

    The schema name is matched by its *ending*, because the same schema is spelled two ways: the
    Newton-flavour asset authors `PhysicsSurfaceDeformableSimAPI` and the PhysX conversion writes
    `OmniPhysicsSurfaceDeformableSimAPI`. Listing the spellings anyone has seen is a table of
    today's data -- measured, a version of this function that held two literal names recognised
    every volume (they are also typed `TetMesh`) and no surface at all, so the cloth and the
    polybag reported "nothing here to deform" on PhysX while the fruit ran. A suffix is the rule
    the renaming actually followed, and it will still hold for the next prefix.
    """
    schemas = _schemas(prim)
    if any(s.endswith(SURFACE_SIM) for s in schemas):
        return "surface"
    if any(s.endswith(VOLUME_SIM) for s in schemas) or prim.IsA(UsdGeom.TetMesh):
        return "volume"
    return None


def is_simulated(prim):
    """Does this prim carry geometry a solver simulates?"""
    return simulated_kind(prim) is not None


def find(stage):
    """The prims this asset declares as deformable, and which kind each is.

    Instance proxies are traversed: `Stage.Traverse()` skips them, and a mesh brought in through
    an instanced reference is exactly the mesh a packaged asset has.
    """
    found = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        kind = simulated_kind(prim)
        if kind:
            found.append((kind, prim))
    return found


def _points(prim):
    value = UsdGeom.PointBased(prim).GetPointsAttr().Get()
    return 0 if value is None else len(value)


def bodies(stage):
    """What this asset asks to be simulated, and what to draw for each: [(kind, sim, render)].

    `render` is the mesh the asset draws for that body, or None where it draws the simulated mesh
    itself (a cloth). It is looked for among the sim prim's *siblings*, because that is where
    UsdPhysics puts geometry that belongs to the same object: the OpenUSD docs describe separate
    collision geometry as "a sibling to the original graphics mesh". A loaded polybag is
    /Polybag/Film/{Simulation, ...} beside /Polybag/Contents/{Simulation, ...}, and the sibling
    rule keeps each film with its own film and each filling with its own filling -- where picking
    the largest drawable in the whole asset, which is what one body needed, would give both bodies
    the same mesh.
    """
    out = []
    for kind, sim in find(stage):
        parent = sim.GetParent()
        drawable = [q for q in Usd.PrimRange(parent, Usd.TraverseInstanceProxies())
                    if q != sim and q.IsA(UsdGeom.PointBased)
                    and simulated_kind(q) is None
                    and UsdGeom.PointBased(q).GetPointsAttr().Get()]
        render = max(drawable, key=lambda q: len(UsdGeom.PointBased(q).GetPointsAttr().Get()),
                     default=None)
        out.append((kind, sim, render))
    return out


def why_not_runnable(stage, asset_name="", most=None, needs=None):
    """-> the reason this asset cannot be run as it is, or None.

    `most` is how many simulated meshes the caller can drive. Newton builds them all into one
    model off one particle array, so its runners pass nothing and take whatever the asset has; the
    PhysX runners drive one body and say so. An asset with more meshes than the caller can drive
    is refused rather than run, because running it would simulate the first and report it under
    the whole asset's name -- an empty bag reported as a loaded one.

    `needs` is the set of body kinds the experiment means anything for (an experiment's `BODIES`).
    An asset with none of them is refused: pressing a surface, which has no thickness to give,
    would produce a verdict about nothing.

    Separated from the raising so that `check` can refuse before it launches anything, and the
    runners can refuse if they are called directly: one rule, asked in two places.
    """
    name = asset_name or stage.GetRootLayer().identifier
    found = find(stage)
    if not found:
        return f"{name} declares no simulated mesh; there is nothing here to deform"
    if most is not None and len(found) > most:
        listed = "; ".join(f"{prim.GetPath()} ({kind}, {_points(prim)} points)"
                           for kind, prim in found)
        return (f"{name} declares {len(found)} simulated meshes and this runner drives "
                f"{most}: {listed}. They are one object, so running it would simulate the first "
                f"and report it under the whole asset's name")
    if needs is not None and not any(kind in needs for kind, _ in found):
        declared = ", ".join(sorted({kind for kind, _ in found}))
        return (f"{name} declares a {declared} body only, and this experiment means something for "
                f"a {' or '.join(sorted(needs))} body; refused rather than answered about nothing")
    return None


def one_body(stage, asset_name=""):
    """The single mesh this asset asks to be simulated; -> (kind, prim), or SystemExit saying why
    not. For a caller that drives one body."""
    reason = why_not_runnable(stage, asset_name, most=1)
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


def _already_built(builder, points):
    """Is this body's geometry already in the builder? Asked of its own first vertex.

    At build time the importer has not moved anything, so a body it imported is in the particle
    array at the coordinates the asset authored. Matching one authored point is enough to tell
    "this body is in there" from "this body is missing", and it does not care what order the
    importer used or how many other bodies there are.
    """
    if not builder.particle_count or points is None or not len(points):
        return False
    q = np.asarray(builder.particle_q, dtype=np.float64)
    first = np.asarray(points[0], dtype=np.float64)
    return bool((np.abs(q - first).max(axis=1) < 1e-9).any())


def add_missing(builder, asset, report=None):
    """Whatever the importer left out, built the way the engine's own examples build it.

    Returns a list of what was added; empty means the importer had already covered everything.

    Asked per body, not once for the whole asset. "The importer produced nothing" was the test,
    and it is wrong for an asset that is more than one body: Newton 1.2.1 imports the *volume* of
    a loaded polybag and knows nothing of `PhysicsSurfaceDeformableSimAPI`, so the film was
    skipped because the filling was there, and the model was a bag of cotton with no bag. That is
    the silent kind of wrong -- 10932 particles where the asset declares 18704 -- and it was
    caught downstream by a count, which is a worse place to catch it than here.
    """
    stage = Usd.Stage.Open(str(asset))
    added = []
    for kind, prim in find(stage):
        if kind != "surface":
            continue                       # both versions' importers build volumes
        points = UsdGeom.PointBased(prim).GetPointsAttr().Get()
        if _already_built(builder, points):
            continue
        added.append(add_surface(builder, stage, prim, report))
    return added
