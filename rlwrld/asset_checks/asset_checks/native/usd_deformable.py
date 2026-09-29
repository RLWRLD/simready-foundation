"""What a deformable USD declares, read the way Newton 1.5.0's importer reads it -- and the parts
its importer leaves out, read by rule: per-body particle radii, surface stiffness in the unit the
asset states it, material damping in the unit the solver's kernel reads, the seal springs and
contact exclusions a vendor authors for its own runtime, and the render meshes a recording binds.

`loader.py` is the one caller that builds; `check.py` asks `why_not_runnable` before launching.
"""
import numpy as np
try:
    from pxr import Usd, UsdGeom
except ImportError:
    # The names and rules in this module are read by tools that have no USD (run.py, the
    # coordinator); every function here needs it and fails on use, not on import.
    Usd = UsdGeom = None

# The attribute names an asset may use for each property live in asset_properties. This module
# is imported bare by the runners (native/ on the path) and as a package member by kit/scene.
try:
    import asset_properties
except ImportError:                                       # pragma: no cover
    from asset_checks.native import asset_properties

# The AOUSD Deformable Body Physics schema names, as Newton's importer looks for them.
BODY = "PhysicsDeformableBodyAPI"
SURFACE_SIM = "PhysicsSurfaceDeformableSimAPI"
VOLUME_SIM = "PhysicsVolumeDeformableSimAPI"
SURFACE_MATERIAL = "PhysicsSurfaceDeformableMaterialAPI"
VOLUME_MATERIAL = "PhysicsVolumeDeformableMaterialAPI"
SIM = {"surface": SURFACE_SIM, "volume": VOLUME_SIM}
MATERIAL = {"surface": SURFACE_MATERIAL, "volume": VOLUME_MATERIAL}



def _schemas(prim):
    listop = prim.GetMetadata("apiSchemas")
    return list(listop.GetAddedOrExplicitItems()) if listop else list(prim.GetAppliedSchemas())


def _value(prim, name):
    attr = prim.GetAttribute(name)
    return float(attr.Get()) if attr and attr.HasAuthoredValue() else None


def _bound_material(stage, prim, wanted):
    """The material this prim (or an ancestor) binds for physics, if it carries `wanted`.

    Bound, not found: a material picked up from anywhere on the stage because the body bound
    none is a number the asset never attached to this body, and it was taken silently.
    """
    from pxr import UsdShade

    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
    if bound and wanted in _schemas(bound.GetPrim()):
        return bound.GetPrim()
    return None


# What a body's material has to state for the experiments to mean anything, per kind, by the
# AOUSD names. Missing, the asset is refused: Newton's builder and this package each had a
# different silent default for these, and a cloth's bend stiffness was 100 on one engine and 0 on
# another, with nothing in either log.
REQUIRED = {
    "surface": (("thickness", asset_properties.THICKNESS), ("stretch stiffness", asset_properties.STRETCH),
                ("bend stiffness", asset_properties.BEND), ("density", asset_properties.DENSITY)),
    "volume": (("Young's modulus", asset_properties.YOUNGS), ("Poisson's ratio", asset_properties.POISSON),
               ("density", asset_properties.DENSITY)),
}


def missing_material(stage, kind, prim):
    """-> the reason this body's material is not enough to simulate it, or None."""
    material = _bound_material(stage, prim, MATERIAL[kind])
    if material is None:
        return f"{prim.GetPath()} binds no {MATERIAL[kind]} material for physics"
    for label, names in REQUIRED[kind]:
        if not any(material.GetAttribute(n) and material.GetAttribute(n).HasAuthoredValue() for n in names):
            return (f"{material.GetPath()} states no {label} ({' or '.join(names)}), which a "
                    f"{kind} body's physics depends on; refused rather than given a default")
    return None


def contact_size_of(stage, kind, prim):
    """This body's own contact size: its declared particle radius, else half its shell thickness,
    else None. Per body, because a loaded polybag's film and filling declare different ones."""
    material = _bound_material(stage, prim, MATERIAL[kind])
    prims = [p for p in (material, prim) if p is not None]
    for p in prims:
        for name in asset_properties.RADIUS:
            attr = p.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                return float(attr.Get())
    for p in prims:
        for name in asset_properties.THICKNESS:
            attr = p.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                return 0.5 * float(attr.Get())
    return None


def _world_points(prim):
    """The prim's points in world coordinates, which is where the importer put them -- an asset
    that carries a transform on its simulation prim is matched there, not at its local values."""
    xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
    local = np.asarray(UsdGeom.PointBased(prim).GetPointsAttr().Get(), dtype=np.float64)
    return local @ xf[:3, :3] + xf[3, :3]


def _authored(prim, names):
    for name in names:
        attr = prim.GetAttribute(name)
        if attr and attr.HasAuthoredValue():
            return float(attr.Get()), name
    return None, None


# The power of the shell thickness each authored surface stiffness is multiplied by when it is
# read as a modulus (membrane ~ E*t, bending ~ E*t^3). One table.
SURFACE_THICKNESS_POWER = {"stretchStiffness": 1, "shearStiffness": 1, "bendStiffness": 3}


def surface_stiffnesses(values, reading):
    """{name: stiffness} for authored surface values (physics:* names without the prefix,
    `thickness` among them), read as `reading` ("modulus" or "stiffness")."""
    if reading not in ("modulus", "stiffness"):
        raise SystemExit(f"unknown surface reading {reading!r}")
    t = values.get("thickness")
    if reading == "modulus" and t is None:
        raise SystemExit("reading a surface stiffness as a modulus needs physics:thickness")
    return {name: (value * t ** SURFACE_THICKNESS_POWER[name] if reading == "modulus" else value)
            for name, value in values.items() if name in SURFACE_THICKNESS_POWER}


def read_surface_stiffness(builder, stage, reading, declared=None, report=None):
    """Put each surface body's stretch and bend stiffness on its own elements, read as `reading`.

    `modulus`: the authored numbers are moduli, times the thickness (tri_ke = s*t, edge_ke = b*t^3),
    which is what Newton 1.5.0's importer already built -- checked, not assumed.
    `stiffness`: the authored numbers are the stiffnesses themselves, OmniPhysics' definition.
    Every run prints both, and which of them the asset's own `newton:triKe` / `newton:edgeKe` (its
    engine-native statement of the same thing, where it makes one) agrees with -- a quantity stated
    twice in two units is a question for whoever authored it, and it must not pass unsaid.
    """
    if reading not in ("modulus", "stiffness", "auto"):
        raise SystemExit(f"unknown surface reading {reading!r}")
    tris = np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3) if builder.tri_indices else np.zeros((0, 3), int)
    edges = np.asarray(builder.edge_indices, dtype=np.int64).reshape(-1, 4) if builder.edge_indices else np.zeros((0, 4), int)
    tri_m = [list(m) for m in builder.tri_materials]
    edge_m = [list(m) for m in builder.edge_bending_properties]
    for kind, sim, _render in bodies(stage):
        if kind != "surface":
            continue
        material = _bound_material(stage, sim, MATERIAL[kind])
        t = _value(material, "physics:thickness")
        stretch = _value(material, "physics:stretchStiffness")
        bend = _value(material, "physics:bendStiffness")
        if t is None or stretch is None or bend is None:
            raise SystemExit(f"{sim.GetPath()}: a surface body needs physics:thickness, "
                             f"physics:stretchStiffness and physics:bendStiffness to be read either way")
        authored = {"thickness": t, "stretchStiffness": stretch, "bendStiffness": bend}
        as_modulus = tuple(surface_stiffnesses(authored, "modulus")[k] for k in ("stretchStiffness", "bendStiffness"))
        as_stiffness = tuple(surface_stiffnesses(authored, "stiffness")[k] for k in ("stretchStiffness", "bendStiffness"))
        mine = set(builder_index(builder, sim).tolist())
        own_tris = [i for i, tri in enumerate(tris) if all(int(v) in mine for v in tri)]
        own_edges = [i for i, e in enumerate(edges) if int(e[2]) in mine and int(e[3]) in mine]
        built_tri = sorted({round(tri_m[i][0], 9) for i in own_tris})
        built_edge = sorted({round(edge_m[i][0], 12) for i in own_edges})
        if (len(built_tri) != 1 or not np.isclose(built_tri[0], as_modulus[0], rtol=1e-5)
                or len(built_edge) != 1 or not np.isclose(built_edge[0], as_modulus[1], rtol=1e-4, atol=1e-15)):
            raise SystemExit(f"{sim.GetPath()}: the builder holds tri_ke {built_tri[:3]} / edge_ke "
                             f"{built_edge[:3]}, not the importer's reading {as_modulus[0]:g} / "
                             f"{as_modulus[1]:g}; the importer changed and this reading no longer "
                             f"knows what it starts from")
        native_tri = _value(material, "newton:triKe")
        native_edge = _value(material, "newton:edgeKe")
        agrees = [name for name, (a, b) in (("modulus", as_modulus), ("stiffness", as_stiffness))
                  if native_tri is not None and native_edge is not None
                  and np.isclose(native_tri, a, rtol=1e-5) and np.isclose(native_edge, b, rtol=1e-4)]
        used = reading
        if reading == "auto":
            if native_tri is None or native_edge is None:
                used = "modulus"          # the importer's reading, which is what Newton does with it
            elif agrees:
                used = agrees[0]
            else:
                raise SystemExit(f"{sim.GetPath()}: the asset states its stiffness twice "
                                 f"(physics:* and newton:triKe/edgeKe) and they agree under neither "
                                 f"reading; which one it means is a question for its author")
        tri_ke, edge_ke = as_modulus if used == "modulus" else as_stiffness
        for i in own_tris:
            tri_m[i][0] = tri_ke
        for i in own_edges:
            edge_m[i][0] = edge_ke
        native = ("the asset authors no newton:triKe/edgeKe" if native_tri is None or native_edge is None
                  else f"the asset's own newton:triKe {native_tri:g} / newton:edgeKe {native_edge:g} "
                       + (f"agree with the {agrees[0]} reading" if agrees
                          else "agree with neither reading"))
        print(f"[baseline] surface stiffness {sim.GetPath()}: read as moduli x t (Newton's importer) "
              f"tri_ke {as_modulus[0]:.4g} edge_ke {as_modulus[1]:.4g}; read as stiffnesses "
              f"(OmniPhysics) tri_ke {as_stiffness[0]:.4g} edge_ke {as_stiffness[1]:.4g}; {native}; "
              f"this run: {reading}" + (f" -> {used}" if used != reading else ""))
        if declared is not None and agrees and agrees[0] == used:
            asset_properties.consume(declared, "newton:triKe", "newton:edgeKe")
        if report is not None:
            report[f"tri_ke {sim.GetPath()}"] = (tri_ke, f"physics:stretchStiffness read as {used}")
            report[f"edge_ke {sim.GetPath()}"] = (edge_ke, f"physics:bendStiffness read as {used}")
    builder.tri_materials = [tuple(m) for m in tri_m]
    builder.edge_bending_properties = [tuple(m) for m in edge_m]


def carry_material_damping(builder, stage, convert, report=None, declared=None):
    """Put each body's authored damping on its own elements, in the builder.

    Newton's importer takes every stiffness from the USD and no damping: a volume's k_damp comes
    out 0 and a membrane's tri_kd the builder's default, whatever the asset says. Measured, an
    undamped 2 kPa cotton rebounds off the floor and kicks the film around it into the floor.

    `convert(kind, value, stiffness)` says what number the running solver's kernel needs for the
    damping to *be* `value` in Pa.s (the runner reads that off the kernel). Elements are matched
    to bodies through their particles, which are at the asset's own coordinates at this stage.
    Where the asset authors nothing the builder's default stays, and is reported as ours.
    """
    q = np.asarray(builder.particle_q, dtype=np.float64)
    where = {tuple(np.round(p, 6)): i for i, p in enumerate(q)}
    tets = np.asarray(builder.tet_indices, dtype=np.int64).reshape(-1, 4) if builder.tet_indices else np.zeros((0, 4), int)
    tris = np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3) if builder.tri_indices else np.zeros((0, 3), int)
    edges = np.asarray(builder.edge_indices, dtype=np.int64).reshape(-1, 4) if builder.edge_indices else np.zeros((0, 4), int)
    tet_m = [list(m) for m in builder.tet_materials]
    tri_m = [list(m) for m in builder.tri_materials]
    edge_m = [list(m) for m in builder.edge_bending_properties]
    for kind, sim, _render in bodies(stage):
        material = _bound_material(stage, sim, MATERIAL[kind])
        points = _world_points(sim)
        mine = {where[tuple(np.round(p, 6))] for p in points if tuple(np.round(p, 6)) in where}
        path = str(sim.GetPath())
        if len(mine) != len(points):
            raise SystemExit(f"{path}: {len(points) - len(mine)} of its {len(points)} points are not in "
                             f"the builder at their world coordinates; its damping cannot be assigned")
        if kind == "volume":
            value, name = _authored(material, asset_properties.TET_DAMPING) if material else (None, None)
            if value is None:
                if report is not None:
                    report[f"tet_damping {path}"] = (tet_m[0][2] if tet_m else 0.0, "the asset authors no "
                                                     "newton:kDamp; Newton's builder default, ours")
                continue
            for t, (a, b, c, d) in enumerate(tets):
                if a in mine:
                    tet_m[t][2] = convert("tet", value, tet_m[t][0])
            if declared is not None:
                asset_properties.consume(declared, name)
            if report is not None:
                report[f"tet_damping {path}"] = (value, f"{name} in Pa.s, handed over as this kernel reads it")
        else:
            value, name = _authored(material, asset_properties.TRI_DAMPING) if material else (None, None)
            bend, bend_name = _authored(material, asset_properties.EDGE_DAMPING) if material else (None, None)
            if value is None:
                if report is not None:
                    report[f"tri_damping {path}"] = (tri_m[0][2] if tri_m else 0.0, "the asset authors no "
                                                     "newton:triKd; Newton's builder default, ours")
            else:
                for t, (a, b, c) in enumerate(tris):
                    if a in mine:
                        tri_m[t][2] = convert("tri", value, tri_m[t][0])
                if declared is not None:
                    asset_properties.consume(declared, name)
                if report is not None:
                    report[f"tri_damping {path}"] = (value, f"{name}, handed over as this kernel reads it")
            if bend is not None:
                for e, (_, _, a, b) in enumerate(edges):
                    if a in mine:
                        edge_m[e][1] = convert("edge", bend, edge_m[e][0])
                if declared is not None:
                    asset_properties.consume(declared, bend_name)
                if report is not None:
                    report[f"edge_damping {path}"] = (bend, f"{bend_name}, handed over as this kernel reads it")
    builder.tet_materials = [tuple(m) for m in tet_m]
    builder.tri_materials = [tuple(m) for m in tri_m]
    builder.edge_bending_properties = [tuple(m) for m in edge_m]


def assign_particle_radii(builder, stage, fallback, report=None):
    """Give every body's particles that body's own contact size, in the builder; -> {body: r}.

    Newton's importer reads neither `newton:particleRadius` nor a thickness, and one
    `default_particle_radius` covered every body of the asset, so a loaded polybag's filling
    (2 mm) got its film's 3 mm. Done at the builder stage, before the runner lifts anything:
    there each body's authored points are in the particle array at their authored coordinates
    (the fact `_already_built` relies on), and the count has to match, or this refuses rather
    than sizing some of a body.
    """
    q = np.asarray(builder.particle_q, dtype=np.float64)
    radii = np.asarray(builder.particle_radius, dtype=np.float64)
    where = {tuple(np.round(p, 6)): i for i, p in enumerate(q)}
    sizes = {}
    for kind, sim, _render in bodies(stage):
        size = contact_size_of(stage, kind, sim)
        if size is None:
            size = fallback
        points = _world_points(sim)
        index = [where.get(tuple(np.round(p, 6))) for p in points]
        lost = sum(i is None for i in index)
        if lost:
            raise SystemExit(f"{sim.GetPath()}: {lost} of its {len(points)} authored points are not "
                             f"in the builder at their authored coordinates; its contact size cannot "
                             f"be assigned")
        radii[index] = size
        sizes[str(sim.GetPath())] = size
    builder.particle_radius = [float(r) for r in radii]
    print("[baseline] particle radius per body: "
          + "; ".join(f"{path} {size * 1000:.2f} mm" for path, size in sizes.items()))
    if report is not None:
        for path, size in sizes.items():
            report[f"particle_radius {path}"] = (size, "this body's own contact size, from its material")
    return sizes


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


def _drawables(sim):
    """Every drawable a body has: the PointBased prims beside its sim prim that are not simulated
    and that a renderer would draw (a `guide` or `proxy` purpose is not)."""
    out = []
    for q in Usd.PrimRange(sim.GetParent(), Usd.TraverseInstanceProxies()):
        if q == sim or not q.IsA(UsdGeom.PointBased) or simulated_kind(q) is not None:
            continue
        if not UsdGeom.PointBased(q).GetPointsAttr().Get():
            continue
        if UsdGeom.Imageable(q).ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        out.append(q)
    return out


def other_drawables(sim, render):
    """The drawables of a body besides the one `bodies` pairs it with -- a polybag's folded lip
    beside its film. Each has to be carried too, or it stays where it was authored."""
    return [q for q in _drawables(sim) if render is None or q.GetPath() != render.GetPath()]


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
        render = max(_drawables(sim), key=lambda q: len(UsdGeom.PointBased(q).GetPointsAttr().Get()),
                     default=None)
        out.append((kind, sim, render))
    return out


def why_not_runnable(stage, asset_name="", needs=None):
    """-> the reason this asset cannot be run as it is, or None.

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
    if needs is not None and not any(kind in needs for kind, _ in found):
        declared = ", ".join(sorted({kind for kind, _ in found}))
        return (f"{name} declares a {declared} body only, and this experiment means something for "
                f"a {' or '.join(sorted(needs))} body; refused rather than answered about nothing")
    for kind, prim in found:
        reason = missing_material(stage, kind, prim)
        if reason:
            return f"{name}: {reason}"
    return None



def check_imported(builder, stage):
    """Refuse unless every simulated body the asset declares is in the builder, point for point.

    The importer is asked per body, because a count hides it: Newton 1.2.1's imported the cotton of
    a loaded polybag and skipped its film, and the model was a bag of cotton with no bag."""
    for _kind, sim, _render in bodies(stage):
        builder_index(builder, sim)          # raises, naming the body, if any point is missing


# ---------------------------------------------------------------- the structure an asset authors
def builder_index(builder, prim):
    """The builder's particle index of each of `prim`'s authored points, matched in world
    coordinates -- refuses if any point is not there, because half a structure is none."""
    q = np.asarray(builder.particle_q, dtype=np.float64)
    where = {tuple(np.round(p, 6)): i for i, p in enumerate(q)}
    points = _world_points(prim)
    index = [where.get(tuple(np.round(p, 6))) for p in points]
    lost = sum(i is None for i in index)
    if lost:
        raise SystemExit(f"{prim.GetPath()}: {lost} of its {len(points)} authored points are not in "
                         f"the builder at their authored coordinates")
    return np.asarray(index, dtype=np.int64)


def seal_pairs(builder, stage):
    """Every seal pair the asset authors, as builder particle indices: (n, 2), possibly empty.

    Read whether or not the run builds springs from them: the gap across a seal is the one
    measurement that says whether a bag stayed a bag, and it is measured in every setup.
    """
    found = []
    for _kind, sim, _render in bodies(stage):
        attr = sim.GetAttribute(asset_properties.SEAL_PAIRS)
        if not (attr and attr.HasAuthoredValue()):
            continue
        local = np.asarray(attr.Get(), dtype=np.int64).reshape(-1, 2)
        found.append(builder_index(builder, sim)[local])
    return np.concatenate(found) if found else np.zeros((0, 2), dtype=np.int64)


def add_seal_springs(builder, stage, declared, report=None):
    """Build the asset's seal as springs; -> the (n, 2) pairs added, for the colouring.

    A spring per authored pair at the stiffness the asset states -- on the prim
    (`rlwrld:sealStiffness`) or on the root (`rlwrld:contact:seal_spring_ke_n_m`); both, if they
    disagree, is two owners and refused. Newton's `add_spring` takes the pair's current distance
    as the rest length, so a seal authored 2 mm open stays 2 mm open and does not pull shut. The
    asset authors no seal damping and none is invented: the spring's kd is 0 and reported as ours.
    """
    root = stage.GetDefaultPrim()
    root_ke, root_where = _authored(root, asset_properties.RECIPE["seal_ke"]) if root else (None, None)
    pairs, made = [], 0
    for _kind, sim, _render in bodies(stage):
        attr = sim.GetAttribute(asset_properties.SEAL_PAIRS)
        if not (attr and attr.HasAuthoredValue()):
            continue
        prim_ke, prim_where = _authored(sim, (asset_properties.SEAL_KE,))
        if prim_ke is not None and root_ke is not None and abs(prim_ke - root_ke) > 1e-9 * max(prim_ke, root_ke):
            raise SystemExit(f"{sim.GetPath()} states a seal stiffness of {prim_ke:g} and the root "
                             f"{root_ke:g}: two owners for one number; refused")
        ke = prim_ke if prim_ke is not None else root_ke
        if ke is None:
            raise SystemExit(f"{sim.GetPath()} authors {asset_properties.SEAL_PAIRS} but no seal "
                             f"stiffness ({asset_properties.SEAL_KE} or "
                             f"{asset_properties.RECIPE['seal_ke'][0]}); the seal cannot be built")
        where = f"{sim.GetPath()}.{prim_where}" if prim_ke is not None else f"{root.GetPath()}.{root_where}"
        index = builder_index(builder, sim)
        local = np.asarray(attr.Get(), dtype=np.int64).reshape(-1, 2)
        for a, b in index[local]:
            builder.add_spring(int(a), int(b), float(ke), 0.0, 0.0)
        pairs.append(index[local]); made += len(local)
        asset_properties.consume(declared, asset_properties.SEAL_PAIRS, prim_where or "", root_where or "")
        if report is not None:
            report[f"seal_springs {sim.GetPath()}"] = (len(local), f"springs at {ke:g} N/m ({where}), "
                                                       f"rest length = authored gap; kd 0, the asset authors none")
    print(f"[baseline] seal: {made} spring(s) from the asset's seal pairs" if made
          else "[baseline] seal: the asset authors no seal pairs")
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)


def vertex_triangle_exclusions(builder, stage, declared, report=None):
    """The asset's (vertex, triangle) self-contact exclusions as builder ids: {vertex: {tri}}.

    Local vertex ids go through the point match; local triangle ids are the prim's face order,
    matched to the builder's triangles by their vertex triple -- a triple the builder does not
    have means the numbering is not what it looks like, and refuses.
    """
    triple_of = {}
    for t, tri in enumerate(np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3)):
        triple_of[tuple(sorted(tri.tolist()))] = t
    out = {}
    for _kind, sim, _render in bodies(stage):
        attr = sim.GetAttribute(asset_properties.VERTEX_TRIANGLE_EXCLUSIONS)
        if not (attr and attr.HasAuthoredValue()):
            continue
        index = builder_index(builder, sim)
        faces = np.asarray(UsdGeom.Mesh(sim).GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts = np.asarray(UsdGeom.Mesh(sim).GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        if (counts != 3).any():
            raise SystemExit(f"{sim.GetPath()}: contact exclusions name triangles, but the mesh has "
                             f"{(counts != 3).sum()} non-triangle face(s); the numbering is undefined")
        faces = faces.reshape(-1, 3)
        for v, t in np.asarray(attr.Get(), dtype=np.int64).reshape(-1, 2):
            triple = tuple(sorted(index[faces[t]].tolist()))
            if triple not in triple_of:
                raise SystemExit(f"{sim.GetPath()}: exclusion names face {t}, whose vertices are not a "
                                 f"triangle of the builder; the face numbering does not match")
            out.setdefault(int(index[v]), set()).add(triple_of[triple])
        asset_properties.consume(declared, asset_properties.VERTEX_TRIANGLE_EXCLUSIONS)
        if report is not None:
            report[f"contact_exclusions {sim.GetPath()}"] = (
                len(attr.Get()), "vertex-triangle self-contact pairs the asset excludes, handed to the solver")
    return out


# An excluded edge pair is two edges that touch at rest (a fold lying on itself). Under the right
# numbering their midpoints are far closer than the same edges paired at random; this is how
# much closer, at least, before the numbering is believed.
EDGE_EXCLUSION_NEARNESS = 0.25


def edge_edge_exclusions(builder, stage, declared, report=None):
    """The asset's edge-edge self-contact exclusions as builder ids: {edge: {edge}}.

    Edge ids are the ones Newton's own `add_cloth_mesh` gives the prim's authored face list --
    the numbering the solver's filtering map takes, and the one the vendor's authoring tool
    produced (it is not stated in the asset; it is measured: see EDGE_EXCLUSION_NEARNESS).
    Each local edge goes to the builder through its two vertices, whatever order our build
    added the bodies in; an edge the builder does not have refuses. A pair is written under both
    of its edges: the solver consults only the querying edge's list, so a pair listed once still
    collides from its other side.
    """
    import newton
    import warp as wp
    edge_of = {}
    for e, (a, b) in enumerate(np.asarray(builder.edge_indices, dtype=np.int64).reshape(-1, 4)[:, 2:4]):
        edge_of[(min(a, b), max(a, b))] = e
    out = {}
    for _kind, sim, _render in bodies(stage):
        attr = sim.GetAttribute(asset_properties.EDGE_EXCLUSIONS)
        if not (attr and attr.HasAuthoredValue()):
            continue
        mesh = UsdGeom.Mesh(sim)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if (np.asarray(mesh.GetFaceVertexCountsAttr().Get()) != 3).any():
            raise SystemExit(f"{sim.GetPath()}: edge exclusions are authored on a mesh with "
                             f"non-triangle faces; Newton's edge numbering is undefined there")
        points = _world_points(sim)
        local = newton.ModelBuilder()
        local.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0,
                             vel=wp.vec3(0.0, 0.0, 0.0), vertices=[wp.vec3(*p) for p in points],
                             indices=faces.tolist(), density=1.0)
        ends = np.asarray(local.edge_indices, dtype=np.int64).reshape(-1, 4)[:, 2:4]
        pairs = np.asarray(attr.Get(), dtype=np.int64).reshape(-1, 2)
        if pairs.min() < 0 or pairs.max() >= len(ends):
            raise SystemExit(f"{sim.GetPath()}: edge exclusions name edge {pairs.max()}, but Newton "
                             f"numbers {len(ends)} edges on this mesh; the numbering is not Newton's")
        mid = points[ends].mean(axis=1)
        near = np.median(np.linalg.norm(mid[pairs[:, 0]] - mid[pairs[:, 1]], axis=1))
        shuffled = np.median(np.linalg.norm(
            mid[np.random.default_rng(0).permutation(pairs[:, 0])] - mid[pairs[:, 1]], axis=1))
        if near > EDGE_EXCLUSION_NEARNESS * shuffled:
            raise SystemExit(f"{sim.GetPath()}: under Newton's edge numbering the excluded edge "
                             f"pairs are {near * 1e3:.1f} mm apart (median), against {shuffled * 1e3:.1f} mm "
                             f"for the same edges paired at random; the numbering is not Newton's")
        index = builder_index(builder, sim)
        for e_a, e_b in pairs:
            ids = []
            for e in (e_a, e_b):
                a, b = index[ends[e]]
                key = (min(a, b), max(a, b))
                if key not in edge_of:
                    raise SystemExit(f"{sim.GetPath()}: excluded edge {e} joins points the builder "
                                     f"has no edge between")
                ids.append(edge_of[key])
            out.setdefault(ids[0], set()).add(ids[1])
            out.setdefault(ids[1], set()).add(ids[0])
        asset_properties.consume(declared, asset_properties.EDGE_EXCLUSIONS)
        print(f"[baseline] {len(pairs)} edge-edge exclusion(s) on {sim.GetPath()} in Newton's edge "
              f"numbering: excluded pairs {near * 1e3:.1f} mm apart at rest (median), "
              f"{shuffled * 1e3:.1f} mm when paired at random")
        if report is not None:
            report[f"edge_exclusions {sim.GetPath()}"] = (
                len(pairs), "edge-edge self-contact pairs the asset excludes, in Newton's edge numbering, handed to the solver")
    return out
