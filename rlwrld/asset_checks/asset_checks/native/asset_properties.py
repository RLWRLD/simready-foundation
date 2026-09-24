"""What the asset says about itself, and what it leaves to us.

The rule this file exists to enforce: **a physical property of the asset comes from the USD.**
The experiment picks the scene and the solver picks its numerics; neither may quietly decide what
the thing is made of. Where the USD is silent we still have to put a number somewhere, and then
the number is reported as ours rather than passed off as the asset's.

That distinction is not pedantic. Newton's importer does not read `newton:particleRadius` at all
-- this banana declares 0.76 mm and gets Newton's 100 mm default -- and it derives the Lame
parameters from `physics:youngsModulus` and `physics:poissonsRatio` rather than from the
`newton:kMu` and `newton:kLambda` the asset also spells out. Somebody has to notice. Equally,
contact friction is a property of the materials in the scene, not of the integrator: giving VBD
mu=0.3 and XPBD mu=1.0 because two Newton examples happened to use those numbers makes the two
solvers incomparable, which is the one thing this whole benchmark must not do.
"""
try:
    from pxr import Usd, UsdGeom
except ImportError:
    # The names and rules in this module are read by tools that have no USD (run.py, the
    # coordinator); every function here needs it and fails on use, not on import.
    Usd = UsdGeom = None

# What the AOUSD physics schemas call these, in the order we prefer them. `newton:` names are what
# an asset authored for Newton spells out directly; `physics:` names are the portable ones.
RADIUS = ("newton:particleRadius",)
FRICTION = ("physics:dynamicFriction", "physics:staticFriction", "newton:dynamicFriction")
RESTITUTION = ("physics:restitution", "newton:restitution")
DENSITY = ("physics:density", "newton:density")
# Used only where the asset declares no friction. It has to be one number for every engine and
# every solver: PhysX's own helper defaults to 0.25 dynamic / 0.5 static and Newton's importer to
# 0.5, so an asset that says nothing would have had the two engines rubbing different floors.
DEFAULT_FRICTION = 0.5
YOUNGS = ("physics:youngsModulus",)
POISSON = ("physics:poissonsRatio",)
# A surface deformable states its shell thickness rather than a particle radius. Newton's own
# importer takes half of it as the collision radius, so that is the contact size the asset is
# asking for, and it is the one every engine should be given.
THICKNESS = ("physics:thickness", "newton:thickness")
# What a surface deformable states instead of a modulus: a membrane's stiffness is a force per
# unit length, not a pressure, and the two are not interchangeable.
STRETCH = ("physics:stretchStiffness", "newton:stretchStiffness")
BEND = ("physics:bendStiffness", "newton:bendStiffness")
SHEAR = ("physics:shearStiffness", "newton:shearStiffness")
# A material damping, where the asset states one. Read so that it is at least *reported*: Newton's
# importer sets its own triangle damping and PhysX's conversion carries none, so a declared value
# reaches neither engine, and a number the asset declares and nobody honours must be said.
DAMPING = ("newton:kDamp", "newton:triKd", "physics:damping")
# Per element kind, the attribute that damps it: a volume's tetrahedra, a membrane's triangles
# and its bending edges. These are what Newton-flavoured assets author (the AOUSD proposal has
# no damping attribute yet); the importer reads none of them.
TET_DAMPING = ("newton:kDamp",)
TRI_DAMPING = ("newton:triKd",)
EDGE_DAMPING = ("newton:edgeKd",)
# Whether the asset collides with itself. This is a physical claim about the thing -- a sheet that
# folds onto itself behaves differently from one that passes through itself -- so it belongs to the
# asset, not to whichever solver happens to offer the switch. Every schema that declares it is
# listed; where an asset is silent the deformable schema's own default answers, and that default is
# `physxDeformableBody:selfCollision = 0`, off.
SELF_COLLISION = ("physxDeformableBody:selfCollision", "physxParticle:selfCollision",
                  "newton:selfCollisionEnabled")
SELF_COLLISION_WHEN_SILENT = False
# What an asset authors for its own runtime, beside the schemas. The polybag family carries a
# recipe under `rlwrld:` on its default prim -- the contact's numbers, the stepping, the seal that
# closes the bag, the self-contact that lets its film meet its filling -- and neither Newton's
# importer nor PhysX reads a word of it. Which of it a run consumes is the setup's choice
# (`setups.py`); that it is authored is printed either way, so no run can say "the asset declares
# none" of a thing the asset declares. Measured before this existed: the canon's log said exactly
# that of this asset's friction and self-contact, both authored here.
RECIPE = {
    "contact_ke": ("rlwrld:contact:soft_contact_ke",),
    "contact_kd": ("rlwrld:contact:soft_contact_kd",),
    "shape_ke": ("rlwrld:contact:shape_contact_ke",),
    "friction": ("rlwrld:simulation:soft_contact_mu",),
    "dt": ("rlwrld:simulation:dt_s",),
    "iterations": ("rlwrld:simulation:iterations",),
    "self_contact_radius": ("rlwrld:contact:self_contact_radius_m",),
    "self_contact_margin": ("rlwrld:contact:self_contact_margin_m",),
    "seal_ke": ("rlwrld:contact:seal_spring_ke_n_m",),
    "particle_radius": ("rlwrld:contact:particle_radius_m",),
    "damping_s": ("rlwrld:simulation:damping_s",),
}
# On a simulated prim: the structure that makes its mesh one object.
SEAL_PAIRS = "rlwrld:sealPairs"                                # (n, 2) prim-local point indices
SEAL_KE = "rlwrld:sealStiffness"                               # N/m, the prim's own word for RECIPE["seal_ke"]
VERTEX_TRIANGLE_EXCLUSIONS = "rlwrld:foldVertexContactExclusions"   # (n, 2) local (vertex, triangle)
EDGE_EXCLUSIONS = "rlwrld:foldEdgeContactExclusions"                # (n, 2) Newton's add_cloth_mesh edge ids
# Namespaces that are USD's own or the schemas': an authored attribute outside these, on a prim a
# run reads, is a vendor's word to this pipeline, and one it does not consume is printed as such.
STANDARD_NAMESPACES = ("primvars", "xformOp", "material", "physics", "inputs", "outputs", "ui")


def _first(prims, names):
    """The first of `names` any of `prims` authors, with the prim and attribute that carried it."""
    for prim in prims:
        for name in names:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                return float(attr.Get()), f"{prim.GetPath()}.{name}"
    return None, None


def _first_flag(prims, names):
    """Like `_first`, for a declared boolean."""
    for prim in prims:
        for name in names:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                return bool(attr.Get()), f"{prim.GetPath()}.{name}"
    return None, None


def read(asset):
    """Everything the asset declares about its physics, each with where it came from.

    Values are `None` when the asset is silent, which is a caller's cue to choose one and say so
    -- never to pretend the asset asked for it.
    """
    stage = Usd.Stage.Open(str(asset))
    materials = [p for p in stage.Traverse()
                 if any("Material" in s for s in (p.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
                                                  if p.GetMetadata("apiSchemas") else []))]
    geometry = [p for p in stage.Traverse() if p.IsA(UsdGeom.PointBased)]
    prims = materials + geometry
    found = {"_asset": str(asset)}          # `account` re-reads it to check restatements
    for key, names in (("particle_radius", RADIUS), ("friction", FRICTION),
                       ("restitution", RESTITUTION), ("density", DENSITY),
                       ("youngs_modulus", YOUNGS), ("poissons_ratio", POISSON),
                       ("thickness", THICKNESS), ("stretch_stiffness", STRETCH),
                       ("bend_stiffness", BEND), ("shear_stiffness", SHEAR),
                       ("damping", DAMPING)):
        value, where = _first(prims, names)
        found[key] = value
        found[key + "_source"] = where
    found["self_collision"], found["self_collision_source"] = _first_flag(prims, SELF_COLLISION)
    # The recipe lives on the default prim, the one that is the asset.
    root = stage.GetDefaultPrim()
    found["recipe"] = {}
    for key, names in RECIPE.items():
        value, where = _first([root] if root else [], names)
        if value is not None:
            found["recipe"][key] = (value, where)
    # Every namespaced attribute a vendor authored on the prims a run reads, so that what a run
    # did not consume can be said. `_consumed` is what the standard reading above took; a runner
    # adds what its setup took, and `report` prints the difference.
    found["_authored"] = {}
    for prim in ([root] if root else []) + prims:
        names = sorted(a.GetName() for a in prim.GetAttributes()
                       if a.HasAuthoredValue() and ":" in a.GetName()
                       and a.GetName().split(":")[0] not in STANDARD_NAMESPACES)
        if names:
            found["_authored"][str(prim.GetPath())] = names
    found["_consumed"] = {found[k + "_source"].rsplit(".", 1)[1] for k in
                          ("particle_radius", "friction", "restitution", "density", "youngs_modulus",
                           "poissons_ratio", "thickness", "stretch_stiffness", "bend_stiffness",
                           "shear_stiffness", "damping") if found.get(k + "_source")}
    if found.get("self_collision_source"):
        found["_consumed"].add(found["self_collision_source"].rsplit(".", 1)[1])
    return found


def consume(declared, *names):
    """Mark attribute names a runner's setup read, so `report` does not list them as unread."""
    declared.setdefault("_consumed", set()).update(names)


# ------------------------------------------------------------------ every authored word, accounted for
# A vendor attribute a run does not read is either deliberately set aside by the run's setup, or
# accounted for below, or the run refuses. Printing "not consumed" and carrying on is how the
# polybag's fold edge exclusions sat unread for two days while the bag it held together flew
# apart (2026-09-22..24). The classes:
#   restated    the same quantity as a consumed attribute, in the vendor's runtime's own terms;
#               checked against it under that runtime's conversion (`restated_disagreements`),
#               and a disagreement refuses -- that check is what would have caught the film's
#               stiffness written in the wrong unit (2026-09-23)
#   experiment  belongs to an experiment fixture (a gripper), not to the asset
#   runtime     a solver buffer or the vendor's file bookkeeping; no physics
#   authoring   the vendor's authoring intermediates, already baked into the authored mesh; not
#               among the runtime items the vendor listed (SpaceAI reply, 2026-09-24)
#   visual      binds a display mesh; a recording concern, which binds by its own rule
# Names, not patterns: a pattern that covers today's fold data would also cover tomorrow's fold
# exclusions, which is exactly the attribute this list exists not to wave through.
ACCOUNTED = {
    "newton:density": "restated", "newton:kMu": "restated", "newton:kLambda": "restated",
    "newton:triKa": "restated", "rlwrld:contact:particle_radius_m": "restated",
    "rlwrld:simulation:damping_s": "restated",
    "rlwrld:simulation:grip_contact_mu": "experiment",
    "rlwrld:contact:rigid_body_particle_contact_buffer_size": "runtime",
    "rlwrld:formatVersion": "runtime", "rlwrld:runtimeAdapter": "runtime",
    "rlwrld:flatPattern": "authoring", "rlwrld:foldBindIndices": "authoring",
    "rlwrld:foldBindWeights": "authoring", "rlwrld:foldFlapFaces": "authoring",
    "rlwrld:foldFlapPattern": "authoring", "rlwrld:foldFlapThickness": "authoring",
    "rlwrld:foldFormedRestPoints": "authoring", "rlwrld:foldLipIndices": "authoring",
    "rlwrld:foldNormalOffsets": "authoring", "rlwrld:foldParticleArea": "authoring",
    "rlwrld:foldTrianglePattern": "authoring", "rlwrld:foldTriangleRest": "authoring",
    "rlwrld:labelTriangles": "authoring",
    "rlwrld:normalOffsets": "visual", "rlwrld:particleIndices": "visual",
}
# What each setup factor owns: under a level other than `asset` these are set aside on purpose.
SETUP_OWNED = {
    "structure": (SEAL_PAIRS, SEAL_KE, VERTEX_TRIANGLE_EXCLUSIONS, EDGE_EXCLUSIONS,
                  *RECIPE["self_contact_radius"], *RECIPE["self_contact_margin"], *RECIPE["seal_ke"]),
    "contact": (*RECIPE["contact_ke"], *RECIPE["contact_kd"], *RECIPE["friction"], *RECIPE["shape_ke"]),
    "stepping": (*RECIPE["dt"], *RECIPE["iterations"]),
}
RESTATED_TOLERANCE = 1e-4          # relative; the vendor's floats are written at float32


def _prim_values(stage, path):
    prim = stage.GetPrimAtPath(path)
    return {a.GetName(): a.Get() for a in prim.GetAttributes() if a.HasAuthoredValue()}


def restated_disagreements(declared):
    """Every `restated` attribute checked against what it restates -> [problem]. A restatement
    whose counterpart is missing is a problem too: it cannot be checked, so it is not accounted for."""
    out = []
    stage = Usd.Stage.Open(declared["_asset"])

    def differ(got, want):
        return abs(got - want) > RESTATED_TOLERANCE * max(abs(want), 1e-12)

    authored = declared.get("_authored", {})
    radii = []
    for path, names in authored.items():
        v = _prim_values(stage, path)
        if "newton:particleRadius" in v:
            radii.append(float(v["newton:particleRadius"]))
    for path, names in sorted(authored.items()):
        v = _prim_values(stage, path)
        E, nu, rho, t = (v.get("physics:youngsModulus"), v.get("physics:poissonsRatio"),
                         v.get("physics:density"), v.get("physics:thickness"))
        for name in names:
            if ACCOUNTED.get(name) != "restated":
                continue
            got = v[name]
            if name == "newton:density":
                want = None if rho is None else rho * (t if t is not None else 1.0)
                what = "physics:density" + (" x physics:thickness (areal)" if t is not None else "")
            elif name in ("newton:kMu", "newton:kLambda"):
                want = None if E is None or nu is None else (
                    E / (2 * (1 + nu)) if name == "newton:kMu" else E * nu / ((1 + nu) * (1 - 2 * nu)))
                what = "the Lame parameter of physics:youngsModulus and physics:poissonsRatio"
            elif name == "newton:triKa":
                want, what = 0.0, "Newton's membrane with no Poisson term, which the schema authors none of"
            elif name == "rlwrld:contact:particle_radius_m":
                want = min(radii, key=lambda r: abs(r - got)) if radii else None
                what = "a body's newton:particleRadius"
            elif name == "rlwrld:simulation:damping_s":
                ratios = []
                for other in authored:
                    ov = _prim_values(stage, other)
                    for kd, ke in (("newton:triKd", "newton:triKe"), ("newton:edgeKd", "newton:edgeKe")):
                        if kd in ov and ov.get(ke):
                            ratios.append((f"{other}.{kd}/{ke}", ov[kd] / ov[ke]))
                bad = [r for r in ratios if differ(r[1], got)]
                if not ratios:
                    out.append(f"{path}.{name} = {got:g} restates damping ratios the asset authors none of")
                for where, r in bad:
                    out.append(f"{path}.{name} = {got:g} but {where} = {r:.6g}")
                continue
            if want is None:
                out.append(f"{path}.{name} = {got:g} restates {what}, which is not authored beside it")
            elif differ(float(got), want):
                out.append(f"{path}.{name} = {got:g} but {what} gives {want:.6g}")
    return out


def account(declared, setup):
    """Refuse the run unless every vendor attribute it does not consume is set aside by the setup
    or accounted for (ACCOUNTED), and every restatement agrees; -> {name: why} of what was set
    aside, for the report."""
    consumed = declared.get("_consumed", set())
    aside, unknown = {}, []
    for path, names in sorted(declared.get("_authored", {}).items()):
        for name in names:
            if name in consumed:
                continue
            owner = next((f for f, owned in SETUP_OWNED.items() if name in owned), None)
            if owner is not None and setup.get(owner) != "asset":
                aside[name] = f"set aside by setup {owner}={setup.get(owner)}"
            elif name in ACCOUNTED:
                aside[name] = ACCOUNTED[name]
            else:
                unknown.append(f"{path}.{name}")
    problems = restated_disagreements(declared)
    if unknown:
        raise SystemExit("refused: the asset authors attribute(s) this run neither reads nor accounts "
                         "for -- read them, or add them to asset_properties.ACCOUNTED with a class: "
                         + ", ".join(unknown))
    if problems:
        raise SystemExit("refused: the asset states a quantity twice and the two disagree: "
                         + "; ".join(problems))
    return aside


def report(tag, declared, chosen, setup):
    """Print, every run, which numbers the asset gave and which we picked -- and refuse the run if
    the asset authors anything it neither reads nor accounts for (`account`).

    A run whose log does not say this cannot be audited later, and a benchmark nobody can audit
    is a benchmark nobody should believe.
    """
    aside = account(declared, setup)
    for key in ("particle_radius", "thickness", "friction", "restitution", "density",
                "youngs_modulus", "poissons_ratio", "stretch_stiffness", "bend_stiffness",
                "shear_stiffness", "damping"):
        value, where = declared.get(key), declared.get(key + "_source")
        if value is not None:
            print(f"[{tag}] asset: {key} = {value:g}  ({where})")
    if declared.get("self_collision") is not None:
        print(f"[{tag}] asset: self_collision = {declared['self_collision']}  "
              f"({declared['self_collision_source']})")
    for key, (value, why) in sorted(chosen.items()):
        print(f"[{tag}] ours:  {key} = {value:g}  -- {why}")
    for key, (value, where) in sorted(declared.get("recipe", {}).items()):
        print(f"[{tag}] asset recipe: {key} = {value:g}  ({where})")
    consumed = declared.get("_consumed", set())
    for path, names in sorted(declared.get("_authored", {}).items()):
        unread = [n for n in names if n not in consumed]
        if unread:
            print(f"[{tag}] authored on {path}, not read by this run ({len(unread)}): "
                  + ", ".join(f"{n} [{aside[n]}]" for n in unread))
    print(f"[{tag}] every authored attribute is read, set aside by the setup, or accounted for; "
          f"restatements agree")


def friction(declared):
    """-> (mu, why). The asset's friction, or the one default shared by every engine and solver.

    One function, because it was two: Newton's runner asked `declared` and PhysX's conversion
    asked the material prim with its own fallback, and an asset declaring only a static friction
    reached one engine as that number and the other as the default.
    """
    if declared.get("friction") is not None:
        return declared["friction"], declared.get("friction_source")
    recipe = declared.get("recipe", {}).get("friction")
    if recipe is not None:
        return DEFAULT_FRICTION, (f"the asset authors {recipe[0]:g} for its own runtime "
                                  f"({recipe[1]}), which this setup's contact source does not read; "
                                  f"the default every engine and solver shares")
    return DEFAULT_FRICTION, ("the asset declares no friction under the schemas; one value for "
                              "every engine and solver so they rub the same floor")


def self_collision(declared):
    """-> (on, why). The asset's answer, or the deformable schema's default when it is silent.

    Every engine is handed the same answer. Where an engine has no switch for it, its runner says
    so rather than leaving the reader to assume the five ran the same model.
    """
    if declared.get("self_collision") is not None:
        return declared["self_collision"], declared["self_collision_source"]
    recipe = declared.get("recipe", {}).get("self_contact_radius")
    if recipe is not None:
        return (SELF_COLLISION_WHEN_SILENT,
                f"the asset authors a self-contact radius for its own runtime ({recipe[1]}), which "
                f"this setup's structure source does not read; the schema default is off")
    return (SELF_COLLISION_WHEN_SILENT,
            "the asset does not declare it under the schemas; physxDeformableBody:selfCollision "
            "defaults to off")


def contact_size(declared):
    """The radius at which this asset expects contact to happen, if it says.

    A volume deformable states a particle radius; a surface one states a shell thickness and
    Newton halves it. Returning the same number to every engine is what keeps them touching the
    floor the same way -- PhysX's own default is scale-free and Newton's importer only reads one
    of the two attributes.
    """
    if declared.get("particle_radius") is not None:
        return declared["particle_radius"], declared.get("particle_radius_source")
    if declared.get("thickness") is not None:
        return 0.5 * declared["thickness"], f"half of {declared.get('thickness_source')}"
    return None, None
