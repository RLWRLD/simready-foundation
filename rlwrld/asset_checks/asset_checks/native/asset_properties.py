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
from pxr import Usd, UsdGeom, UsdPhysics

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


def _first(prims, names):
    """The first of `names` any of `prims` authors, with the prim and attribute that carried it."""
    for prim in prims:
        for name in names:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                return float(attr.Get()), f"{prim.GetPath()}.{name}"
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
    found = {}
    for key, names in (("particle_radius", RADIUS), ("friction", FRICTION),
                       ("restitution", RESTITUTION), ("density", DENSITY),
                       ("youngs_modulus", YOUNGS), ("poissons_ratio", POISSON)):
        value, where = _first(prims, names)
        found[key] = value
        found[key + "_source"] = where
    return found


def report(tag, declared, chosen):
    """Print, every run, which numbers the asset gave and which we picked.

    A run whose log does not say this cannot be audited later, and a benchmark nobody can audit
    is a benchmark nobody should believe.
    """
    for key in ("particle_radius", "friction", "restitution", "density",
                "youngs_modulus", "poissons_ratio"):
        value, where = declared.get(key), declared.get(key + "_source")
        if value is not None:
            print(f"[{tag}] asset: {key} = {value:g}  ({where})")
    for key, (value, why) in sorted(chosen.items()):
        print(f"[{tag}] ours:  {key} = {value:g}  -- {why}")
