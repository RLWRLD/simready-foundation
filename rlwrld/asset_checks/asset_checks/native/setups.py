# SPDX-License-Identifier: Apache-2.0
"""A setup: where every number the USD does not settle comes from. Four factors, each a source.

The geometry, the material and the mass are the asset's (`asset_properties`). Everything else a
solver needs has to come from somewhere -- what joins the asset's bodies into one object, the
contact's numbers, how finely a frame is stepped -- and a run that does not say where is a run
nobody can compare. This module names the sources. A setup is one choice per factor, and its name
is the four choices joined with dashes: `none-derived-canon-modulus` is the canon,
`asset-asset-asset-stiffness` is the recipe the asset authors for its own runtime, followed in full.

Why this exists: the loaded polybag carries such a recipe (`rlwrld:*`, see
`asset_properties.RECIPE`) -- seal springs that close the bag, self-contact so its film meets its
filling, its own contact numbers, four times the substeps and eight times the iterations -- and the
canon read none of it, so its film lay flat through its filling and every cell still said `pass`.
Its film stiffness is also stated in a unit two readers disagree on (`surface`, below). Which
source is right is a question to measure, not to assume; the setups are how it is measured, and
the canon's default is not changed here.

structure -- what joins the asset's bodies into one object
    none      nothing beyond the schemas: every body meets the floor and nothing else
    asset     the asset's own: seal springs (`rlwrld:sealPairs`), self-contact at the radius and
              margin it authors, its vertex-triangle contact exclusions where the solver takes them
contact -- the contact's stiffness, damping and friction
    derived   the asset's material at its own resolution (`newton_drop.contact_stiffness`),
              critically damped (`contact_damping`), the asset's friction or the shared default
    asset     the numbers the asset authors for its own runtime, handed to the kernel unchanged
    example   the constants of Newton's `cloth/example_cloth_hanging.py`: that scene's, as written
stepping -- how finely a frame is stepped, and how many solver iterations each step takes
    canon     `stepping.SUBSTEPS` x `newton_drop.ITERATIONS` at `stepping.FPS`
    example   Newton's cloth example: 10 substeps x 10 iterations at 60 fps
    asset     the asset's own `rlwrld:simulation:dt_s` and `rlwrld:simulation:iterations`
surface -- how a surface body's `physics:stretchStiffness` / `physics:bendStiffness` are read
    modulus   as moduli, times the shell thickness (t, t^3): Newton 1.5.0's importer, whose comment
              says the proposal authors them in force/area
    stiffness as the stiffness itself: OmniPhysics' definition of the same quantity ("override
              for stretching stiffness; by default derived from youngsModulus and thickness")
    The two differ by 1/t in stretch and 1/t^3 in bending -- 333x and 3.7e7x for a 3 mm film.
    Nothing reads a volume's modulus two ways, so this changes surface bodies only.

Contact damping is stated in N*s/m in every source and handed to the kernel through
`newton_drop.damping_as_the_kernel_reads_it`, as the derived one always was: Newton 1.2.1's contact
law multiplies kd by ke, and its own cloth example writes kd = 1.0 where 1.5.0's writes 1e2 for the
same scene (ke = 1e2) -- one physics, two spellings.

A level that asks the asset for something it does not author refuses the cell -- a setup is a
statement of where a number came from, and a number that came from nowhere is not that setup.
"""
import itertools

import asset_properties

FACTORS = {
    "structure": ("none", "asset"),
    "contact": ("derived", "asset", "example"),
    "stepping": ("canon", "example", "asset"),
    "surface": ("modulus", "stiffness"),
}
NAMES = tuple("-".join(levels) for levels in itertools.product(*FACTORS.values()))
DEFAULT = "none-derived-canon-modulus"

# newton/examples/cloth/example_cloth_hanging.py: fps 60, 10 substeps (its VBD and XPBD branches;
# 32 is its semi-implicit branch), 10 iterations; soft_contact_ke 1e2, kd 1e2 for VBD and 1e0 for
# XPBD, mu 1.0. A metre-scale cloth's constants, carried as what they are. The kd is 1.5.0's, in
# N*s/m: 1.2.1's copy of the same example writes 1.0 under its Rayleigh law (1.0 x ke = 1e2).
EXAMPLE_STEPPING = {"fps": 60.0, "substeps": 10, "iterations": 10}
EXAMPLE_CONTACT = {"vbd": {"ke": 1.0e2, "kd": 1.0e2, "mu": 1.0},
                   "xpbd": {"ke": 1.0e2, "kd": 1.0e0, "mu": 1.0}}


def parse(name):
    """'structure-contact-stepping' -> {factor: level}; an unknown name is an error, not a default."""
    parts = name.split("-")
    if len(parts) != len(FACTORS) or any(p not in levels for p, levels in zip(parts, FACTORS.values())):
        raise SystemExit(f"unknown setup {name!r}; a setup is structure-contact-stepping with "
                         + ", ".join(f"{f} in {{{', '.join(l)}}}" for f, l in FACTORS.items()))
    return dict(zip(FACTORS, parts))


def stepping_of(setup, recipe, canon_fps, canon_substeps, canon_iterations):
    """-> (fps, substeps, iterations, why) for this setup's stepping source."""
    level = setup["stepping"]
    if level == "canon":
        return canon_fps, canon_substeps, canon_iterations, "the canon: stepping.SUBSTEPS x ITERATIONS at stepping.FPS"
    if level == "example":
        e = EXAMPLE_STEPPING
        return e["fps"], e["substeps"], e["iterations"], "Newton's cloth example (example_cloth_hanging.py)"
    dt, dt_source = recipe.get("dt", (None, None))
    iterations, it_source = recipe.get("iterations", (None, None))
    if dt is None or iterations is None:
        raise SystemExit("this setup takes the stepping from the asset, and the asset authors "
                         f"{'no dt' if dt is None else 'no iteration count'} "
                         f"(looked for {', '.join(name for key in ('dt', 'iterations') for name in asset_properties.RECIPE[key])})")
    substeps = 1.0 / (canon_fps * dt)
    if abs(substeps - round(substeps)) > 1e-6:
        raise SystemExit(f"the asset's dt {dt:g} s ({dt_source}) does not divide a 1/{canon_fps:g} s "
                         f"frame into whole substeps ({substeps:.4f}); the frame rate is the experiment's")
    return canon_fps, int(round(substeps)), int(iterations), (
        f"the asset's own: dt {dt:g} s ({dt_source}) is {int(round(substeps))} substeps at "
        f"{canon_fps:g} fps, {int(iterations)} iterations ({it_source})")


def contact_of(setup, solver_name, recipe):
    """-> {ke, kd, mu, shape_ke, why} for the asset or example levels; None for `derived`, which
    the runner computes from the model it built. `kd` is in N*s/m; the runner converts it to what
    the running kernel reads. The asset's own kd carries no unit in its name; it is read as N*s/m,
    the law of the Newton its recipe was written against (>= 1.4, the one whose example uses 1e2),
    and the runner says so."""
    level = setup["contact"]
    if level == "derived":
        return None
    if level == "example":
        e = EXAMPLE_CONTACT[solver_name]
        return {"ke": e["ke"], "kd": e["kd"], "mu": e["mu"], "shape_ke": e["ke"],
                "why": f"Newton's example_cloth_hanging.py constants for {solver_name}, as written"}
    missing = [key for key in ("contact_ke", "contact_kd", "friction") if recipe.get(key, (None,))[0] is None]
    if missing:
        raise SystemExit("this setup takes the contact numbers from the asset, and the asset authors "
                         f"none for {', '.join(missing)} (looked for "
                         + ", ".join(n for k in missing for n in asset_properties.RECIPE[k]) + ")")
    shape_ke = recipe.get("shape_ke", (None,))[0]
    return {"ke": recipe["contact_ke"][0], "kd": recipe["contact_kd"][0], "mu": recipe["friction"][0],
            "shape_ke": recipe["contact_ke"][0] if shape_ke is None else shape_ke,
            "why": "the asset's own runtime numbers (" + ", ".join(
                recipe[k][1] for k in ("contact_ke", "contact_kd", "friction")) + "), handed to the kernel unchanged"}


def self_contact_of(setup, recipe):
    """-> (radius, margin, why) the asset authors, for structure=asset; refuses if it authors none."""
    radius, r_source = recipe.get("self_contact_radius", (None, None))
    margin, m_source = recipe.get("self_contact_margin", (None, None))
    if radius is None or margin is None:
        raise SystemExit("this setup takes the structure from the asset, and the asset authors no "
                         "self-contact radius/margin (looked for "
                         + ", ".join(asset_properties.RECIPE["self_contact_radius"] + asset_properties.RECIPE["self_contact_margin"]) + ")")
    return radius, margin, f"{r_source}, {m_source}"
