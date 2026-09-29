# SPDX-License-Identifier: Apache-2.0
"""A setup: where every number the USD does not settle comes from. Four factors, each a source.

The geometry, the material and the mass are the asset's (`asset_properties`). Everything else a
solver needs has to come from somewhere -- what joins the asset's bodies into one object, the
contact's numbers, how finely a frame is stepped -- and a run that does not say where is a run
nobody can compare. This module names the sources. A setup is one choice per factor, and its name
is the four choices joined with dashes, structure-contact-stepping-surface.

**The canon is `CANON = auto-auto-auto-auto`: everything the USD states, and only the rest ours**
(the user's decision of 2026-09-29, once the polybag's fold exclusions were read and its own
128x80 stepping stopped throwing the bag). Every entry point defaults to it; the other levels
exist to measure a source against another, not to be picked by default.

Why this exists: the polybags carry a recipe for the vendor's own runtime (`rlwrld:*`, see
`asset_properties.RECIPE`) -- seal springs that close the bag, self-contact so its film meets its
filling, its own contact numbers and stepping -- and a run that read none of it had the film lie
flat through its filling while every cell still said `pass`.

structure -- what joins the asset's bodies into one object
    none      nothing beyond the schemas: every body meets the floor and nothing else
    asset     the asset's own: seal springs (`rlwrld:sealPairs`), self-contact at the radius and
              margin it authors, its vertex-triangle and edge-edge contact exclusions
contact -- the contact's stiffness, damping and friction
    derived   the asset's material at its own resolution (`loader.contact_stiffness`),
              critically damped (`contact_damping`), the asset's friction or the shared default
    asset     the numbers the asset authors for its own runtime, handed to the kernel unchanged
    example   the constants of Newton's `cloth/example_cloth_hanging.py`: that scene's, as written
stepping -- how finely a frame is stepped, and how many solver iterations each step takes
    canon     `stepping.SUBSTEPS` x `stepping.ITERATIONS` at `stepping.FPS`
    example   Newton's cloth example: 10 substeps x 10 iterations at 60 fps
    asset     the asset's own `rlwrld:simulation:dt_s` and `rlwrld:simulation:iterations`
surface -- how a surface body's `physics:stretchStiffness` / `physics:bendStiffness` are read
    modulus   as moduli, times the shell thickness (t, t^3): Newton 1.5.0's importer, whose comment
              says the proposal authors them in force/area
    stiffness as the stiffness itself: OmniPhysics' definition of the same quantity ("override
              for stretching stiffness; by default derived from youngsModulus and thickness")
    The two differ by 1/t in stretch and 1/t^3 in bending -- 333x and 3.7e7x for a 3 mm film.
    Nothing reads a volume's modulus two ways, so this changes surface bodies only.

auto, a level of every factor -- the asset's own source where it states one, ours where it does
not (`resolve`): structure -> asset if it authors any joining structure, else none; contact ->
asset if it authors ke, kd and friction, derived if none of them; stepping -> asset if it authors
dt and iterations, canon if neither; surface -> the reading its own newton:triKe/edgeKe agree
with, the importer's where it states none. Half a recipe is refused, not completed.

Contact damping is stated in N*s/m in every source, the unit Newton 1.5.0's VBD contact kernel reads
(`loader.damping_as_the_kernel_reads_it` checks that it still does).

A level that asks the asset for something it does not author refuses the cell -- a setup is a
statement of where a number came from, and a number that came from nowhere is not that setup.
"""
import itertools

import asset_properties

FACTORS = {
    "structure": ("none", "asset", "auto"),
    "contact": ("derived", "asset", "example", "auto"),
    "stepping": ("canon", "example", "asset", "auto"),
    "surface": ("modulus", "stiffness", "auto"),
}
NAMES = tuple("-".join(levels) for levels in itertools.product(*FACTORS.values()))
CANON = "auto-auto-auto-auto"
# What every run before 2026-09-23 ran, when there were no setups: history, for results that
# record none. Not a default of anything.
BEFORE_SETUPS = "none-derived-canon-modulus"

# newton/examples/cloth/example_cloth_hanging.py (Newton 1.5.0), its VBD branch: fps 60, 10
# substeps, 10 iterations; soft_contact_ke 1e2, kd 1e2 N*s/m, mu 1.0. A metre-scale cloth's
# constants, carried as what they are.
EXAMPLE_STEPPING = {"fps": 60.0, "substeps": 10, "iterations": 10}
EXAMPLE_CONTACT = {"ke": 1.0e2, "kd": 1.0e2, "mu": 1.0}


def parse(name):
    """'structure-contact-stepping' -> {factor: level}; an unknown name is an error, not a default."""
    parts = name.split("-")
    if len(parts) != len(FACTORS) or any(p not in levels for p, levels in zip(parts, FACTORS.values())):
        raise SystemExit(f"unknown setup {name!r}; a setup is {'-'.join(FACTORS)} with "
                         + ", ".join(f"{f} in {{{', '.join(l)}}}" for f, l in FACTORS.items()))
    return dict(zip(FACTORS, parts))


def resolve(setup, declared):
    """Replace `auto` in structure, contact and stepping with the level the asset's own authoring
    picks; -> (setup, [why]). `surface: auto` stays: it is decided per body, where the numbers are
    (`usd_deformable.read_surface_stiffness`)."""
    recipe = declared.get("recipe", {})
    authored = {n for names in declared.get("_authored", {}).values() for n in names}
    out, why = dict(setup), []

    def all_or_none(factor, keys, if_all, if_none):
        have = [k for k in keys if k in recipe]
        if len(have) == len(keys):
            why.append(f"{factor}: auto -> {if_all} (the asset authors "
                       + ", ".join(recipe[k][1] for k in keys) + ")")
            return if_all
        if not have:
            why.append(f"{factor}: auto -> {if_none} (the asset authors none of "
                       + ", ".join(n for k in keys for n in asset_properties.RECIPE[k]) + ")")
            return if_none
        missing = [n for k in keys if k not in recipe for n in asset_properties.RECIPE[k]]
        raise SystemExit(f"{factor}: the asset authors part of its own {factor} "
                         f"({', '.join(recipe[k][1] for k in have)}) and not {', '.join(missing)}; "
                         f"half a recipe is refused, not completed with ours")

    if setup["structure"] == "auto":
        joins = sorted(n for n in (asset_properties.SEAL_PAIRS, asset_properties.VERTEX_TRIANGLE_EXCLUSIONS,
                                   asset_properties.EDGE_EXCLUSIONS) if n in authored)
        joins += [recipe[k][1] for k in ("self_contact_radius", "self_contact_margin") if k in recipe]
        out["structure"] = "asset" if joins else "none"
        why.append(f"structure: auto -> {out['structure']} ("
                   + (f"the asset authors {', '.join(joins)}" if joins else "the asset authors no joining structure")
                   + ")")
    if setup["contact"] == "auto":
        out["contact"] = all_or_none("contact", ("contact_ke", "contact_kd", "friction"), "asset", "derived")
    if setup["stepping"] == "auto":
        out["stepping"] = all_or_none("stepping", ("dt", "iterations"), "asset", "canon")
    if setup["surface"] == "auto":
        why.append("surface: auto -> decided per surface body from the asset's own newton:triKe/edgeKe")
    return out, why


def stepping_of(setup, recipe, canon_fps, canon_substeps, canon_iterations):
    """-> (fps, substeps, iterations, why) for this setup's stepping source."""
    level = setup["stepping"]
    if level == "canon":
        return canon_fps, canon_substeps, canon_iterations, "the canon: stepping.SUBSTEPS x stepping.ITERATIONS at stepping.FPS"
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


def contact_of(setup, recipe):
    """-> {ke, kd, mu, shape_ke, why} for the asset or example levels; None for `derived`, which
    the runner computes from the model it built. `kd` is in N*s/m; the runner converts it to what
    the running kernel reads. The asset's own kd carries no unit in its name; it is read as N*s/m,
    the law of the Newton its recipe was written against (>= 1.4, the one whose example uses 1e2),
    and the runner says so."""
    level = setup["contact"]
    if level == "derived":
        return None
    if level == "example":
        e = EXAMPLE_CONTACT
        return {"ke": e["ke"], "kd": e["kd"], "mu": e["mu"], "shape_ke": e["ke"],
                "why": "Newton's example_cloth_hanging.py VBD constants, as written"}
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
