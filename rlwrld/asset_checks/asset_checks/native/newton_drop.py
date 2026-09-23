"""Drop a deformable USD on a ground plane in plain Newton, and report what happens.

    <venv>/bin/python native/newton_drop.py <asset.usda> [--solver vbd|xpbd]
                                               [--fps 60] [--seconds 3] [--drop 0.05]
                                               [--radius auto|<m>] [--usd out.usda]

Built to the shape of Newton's own examples -- `multiphysics/example_rigid_soft_contact.py`
for the contact constants and `softbody/example_softbody_*.py` for the substep loop.
Two things are taken from them literally and matter more than anything else we had
guessed at:

  * `builder.default_particle_radius` is set **before** `add_usd`, which is the documented
    way to size particles (the example writes `builder.default_particle_radius = 0.01`);
  * the soft-contact constants come in a *set* per solver, and every official damping
    value is between 0 and 2e-1. We had been running 1e2.

This exists to separate two questions that kept getting answered together: whether the
asset and solver can simulate at all, and whether Isaac's stage is driving them
correctly. It runs in a few seconds, so the parameters are found here and then applied.
"""
import argparse
import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import warp as wp

import newton
from pxr import Usd

import asset_properties
import integrity
import setups
import stepping
import drop_shape
import usd_deformable
import recording

# Penalty constants, quoted from newton/examples/multiphysics/example_rigid_soft_contact.py. They
# are numerics, not material: the two solvers need different numbers to express the same contact,
# and `ke` without its matching `kd` is a different simulation.
#
# There is no table of contact numbers here. Friction is the asset's, or one stated default shared
# by every solver; the contact's stiffness is the asset's own material at the contact's own scale
# (`contact_stiffness`); its damping is critical for that stiffness and the asset's own mass
# (`contact_damping`). Newton's examples set these per scene -- ke 1e2 for a cloth, 1e5 for a
# soft body, 2e6 for a duck in a gripper -- and each of those numbers is right for its scene and
# wrong for the next asset, which is what a copied constant is.
# The contact spring is critically damped for the typical particle: c = 2 * zeta * sqrt(ke * m).
# It was a constant per solver before (100 for VBD, 1 for XPBD, read off two shipped examples), and
# Newton's own examples use 1, 10 and 100 for scenes of different mass, so the number was a
# property of those scenes. Measured across this canon it was 6x critical on the banana and 1600x
# on the empty polybag; on the loaded polybag's 3 mg film it was 650x, and the film left the
# floor at 345 km/s on the frame it landed -- all five of that asset's Newton cells read
# `diverged` because of it. At critical the film settles (0.06 m/s at 0.33 s).
CONTACT_DAMPING_RATIO = 1.0


def contact_damping(model):
    """-> the contact damping in N*s/m: critical for the median particle mass at this model's
    contact stiffness. The stiffness has to be set first."""
    mass = model.particle_mass.numpy()
    mass = mass[mass > 0.0]
    return 2.0 * CONTACT_DAMPING_RATIO * math.sqrt(float(model.soft_contact_ke) * float(np.median(mass)))


def element_damping_as_the_kernel_reads_it(solver_name):
    """-> convert(kind, value, stiffness): the number this solver's element kernels need so that
    a material damping *is* `value` in Pa.s.

    Newton 1.5.0's VBD scales the elastic force by `rest_volume * damping` and its XPBD uses
    `gamma = k_damp / (stiffness * dt)`: absolute. 1.2.1's VBD multiplies the elastic Hessian by
    `(1.0 + damping * inv_dt)`, a Rayleigh multiplier in seconds, so there the value is divided
    by the element's own stiffness. Read off the kernel source, never a version number.
    """
    module = sys.modules[solver_class(solver_name).__module__]
    package = module.__name__.rsplit(".", 1)[0]
    import importlib
    kernels = importlib.import_module(package + (".particle_vbd_kernels" if solver_name == "vbd" else ".kernels"))
    source = "\n".join(line for line in inspect.getsource(kernels).splitlines()
                       if not line.lstrip().startswith("#"))
    if "rest_volume * damping" in source or "k_damp / (stiffness * dt)" in source:
        return lambda kind, value, stiffness: value
    if "(1.0 + damping * inv_dt)" in source or "hessian * (damping / dt)" in source:
        return lambda kind, value, stiffness: value / stiffness if stiffness > 0.0 else 0.0
    if "materials[tid, 2]" not in source:
        # This solver's element kernels never read the material's damping column (1.2.1's XPBD
        # has the line commented out). The value is carried unchanged, for the record only.
        print(f"[baseline] {solver_name}'s element kernels read no material damping; the asset's is "
              f"carried for the record only")
        return lambda kind, value, stiffness: value
    raise SystemExit(f"{solver_name}'s element kernels read damping in a way this runner does not "
                     f"know; read {kernels.__name__} and say which")


def damping_as_the_kernel_reads_it(value, ke):
    """The number to hand the solver so that the contact damping *is* `value` N*s/m.

    Newton 1.2.1's VBD contact force law multiplies kd by ke (`damping_coeff = kd * ke`: a
    Rayleigh multiplier); 1.4.0 made kd absolute and called it a breaking change. The kernel's own
    source says which, and that is what is read -- never a version number, because the same name
    meaning two things across versions is exactly how `soft_body_relaxation` already bit once.
    XPBD's kernels read neither, so for XPBD the value is a record and changes nothing.
    """
    from newton._src.solvers.vbd import rigid_vbd_kernels
    source = inspect.getsource(rigid_vbd_kernels._compute_body_particle_contact_force)
    if "kd * ke" in source:
        return value / ke
    if "kd / dt" in source:
        return value
    raise SystemExit("this Newton's contact kernel reads kd in a way this runner does not know; "
                     "read _compute_body_particle_contact_force and say which")


def contact_stiffness(model):
    """The contact spring in N/m per particle contact: the asset's material at the contact's scale.

    A particle contact stands in for one element's worth of material -- a patch of the asset's
    own resolution l (two particle radii) across and l deep. A column of a volume that size has
    stiffness E*A/h = E*l, with E from the Lame parameters the model holds
    (E = mu (3 lambda + 2 mu) / (lambda + mu)); a membrane patch resists out of plane through its
    tension E*t, which is `tri_ke` already, so a surface gives tri_ke as it is.
    Both are N/m, both come from the USD, and the stiffest body the contact touches decides,
    because a contact softer than the material gives where the material should. Under a plate
    the N contacts act in parallel (E*A/l) against the body beneath them (E*A/h), h/l ~ 12-20
    here, so the body still gives first.

    This replaces `2 x median(k_mu)`: a Pa quantity times a ratio, stored as N/m. It was the ratio
    of two numbers of different dimension in one shipped example (k_mu 1e6, soft_contact_ke 2e6)
    and on this banana it made the contact 3.4e6 N/m, a thousand times its material at its own
    resolution; `contact_damping`, critical for that stiffness, inherited the error.
    """
    radius = model.particle_radius.numpy()
    candidates = []
    if model.tet_count:
        tets = model.tet_indices.numpy().reshape(-1, 4)
        spacing = 2.0 * radius[tets].mean(axis=1)
        mu, lam = model.tet_materials.numpy()[:, 0], model.tet_materials.numpy()[:, 1]
        young = mu * (3.0 * lam + 2.0 * mu) / np.maximum(lam + mu, 1e-30)
        candidates.append(float((young * spacing).max()))
    if model.tri_count:
        tri_ke = model.tri_materials.numpy()[:, 0]
        if (tri_ke > 0.0).any():
            candidates.append(float(tri_ke[tri_ke > 0.0].max()))
    if not candidates:
        raise SystemExit("this model has no element with a stiffness to set a contact against")
    return max(candidates)


def solver_reads(solver_name, name):
    """Does this solver's own module ever reference `name`? A solver that never mentions
    `tri_materials` has no triangle kernel; one that never mentions `soft_contact_ke` cannot be
    tuned by it. Read from the source, so it stays true on a version nobody has looked at."""
    module = sys.modules[solver_class(solver_name).__module__]
    return name in inspect.getsource(module)


def full_surface_contact(solver_name, wanted):
    """Whether this run generates edge/face soft contacts, not only particle ones.

    They exist where the pipeline offers the switch (Newton 1.4.0+), and only standalone
    SolverVBD consumes them: the 1.5.0 changelog says every other solver rejects them, and
    SolverXPBD raises NotImplementedError on one. That is a capability of the solver, stated in
    one place with its reason, and the run says which it used.
    """
    return bool(wanted and solver_name == "vbd" and _pipeline_takes_full_surface())


# Self-collision is not here: it is a property of the asset, read by
# `asset_properties.self_collision`. Keying it on the element type made a cloth self-collide and a
# soft body not -- a decision about the asset that the asset never asked for, and the opposite of
# what the deformable schema itself defaults to.


def colour_for_vbd(builder, springs=()):
    """Give SolverVBD its colour groups, with the bending edges -- and any springs -- in the graph.

    `builder.color` builds its graph from triangles, tetrahedra and (asked) bending edges, and
    from nothing else: a spring's two particles can land in one colour and then move in the same
    Gauss-Seidel sweep, each blind to the other. Where the run added springs (an asset's seal),
    the graph is built the way `builder.color` builds it, with the springs appended, and coloured
    with Newton's own `color_graph`.

    VBD sweeps one colour at a time and treats the other colours as fixed, so two
    particles joined by a constraint must never share a colour.  `builder.color`
    leaves bending edges out of that graph unless asked -- its own docstring says to
    set `include_bending` "if your model contains bending edges", and Newton's cloth
    examples all do.  Left out, a sheet is stable until something disturbs it and
    then diverges on the frame it lands, for every stiffness, damping and substep
    count.  A tetrahedral body has no bending edges, so this reduces to a plain
    colouring for one.
    """
    bending = len(builder.edge_indices)
    # Always: `builder.color` also colours the rigid bodies (a press plate), which SolverVBD
    # demands whenever a body is present. With springs, the particle groups are then redone
    # below with the springs in the graph; the body groups stay.
    builder.color(include_bending=bending > 0)
    if len(springs) == 0:
        print(f"[baseline] VBD colouring: {bending} bending edge(s) in the graph")
        return
    try:
        from newton._src.sim.graph_coloring import color_graph, construct_particle_graph
    except ImportError as e:
        raise SystemExit(f"this Newton keeps its colouring elsewhere ({e}); springs cannot be coloured")
    tri = np.array(builder.tri_indices, dtype=np.int32) if builder.tri_indices else None
    tri_m = np.array(builder.tri_materials)
    tet = np.array(builder.tet_indices, dtype=np.int32) if builder.tet_indices else None
    tet_m = np.array(builder.tet_materials)
    bend = np.array(builder.edge_indices, dtype=np.int32) if bending else None
    bend_props = np.array(builder.edge_bending_properties) if bending else None
    bend_mask = ((bend_props[:, 0] != 0.0) | (bend_props[:, 1] != 0.0)) if bending else None
    edges = construct_particle_graph(tri, tri_m[:, 0] * tri_m[:, 1] if len(tri_m) else None,
                                     bend, bend_mask, tet, tet_m[:, 0] * tet_m[:, 1] if len(tet_m) else None)
    edges = np.asarray(edges.numpy() if hasattr(edges, "numpy") else edges, dtype=np.int32).reshape(-1, 2)
    edges = np.concatenate([edges, np.asarray(springs, dtype=np.int32).reshape(-1, 2)])
    builder.particle_color_groups = color_graph(builder.particle_count,
                                                wp.array(edges, dtype=wp.int32, device="cpu"))
    colour_of = np.full(builder.particle_count, -1, dtype=np.int64)
    for c, group in enumerate(builder.particle_color_groups):
        colour_of[np.asarray(group, dtype=np.int64)] = c
    same = int((colour_of[np.asarray(springs)[:, 0]] == colour_of[np.asarray(springs)[:, 1]]).sum())
    if same:
        raise SystemExit(f"{same} spring(s) still have both ends in one colour after colouring")
    print(f"[baseline] VBD colouring: {bending} bending edge(s) and {len(springs)} spring(s) in the "
          f"graph, {len(builder.particle_color_groups)} colours")


# There is no separate stiffness for the floor or the plate. Newton's grasping example sets the
# shapes' material to the very same number it sets the soft contact to
# (`shape_material_ke.fill_(self.soft_contact_ke)`), because for a rigid-soft contact VBD reads
# the shape's material -- and a fixture softer than the contact is the softer of the two, so the
# contact gives there instead. Measured: with the contact raised to the asset's own modulus but
# the fixtures left at 2e5, a plate indenting 6.2 mm moved the banana 1.3 mm with 2003 contacts.
# How far out a soft contact is generated, in particle radii. Newton's examples say 0.01 m, but
# they are metre-scale scenes; on a 17 cm banana that margin is a centimetre of empty space
# around every particle, and it forces every other length in the scene -- the press plate has to
# be thicker than it -- to a size that has nothing to do with the asset. Two radii is the
# asset's own resolution, which is what everything else here is derived from.
# How far out a soft contact is generated, in particle radii. Newton's examples say 0.01 m, but
# they are metre-scale scenes; on a 17 cm banana that margin is a centimetre of empty space
# around every particle, and it forces every other length in the scene -- the press plate has to
# be thicker than it -- to a size that has nothing to do with the asset. Two radii is the
# asset's own resolution, which is what everything else here is derived from.
CONTACT_MARGIN_OF_RADIUS = 2.0
# A penalty contact exists only while the particle is inside the band, so the band is the deepest
# indentation the model can represent. Measured on this banana with a band of 1.5 mm, a plate
# driven 10 mm in touched 95 of 3074 particles -- a shell -- and swept through the rest without
# ever meeting them. The experiment says how deep it presses; this is how an engine is made able
# to feel that, which is the engine's side of the bargain and not the experiment's.
#


def contact_margin(radius, substeps, fps, drop, depth=0.0, gravity=9.81):
    """How far out to look for contact: wide enough for everything the experiment asks of it.

    `depth` is the indentation the experiment intends, which the band has to cover or the
    particles past it feel nothing. `drop` is the fall it intends, which sets how far a particle
    moves in one substep.

    The rest offset is the asset's -- half its declared thickness, or its declared particle
    radius -- and it decides where the asset comes to rest. The *detection* band is a property of
    how the scene is being stepped, not of the asset, and tying it to the asset alone is what let
    one solver fall through a floor another solver held.

    A particle arriving from a drop of `drop` is doing sqrt(2*g*drop), and in one substep it
    covers that divided by fps*substeps. If the band is narrower than that stride, the particle
    can be above the floor at one substep and below it at the next with no contact in between:
    measured, the same cloth tunnelled on VBD's 10 substeps (1.65 mm per step against a 1 mm
    band) and did not on XPBD's 32 (0.52 mm). That difference was ours, not the solvers'.
    """
    stride = (2.0 * gravity * max(drop, 0.0)) ** 0.5 / (fps * substeps)
    return max(CONTACT_MARGIN_OF_RADIUS * radius, depth, stride)
ITERATIONS = 10             # every official soft-body example is 5-10
XPBD_MAX_RELAXATION = 0.9   # SolverXPBD's own default; never raise it, only lower it
# What counts as a pass, as fractions of the asset's own size and its own drop.
# "Settled" is judged on the 99th percentile of node speed, not the maximum. A maximum over a
# few thousand nodes is decided by whichever single node is jittering, so an asset that has not
# moved a tenth of a millimetre in a second still reads as moving; the percentile asks whether
# the body is at rest, which is the question.


def solver_elements(model, kinds=None):
    """What each of this asset's bodies is made of, in the order the asset declares them.

    Tetrahedra for a volume, triangles for a surface. Asking the model beats assuming, and it is
    the same question for any asset -- but an asset may be more than one body, and then "the
    elements" is not one array: a loaded polybag is a film of triangles around a filling of
    tetrahedra, both indexed off the one particle array Newton builds. `kinds` is what
    `usd_deformable.bodies` said, so the arrays come back in the order the render meshes do and
    each is bound to its own.

    Without `kinds` this answers for a single body, which is what a caller that has not been told
    about several should get.
    """
    tets = model.tet_indices.numpy() if model.tet_count else None
    tris = model.tri_indices.numpy() if model.tri_count else None
    if kinds is None:
        if tets is not None:
            return [tets]
        if tris is not None:
            return [tris]
        raise SystemExit("the solver built particles but no elements; there is no surface to draw")
    have = {"volume": tets, "surface": tris}
    if len(set(kinds)) != len(kinds):
        raise SystemExit(f"this asset declares two bodies of the same kind ({', '.join(kinds)}), "
                         f"and one element array cannot be split between them by kind alone")
    out = []
    for kind in kinds:
        if have.get(kind) is None:
            raise SystemExit(f"the asset declares a {kind} body but the solver built no "
                             f"{'tetrahedra' if kind == 'volume' else 'triangles'} for it")
        out.append(have[kind])
    return out


def relaxation_is_a_jacobi_factor():
    """Does this Newton's tet kernel spend `soft_body_relaxation` on the correction, or on the
    compliance?

    The name is the same in both versions and the meaning is not. Newton 1.5.0 multiplies the
    position correction by it, exactly as the docstring says, and takes the compliance from the
    material. Newton 1.2.1 has the material lines commented out and assigns
    `stretching_compliance = relaxation` instead -- so lowering it there does not average the
    sweep, it makes the tetrahedra thirty times stiffer, and the banana that 1.5.0 settles
    is the one that kills 1.2.1 with an illegal memory access.

    Asking the kernel which it is beats keeping a table of version numbers: the table is right
    only about the versions someone has already tried.
    """
    import inspect
    from newton._src.solvers.xpbd import kernels
    try:
        source = inspect.getsource(kernels.solve_tetrahedra.func)
    except (OSError, TypeError, AttributeError) as exc:
        raise SystemExit(f"cannot read this Newton's solve_tetrahedra to see what relaxation means: {exc}")
    return "compliance = inv_rest_volume / k_mu" in source


def xpbd_relaxation(tet_count, particle_count):
    """How far to trust one Jacobi sweep of XPBD's tet solve on *this* mesh.

    `apply_particle_deltas` adds every constraint's correction to a particle in full -- it never
    divides by how many constraints touched it. `soft_body_relaxation` is therefore the only
    averaging in the sweep, and a particle shared by n constraints is moved n times too far
    unless it carries the 1/n. A tet contributes two constraints (deviatoric and volume) to each
    of its four particles, so a mesh's own connectivity says what the factor has to be.

    `soft_body_relaxation` reaches only the tet solve (SolverXPBD passes it to `solve_tetrahedra`
    and nothing else), so only tets are counted; a cloth with no tets keeps the default.

    This is why the same solver name behaves so differently on the same asset across Newton
    versions: 1.2.1's tet kernel never multiplies by relaxation at all -- it spends it as a
    constant compliance and ignores the material entirely -- while 1.5.0 applies it to the
    correction and reads the real Lame parameters. On a 12936-tet banana the rule gives 0.03;
    measured, 0.9 and 0.3 diverge, 0.1 and 0.03 settle.
    """
    per_particle = 2.0 * 4.0 * tet_count / max(1, particle_count)
    return min(XPBD_MAX_RELAXATION, 1.0 / per_particle) if per_particle > 0 else XPBD_MAX_RELAXATION


def auto_radius(points):
    """Half the median nearest-neighbour distance: the asset's own resolution, in its own units."""
    picked = np.random.default_rng(0).choice(len(points), size=min(512, len(points)), replace=False)
    distances = np.sqrt(((points[picked][:, None, :] - points[None, :, :]) ** 2).sum(-1))
    distances[distances < 1e-9] = np.inf
    return float(np.median(distances.min(axis=1)) * 0.5)


def _pipeline_takes_full_surface():
    import inspect
    return "enable_rigid_soft_full_surface_contact" in inspect.signature(newton.CollisionPipeline.__init__).parameters


def contact_material(declared, chosen):
    """Friction and restitution for the whole scene: the asset's if it says, ours if it does not.

    Newton's importer leaves its own defaults in place when the asset is silent, which is a
    reasonable thing for it to do and a terrible thing for us to leave unexamined -- the number
    ends up in the results looking like it came from the banana.
    """
    friction, why = asset_properties.friction(declared)
    if declared.get("friction") is None:
        chosen["soft_contact_mu"] = (friction, why)
    restitution = declared.get("restitution")
    if restitution is None:
        restitution = 0.0
        chosen["soft_contact_restitution"] = (restitution, "the asset declares no restitution")
    return friction, restitution


def solver_class(solver_name):
    """The Newton solver this name means. One place, so the runners and the rules agree."""
    return {"vbd": newton.solvers.SolverVBD, "xpbd": newton.solvers.SolverXPBD}[solver_name]


# What Newton's own XPBD cloth example (`cloth/example_cloth_hanging.py`) runs its springs at:
# spring_ke 1e3 on 0.1 kg particles, 10 substeps at 60 fps -> ke*dt^2/m = 1e3 * (1/600)^2 / 0.1.
# Printed beside every membrane's own ratio, as the scale the solver is known to hold.
XPBD_EXAMPLE_SPRING_RATIO = 1.0e3 * (1.0 / 600.0) ** 2 / 0.1


def membrane_as_springs(builder, solver_name, dt, report=None):
    """Give a solver with no triangle kernel the membrane as edge springs, from the same numbers.

    SolverXPBD solves springs, bending edges and tetrahedra, and nothing for a triangle: a cloth
    built for it from `add_cloth_mesh` alone has bending and no in-plane stiffness at all, and the
    asset's stretch stiffness is silently gone. Newton's `example_cloth_hanging.py` gives XPBD
    `add_springs=True`; this is that, as a rule.

    Van Gelder (J. Graphics Tools 3(2), 1998): an edge spring standing in for a membrane of
    stiffness E*t carries k = E*t * (adjacent triangle areas) / length^2. `tri_ke` is E*t (the
    importer has already multiplied the authored stretch stiffness by the thickness). Triangles
    with no stretch stiffness -- the surface of a tetrahedral body -- get no spring.

    The spring's damping is not carried: there is no sourced mapping from a triangle's damping
    to a spring's, and XPBD's own cloth example runs its springs at kd*dt/m = 0.005, near zero.
    A declared triangle damping is reported as dropped for this solver.

    What XPBD can hold is printed with the springs. `solve_springs` applies each spring's full
    correction and `apply_particle_deltas` sums them with no Jacobi relaxation (the tetrahedral
    path has `soft_body_relaxation` for this), so with ~6 springs per particle each moving it by
    r/(2r+1) of the error, r = ke*dt^2/m, the sum overshoots once r is of order one. Newton's
    cloth example runs at r = 0.003; a 3 mm film of 3 mg particles at 1920 Hz is at r = 13, and
    measured, it leaves the scene in its first frame at every substep count up to 384 and with
    any damping. The ratio is printed against the example's so that cell's log says why.
    """
    if solver_reads(solver_name, "tri_materials") or not builder.tri_count:
        return 0
    q = np.asarray(builder.particle_q, dtype=np.float64)
    tris = np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3)
    mats = np.asarray(builder.tri_materials, dtype=np.float64)   # (ke, ka, kd, drag, lift) per triangle
    area = 0.5 * np.linalg.norm(np.cross(q[tris[:, 1]] - q[tris[:, 0]], q[tris[:, 2]] - q[tris[:, 0]]), axis=1)
    weighted_ke = {}
    for t, (i, j, k) in enumerate(tris):
        if mats[t, 0] <= 0.0:
            continue
        for a, b in ((i, j), (j, k), (k, i)):
            edge = (min(a, b), max(a, b))
            weighted_ke[edge] = weighted_ke.get(edge, 0.0) + mats[t, 0] * area[t]
    mass = np.asarray(builder.particle_mass, dtype=np.float64)
    ratios = []
    for (a, b), ke_area in weighted_ke.items():
        length2 = float(((q[a] - q[b]) ** 2).sum())
        if length2 <= 0.0:
            continue
        ke = ke_area / length2
        builder.add_spring(int(a), int(b), ke, 0.0, 0.0)
        ratios.append(ke * dt * dt / max(min(mass[a], mass[b]), 1e-12))
    made = len(weighted_ke)
    if not made:
        return 0
    ratios = np.asarray(ratios)
    print(f"[baseline] membrane as springs: {made} edge spring(s) from {int((mats[:, 0] > 0).sum())} "
          f"triangle(s) with stretch stiffness, because {solver_name} reads no triangle material "
          f"(Van Gelder: k = tri_ke * adjacent area / length^2); damping not carried")
    print(f"[baseline] XPBD spring stiffness ratio ke*dt^2/m: median {np.median(ratios):.3g}, "
          f"max {ratios.max():.3g}; Newton's cloth example runs at {XPBD_EXAMPLE_SPRING_RATIO:.3g}, "
          f"and the Jacobi sum of ~6 springs per particle overshoots once this is of order one")
    if report is not None:
        report["membrane_springs"] = (made, f"{solver_name} has no triangle kernel; the asset's "
                                             f"stretch stiffness carried to edge springs")
        report["spring_ratio_median"] = (float(np.median(ratios)),
                                         f"ke*dt^2/m; {solver_name} holds springs only well below 1 "
                                         f"(its own example: {XPBD_EXAMPLE_SPRING_RATIO:.3g})")
        if (mats[:, 2] > 0.0).any():
            report["tri_kd"] = (float(np.max(mats[:, 2])), f"dropped: {solver_name} reads no triangle "
                                                            f"damping and no spring mapping is sourced")
    return made


def build(asset, solver_name, iterations, radius, drop, margin, full_surface, substeps, fps,
          setup=None):
    """`margin` and `radius` of 0/"auto" mean: take it from the asset, and say where it came from.
    `setup` is `setups.parse(...)`: where the structure and the contact numbers come from."""
    setup = setup or setups.parse(setups.DEFAULT)
    # add_usd stamps `default_particle_radius` onto every particle it imports, so the radius
    # has to be known first. Import once cheaply to measure the asset, then again to build it.
    measure = newton.ModelBuilder()
    measure.add_usd(Usd.Stage.Open(asset))
    # Where this Newton's importer has no path for what the asset declares -- 1.2.1 knows nothing
    # of PhysicsSurfaceDeformableSimAPI, though it ships eight cloth examples -- read the
    # declaration and build it with the engine's own constructor, using 1.5.0's conversion.
    usd_deformable.add_missing(measure, asset)
    if measure.particle_count == 0:
        raise SystemExit(f"{asset}: nothing deformable -- neither this Newton's importer nor its "
                         f"declared schemas produced any particles")
    points = np.asarray(measure.particle_q, dtype=np.float64)
    height = float(points[:, 2].max() - points[:, 2].min())
    declared, chosen = asset_properties.read(asset), {}
    if radius != "auto":
        radius = float(radius)
        chosen["requested_particle_radius"] = (radius, "asked for on the command line")
    elif asset_properties.contact_size(declared)[0] is not None:
        # Newton's importer reads neither `newton:particleRadius` nor a surface's shell thickness,
        # so an asset that spells out its own contact size gets Newton's 0.1 m default instead.
        # One function answers for both kinds, so every engine is given the same number.
        radius = asset_properties.contact_size(declared)[0]
    else:
        radius = auto_radius(points)
        chosen["derived_particle_radius"] = (radius, "the asset declares none; half the median "
                                                      "distance between neighbouring nodes")

    builder = newton.ModelBuilder()
    builder.default_particle_radius = radius
    # The stage has to be held in a name: a traversal of one opened inline outlives the stage
    # itself and the iteration dies on an expired prim.
    stage = Usd.Stage.Open(asset)
    _refusal = usd_deformable.why_not_runnable(stage, asset)   # any number of bodies
    if _refusal:
        raise SystemExit(_refusal)
    builder.add_usd(stage)
    built = usd_deformable.add_missing(builder, asset, chosen)
    # The radius the run uses is the one the builder ended up with, not the one asked for. A
    # volume deformable takes `default_particle_radius`; a cloth's constructor sets its own from
    # the declared shell thickness and ignores it. Reading it back is the only way the contact
    # margin, the plate's size and the landing tolerance are all talking about the same number.
    sizes = usd_deformable.assign_particle_radii(builder, stage, radius, chosen)
    # Before the damping: a Rayleigh kernel's damping is handed over relative to the stiffness.
    usd_deformable.read_surface_stiffness(builder, stage, setup["surface"], declared, chosen)
    usd_deformable.carry_material_damping(builder, stage,
                                          element_damping_as_the_kernel_reads_it(solver_name), chosen)
    recipe = declared["recipe"]
    if setup["stepping"] == "asset":
        asset_properties.consume(declared, *asset_properties.RECIPE["dt"], *asset_properties.RECIPE["iterations"])
    seal, exclusions, bag = structure(builder, stage, setup, declared, chosen, sizes)
    # Every scene length below -- the contact band, the plate, the landing tolerance -- is sized
    # from the coarsest body, so that no body's contact is narrower than its own particles.
    radius = max(sizes.values()) if sizes else float(np.median(np.asarray(builder.particle_radius, dtype=np.float64)))
    if built:
        print(f"[baseline] this Newton's importer did not build {len(built)} of the asset's "
              f"bodies; built from the asset's declaration instead: {built}")
    # Which prim the solver simulates, so the recording can hide the asset's still copy of it.
    # Which prims the solver simulates and what each is, so the recording can hide the
    # asset's still copy of each and bind the right render mesh to the right body.
    simulated = usd_deformable.find(stage)
    kinds = [kind for kind, _ in simulated]
    sim_path = next((str(prim.GetPath()) for _, prim in simulated), None)

    lift = drop - float(points[:, 2].min())          # lowest point starts `drop` above the plane
    q = np.asarray(builder.particle_q, dtype=np.float64)
    q[:, 2] += lift
    builder.particle_q = [wp.vec3(*p) for p in q]
    ground_shape = builder.shape_count
    builder.add_ground_plane()

    # The USD import also brings the asset's render mesh in as a shape with no collision flags:
    # a still copy of the asset sitting at its authored pose, which the viewer would draw next to
    # the one being simulated. Nothing reads it, so it is switched off and said so.
    ghosts = [i for i in range(builder.shape_count)
              if not int(builder.shape_flags[i]) & (int(newton.ShapeFlags.COLLIDE_SHAPES)
                                                    | int(newton.ShapeFlags.COLLIDE_PARTICLES))]
    for i in ghosts:
        builder.shape_flags[i] = 0
    if ghosts:
        print(f"[baseline] hid {len(ghosts)} visual-only shape(s) the USD import added: {ghosts}")

    membrane_as_springs(builder, solver_name, 1.0 / (fps * substeps), chosen)
    elements_report("baseline", builder)
    if solver_name == "vbd":
        colour_for_vbd(builder, seal)
    model = builder.finalize()
    model.soft_contact_ke = contact_stiffness(model)
    chosen["soft_contact_ke"] = (model.soft_contact_ke,
                                 "N/m per contact: the asset's own material at its own resolution "
                                 "(E * 2r for a volume, tri_ke for a membrane), stiffest body; "
                                 + (f"{solver_name} reads it" if solver_reads(solver_name, "soft_contact_ke")
                                    else f"{solver_name} reads no contact stiffness at all"))
    damping = contact_damping(model)
    model.soft_contact_kd = damping_as_the_kernel_reads_it(damping, model.soft_contact_ke)
    chosen["soft_contact_kd"] = (model.soft_contact_kd,
                                 f"{damping:.4g} N*s/m, {CONTACT_DAMPING_RATIO:g}x critical for the "
                                 f"median particle at this contact stiffness, as this kernel reads it")
    friction, restitution = contact_material(declared, chosen)
    model.soft_contact_mu = friction
    model.soft_contact_restitution = restitution
    # Only the floor we added is ours to give a material to. Filling every shape would overwrite
    # whatever the asset's own shapes were imported with.
    for array, value in ((model.shape_material_ke, model.soft_contact_ke),
                         (model.shape_material_kd, model.soft_contact_kd),
                         (model.shape_material_mu, friction)):
        values = array.numpy()
        values[ground_shape] = value
        array.assign(wp.array(values, dtype=float))
    chosen["particle_radius_used"] = (radius, "read back from the model, whatever set it")
    chosen["floor_ke"] = (model.soft_contact_ke, "the floor is as stiff as the contact, because "
                                                 "it is the same contact")
    contact_source("baseline", model, solver_name, setup, recipe, declared, chosen, [ground_shape])
    asset_properties.report("baseline", declared, chosen)

    # Pipeline first, then the solver: SolverVBD sizes its per-body contact state from the
    # contacts that already exist, and Newton's own message says to construct CollisionPipeline
    # before SolverVBD. A static ground survives the wrong order; a rigid body does not.
    margin = margin or contact_margin(radius, substeps, fps, drop)
    print(f"[baseline] soft contact margin {margin * 1000:.2f} mm "
          f"(rest offset {radius * 1000:.2f} mm, {substeps} substeps)")
    kwargs = {"broad_phase": "nxn", "soft_contact_margin": margin}
    if full_surface_contact(solver_name, full_surface):
        # Without it a rigid shape only ever meets the particles, never the surface between them,
        # and the gripper example's docstring says the mesh then slips out of the jaws.
        kwargs["enable_rigid_soft_full_surface_contact"] = True
    pipeline = newton.CollisionPipeline(model, **kwargs)
    print(f"[baseline] full-surface soft contact: {kwargs.get('enable_rigid_soft_full_surface_contact', False)}")

    self_collision, self_radius, self_margin, why = self_contact(setup, recipe, declared, radius, margin)
    print(f"[baseline] self-collision {'on' if self_collision else 'off'} -- {why}")
    if solver_name == "vbd" and not self_collision and not model.tet_count:
        # Context for a cell that dies here. Every cloth example Newton ships that has a ground
        # plane -- bending, franka, hanging, poker_cards, rollers -- enables self-contact, so a
        # sheet run without it is outside anything the engine demonstrates. 1.5.0 handles it; 1.2.1
        # diverges on the first frame. We follow the asset either way and report what happened.
        print(f"[baseline] no cloth example Newton ships runs a sheet over a ground plane with "
              f"self-contact off; if this cell diverges, that is the reason to look at first")
    if solver_name != "vbd":
        print(f"[baseline] SolverXPBD has no particle self-collision switch, so this run has none")

    if solver_name == "vbd":
        solver = newton.solvers.SolverVBD(
            model, iterations=iterations,
            particle_enable_self_contact=self_collision,
            particle_self_contact_radius=self_radius, particle_self_contact_margin=self_margin,
            rigid_body_particle_contact_buffer_size=max(256, model.particle_count),
            **exclusion_kwargs("baseline", exclusions, self_collision))
    else:
        particle_contact_report("baseline", model)
        if relaxation_is_a_jacobi_factor():
            relaxation = xpbd_relaxation(model.tet_count, model.particle_count)
            print(f"[baseline] soft_body_relaxation {relaxation:.4f} from "
                  f"{model.tet_count} tets over {model.particle_count} particles")
        else:
            relaxation = XPBD_MAX_RELAXATION
            print(f"[baseline] soft_body_relaxation left at {relaxation} -- this Newton's tet kernel "
                  f"spends it as the compliance and never reads the material")
        solver = newton.solvers.SolverXPBD(model, iterations=iterations, soft_body_relaxation=relaxation)
    return model, solver, pipeline, radius, lift, sim_path, kinds, margin, bag


# ---------------------------------------------------------------- what a setup adds, shared by the runners
def structure(builder, stage, setup, declared, chosen, sizes):
    """Build what the setup's structure source says joins the bodies; -> (seal pairs added,
    exclusions, bag). Called before anything is lifted: the match is at authored coordinates.

    `bag` is what `integrity` measures in every setup: the authored seal pairs (springs or not),
    each body's particles, and the reach (the coarsest contact size, doubled)."""
    authored_seal = usd_deformable.seal_pairs(builder, stage)
    bodies = [(usd_deformable.builder_index(builder, sim), usd_deformable._world_points(sim))
              for _kind, sim, _render in usd_deformable.bodies(stage)]
    bag = {"seal": authored_seal, "bodies": bodies,
           "reach": 2.0 * (max(sizes.values()) if sizes else float(builder.default_particle_radius))}
    if setup["structure"] != "asset":
        if len(authored_seal):
            print(f"[baseline] the asset authors {len(authored_seal)} seal pair(s); this setup's "
                  f"structure source is '{setup['structure']}', so no seal is built")
        return np.zeros((0, 2), dtype=np.int64), {}, bag
    seal = usd_deformable.add_seal_springs(builder, stage, declared, chosen)
    exclusions = usd_deformable.vertex_triangle_exclusions(builder, stage, declared, chosen)
    asset_properties.consume(declared, *asset_properties.RECIPE["self_contact_radius"],
                             *asset_properties.RECIPE["self_contact_margin"])
    return seal, exclusions, bag


def self_contact(setup, recipe, declared, radius, margin):
    """-> (on, radius, margin, why): the asset's own self-contact under structure=asset, the
    schema's answer otherwise (at the run's contact size and band)."""
    if setup["structure"] == "asset":
        self_radius, self_margin, where = setups.self_contact_of(setup, recipe)
        asset_properties.consume(declared, *asset_properties.RECIPE["self_contact_radius"],
                                 *asset_properties.RECIPE["self_contact_margin"])
        return True, self_radius, self_margin, (f"the asset's own structure: radius "
                                                f"{self_radius * 1e3:.2f} mm, margin {self_margin * 1e3:.2f} mm ({where})")
    on, why = asset_properties.self_collision(declared)
    return on, radius, margin, why


def exclusion_kwargs(tag, exclusions, self_collision):
    """SolverVBD's keyword for the asset's vertex-triangle exclusions, if this Newton has it."""
    if not exclusions:
        return {}
    if not self_collision:
        print(f"[{tag}] contact exclusions are authored but self-contact is off; nothing to exclude")
        return {}
    name = "particle_external_vertex_contact_filtering_map"
    if name not in inspect.signature(newton.solvers.SolverVBD.__init__).parameters:
        raise SystemExit(f"this Newton's SolverVBD takes no {name}; the asset's contact exclusions "
                         f"cannot be applied here")
    print(f"[{tag}] {sum(len(v) for v in exclusions.values())} vertex-triangle exclusion(s) on "
          f"{len(exclusions)} vertices handed to SolverVBD")
    return {name: exclusions}


def contact_source(tag, model, solver_name, setup, recipe, declared, chosen, fixtures):
    """Replace the derived contact numbers with the setup's source, where it is not `derived`."""
    source = setups.contact_of(setup, solver_name, recipe)
    if source is None:
        return
    # Stated in N*s/m; handed over the way this kernel reads it, as the derived damping is.
    kd = damping_as_the_kernel_reads_it(source["kd"], source["ke"])
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = source["ke"], kd, source["mu"]
    for array, value in ((model.shape_material_ke, source["shape_ke"]),
                         (model.shape_material_kd, kd),
                         (model.shape_material_mu, source["mu"])):
        values = array.numpy()
        values[fixtures] = value
        array.assign(wp.array(values, dtype=float))
    for key, value in (("soft_contact_ke", source["ke"]), ("soft_contact_kd", kd),
                       ("soft_contact_mu", source["mu"])):
        chosen[key] = (value, f"{setup['contact']} contact source: {source['why']}"
                       + (f"; {source['kd']:g} N*s/m, as this kernel reads it" if key == "soft_contact_kd" else ""))
    for key in list(chosen):
        if key.endswith("_ke") and key not in ("soft_contact_ke",):
            chosen[key] = (source["shape_ke"], f"the fixtures' stiffness from the same source")
    if setup["contact"] == "asset":
        asset_properties.consume(declared, *(n for k in ("contact_ke", "contact_kd", "shape_ke", "friction")
                                             for n in asset_properties.RECIPE[k]))
    print(f"[{tag}] contact numbers from the {setup['contact']} source: ke {source['ke']:g} "
          f"kd {source['kd']:g} N*s/m (this kernel is handed {kd:g}) mu {source['mu']:g} "
          f"fixtures ke {source['shape_ke']:g}")


def particle_contact_report(tag, model):
    """XPBD meets particles with particles through `particle_*`, which no asset authors and no
    run had ever printed."""
    print(f"[{tag}] particle-particle contact (XPBD): ke {model.particle_ke:g} kd {model.particle_kd:g} "
          f"kf {model.particle_kf:g} mu {model.particle_mu:g} -- Newton's defaults; the asset "
          f"authors none and this run sets none")


def elements_report(tag, builder):
    print(f"[{tag}] elements before finalize: {builder.particle_count} particles, "
          f"{builder.tri_count} triangles, {builder.tet_count} tetrahedra, "
          f"{len(builder.edge_indices)} bending edges, {builder.spring_count} springs")


def resolve_stepping(args):
    """-> (setup, substeps, iterations) from `--setup`, with `--substeps`/`--iterations` allowed
    only over the canon stepping: a number given twice has two owners."""
    setup = setups.parse(args.setup)
    recipe = asset_properties.read(args.asset)["recipe"]
    fps, substeps, iterations, why = setups.stepping_of(setup, recipe, args.fps, stepping.SUBSTEPS, ITERATIONS)
    if fps != args.fps:
        raise SystemExit(f"the {setup['stepping']} stepping is at {fps:g} fps and the run at {args.fps:g}")
    if args.substeps or args.iterations is not None:
        if setup["stepping"] != "canon":
            raise SystemExit(f"--substeps/--iterations and a '{setup['stepping']}' stepping source: "
                             f"two owners for the step; give one")
        substeps = args.substeps or substeps
        iterations = args.iterations if args.iterations is not None else iterations
        why = "the command line"
    print(f"[baseline] setup {args.setup}: {substeps} substeps x {iterations} iterations at "
          f"{fps:g} fps -- {why}")
    return setup, substeps, iterations


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asset")
    ap.add_argument("--solver", default="vbd", choices=("vbd", "xpbd"))
    ap.add_argument("--setup", default=setups.DEFAULT, help="structure-contact-stepping; see setups.py")
    ap.add_argument("--iterations", type=int, default=None, help="default: the setup's")
    ap.add_argument("--substeps", type=int, default=0, help="0 = the setup's")
    ap.add_argument("--fps", type=float, default=stepping.FPS)
    ap.add_argument("--seconds", type=float, required=True,
                help="simulated seconds: the experiment's own SECONDS, which run.py passes")
    ap.add_argument("--radius", default="auto")
    ap.add_argument("--drop", type=float, default=drop_shape.DROP_HEIGHT)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="soft_contact_margin; 0 derives it from the asset's particle radius")
    ap.add_argument("--no-full-surface", action="store_true")
    ap.add_argument("--usd", default=None, help="write an animated USD of the run here")
    args = ap.parse_args()

    setup, substeps, iterations = resolve_stepping(args)
    args.iterations = iterations
    model, solver, pipeline, radius, lift, sim_path, kinds, band, bag = build(
        args.asset, args.solver, args.iterations, args.radius, args.drop, args.margin,
        not args.no_full_surface, substeps, args.fps, setup)
    frames = int(args.seconds * args.fps)
    # The recording is written here rather than by newton.viewer.ViewerUSD: the two Newton
    # versions lay their viewer output out differently, and neither carries the asset's render
    # mesh, so there would be nothing to photograph the textured banana from.
    tape = None
    if args.usd:
        tape = recording.Recording(args.usd, int(args.fps), frames,
                                   np.asarray(model.particle_q.numpy()),
                                   solver_elements(model, kinds), asset=args.asset,
                                   sim_prim_path=sim_path,
                                   ground_half=recording.ground_half(model.particle_q.numpy()))

    state_0, state_1, control = model.state(), model.state(), model.control()
    contacts = pipeline.contacts()
    dt = 1.0 / (args.fps * substeps)

    start = np.asarray(state_0.particle_q.numpy())
    print(f"[baseline] {args.asset.split('/')[-1]} on {args.solver}: {model.particle_count} particles, "
          f"radius {radius * 1000:.2f} mm, lifted {lift * 100:.1f} cm, {args.iterations} iterations x "
          f"{substeps} substeps at {args.fps:g} fps")
    print(f"[baseline] contact ke {model.soft_contact_ke:.4g} kd {model.soft_contact_kd:g}")
    print(f"[baseline] starts z [{start[:, 2].min():.4f}, {start[:, 2].max():.4f}]")
    first_frame = None
    for frame in range(frames):
        # SolverVBD keeps a bounding-volume hierarchy for collision and it does not notice the
        # scene moving on its own: Newton's grasping example rebuilds it once per frame, right
        # before the substep loop, and we never did. The cost of not doing it is invisible until
        # something moves a long way -- measured, a plate held 5 mm inside the banana while only
        # 553 of its 3074 particles were ever in contact, because the tree still described where
        # everything had been at the start.
        if hasattr(solver, "rebuild_bvh"):
            solver.rebuild_bvh(state_0)
        for _ in range(substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        q = np.asarray(state_0.particle_q.numpy())
        if not np.isfinite(q).all():
            print(f"[baseline] diverged at {frame / args.fps:.2f}s")
            return
        if tape is not None:
            tape.frame(frame, q)
        if frame == 0:
            first_frame, said = drop_shape.check_free_fall(
                float(start[:, 2].min() - q[:, 2].min()), args.fps, float(start[:, 2].min()))
            print(f"[baseline] {said}", flush=True)
        if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
            speed = float(np.abs(np.asarray(state_0.particle_qd.numpy())).max())
            print(f"[baseline] t={frame / args.fps:5.2f}s  z [{q[:, 2].min():8.4f}, {q[:, 2].max():8.4f}]  "
                  f"max|v| {speed:8.3f}")
    q = np.asarray(state_0.particle_q.numpy())
    qd = np.asarray(state_0.particle_qd.numpy())
    fell = float(start[:, 2].min() - q[:, 2].min())
    below = drop_shape.below_floor(float(q[:, 2].min()), radius)
    speed = float(np.percentile(np.abs(qd), 99))
    peak = float(np.abs(qd).max())
    height = float(start[:, 2].max() - start[:, 2].min())
    # The verdict and the line it is printed on belong to the experiment, which is why
    # they are asked for rather than written out here: the same words were spelled out
    # in both drop runners, and a pair of copies is a pair waiting to drift.
    decision = drop_shape.verdict(bool(np.isfinite(q).all()), fell,
                                  float(start[:, 2].min()), below, height,
                                  speed, radius, band,
                                  extent=float(q[:, 2].max() - q[:, 2].min()))
    kept = drop_shape.height_kept(float(q[:, 2].max() - q[:, 2].min()), height, radius)
    extra = integrity.result_tokens(q, bag["seal"], bag["bodies"], bag["reach"])
    print(drop_shape.result_line("baseline", fell, float(q[:, 2].min()), float(q[:, 2].max()),
                                 below, speed, peak, decision, kept, first_frame)
          + (" " + extra if extra else ""))
    if tape is not None:
        tape.close()
        print(f"[baseline] wrote {args.usd}")


if __name__ == "__main__":
    wp.config.quiet = True
    main()
