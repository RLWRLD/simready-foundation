"""Load a deformable USD into a Newton builder the way the asset says -- everything Newton's USD
importer leaves out, and nothing the asset does not say.

    import loader
    asset = loader.load("fruit.usdz", solver_name="vbd")        # -> loader.Loaded
    ... add the scene's own fixtures (ground, plate, gripper) to asset.builder ...
    loader.colour(asset)                                         # VBD only, before finalize
    model = asset.builder.finalize()
    loader.configure(asset, model, fixtures=[ground_shape])      # contact numbers, and the audit
    solver = loader.vbd_solver(asset, model, iterations, margin)

Newton's importer (1.2.1 and 1.5.0) builds the meshes and reads the moduli, and silently drops
the rest: `newton:particleRadius` (every particle gets the builder's 0.1 m default), a volume's
`newton:kDamp` and a membrane's `newton:triKd`/`edgeKd` (damping 0), and everything a vendor
authors for its own runtime (seal springs, contact exclusions, self-contact radius and margin,
contact numbers). `load` carries each of them, in the unit the running kernel reads, and prints
one line per value saying where it came from. `configure` then refuses the run if the asset
authors anything nobody read (`asset_properties.account`).

The two experiment runners (newton_drop, newton_press) build through this module; there is no
second loading path.
"""
import inspect
import math
import types

import numpy as np
import warp as wp

import newton
from pxr import Usd

import asset_properties
import setups
import usd_deformable


# There is no table of contact numbers here. Friction is the asset's, or one stated default shared
# by every solver; the contact's stiffness is the asset's own material at the contact's own scale
# (`contact_stiffness`); its damping is critical for that stiffness and the asset's own mass
# (`contact_damping`). Newton's examples set these per scene -- ke 1e2 for a cloth, 1e5 for a
# soft body, 2e6 for a duck in a gripper -- and each of those numbers is right for its scene and
# wrong for the next asset, which is what a copied constant is.
# The contact spring is critically damped for the typical particle: c = 2 * zeta * sqrt(ke * m).
# It was a constant before (100, read off a shipped example), and Newton's own examples use 1, 10
# and 100 for scenes of different mass, so the number was a
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

def element_damping_as_the_kernel_reads_it():
    """-> convert(kind, value, stiffness): the number SolverVBD's element kernels need so that a
    material damping *is* `value` in Pa.s. Newton 1.5.0's VBD scales the elastic force by
    `rest_volume * damping`, an absolute damping, so the value is handed over as it is. Read off
    the kernel source: a Newton whose kernel says otherwise (1.2.1's was a Rayleigh multiplier,
    `(1.0 + damping * inv_dt)`) is refused rather than handed a number in the wrong unit."""
    from newton._src.solvers.vbd import particle_vbd_kernels
    source = "\n".join(line for line in inspect.getsource(particle_vbd_kernels).splitlines()
                       if not line.lstrip().startswith("#"))
    if "rest_volume * damping" not in source:
        raise SystemExit("this Newton's VBD element kernels do not read damping as Newton 1.5.0's do "
                         "(rest_volume * damping); this package is for Newton 1.5")
    return lambda kind, value, stiffness: value

def damping_as_the_kernel_reads_it(value, ke):
    """The number to hand SolverVBD so that the contact damping *is* `value` N*s/m. Newton 1.5.0's
    contact law takes kd absolute (`kd / dt`); 1.2.1's multiplied it by ke. Read off the kernel's
    source, and refused if it is not 1.5.0's -- the same name meaning two things across versions."""
    from newton._src.solvers.vbd import rigid_vbd_kernels
    source = inspect.getsource(rigid_vbd_kernels._compute_body_particle_contact_force)
    if "kd / dt" not in source or "kd * ke" in source:
        raise SystemExit("this Newton's contact kernel does not read kd as Newton 1.5.0's does "
                         "(kd / dt); this package is for Newton 1.5")
    return value

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
    and on the banana (measured 2026-09) it made the contact 3.4e6 N/m, a thousand times its material at its own
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

def auto_radius(points):
    """Half the median nearest-neighbour distance: the asset's own resolution, in its own units."""
    picked = np.random.default_rng(0).choice(len(points), size=min(512, len(points)), replace=False)
    distances = np.sqrt(((points[picked][:, None, :] - points[None, :, :]) ** 2).sum(-1))
    distances[distances < 1e-9] = np.inf
    return float(np.median(distances.min(axis=1)) * 0.5)

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


# ---------------------------------------------------------------- what a setup adds, shared by the runners
def structure(builder, stage, setup, declared, chosen, sizes):
    """Build what the setup's structure source says joins the bodies; -> (seal pairs added,
    exclusions (vertex-triangle, edge-edge), bag). Called before anything is lifted: the match
    is at authored coordinates.

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
        return np.zeros((0, 2), dtype=np.int64), ({}, {}), bag
    seal = usd_deformable.add_seal_springs(builder, stage, declared, chosen)
    exclusions = (usd_deformable.vertex_triangle_exclusions(builder, stage, declared, chosen),
                  usd_deformable.edge_edge_exclusions(builder, stage, declared, chosen))
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
    """SolverVBD's keywords for the asset's (vertex-triangle, edge-edge) exclusions; refuses if
    this Newton has no such keyword."""
    kwargs = {}
    for what, name, found in (
            ("vertex-triangle", "particle_external_vertex_contact_filtering_map", exclusions[0]),
            ("edge-edge", "particle_external_edge_contact_filtering_map", exclusions[1])):
        if not found:
            continue
        if not self_collision:
            print(f"[{tag}] {what} contact exclusions are authored but self-contact is off; nothing to exclude")
            continue
        if name not in inspect.signature(newton.solvers.SolverVBD.__init__).parameters:
            raise SystemExit(f"this Newton's SolverVBD takes no {name}; the asset's {what} contact "
                             f"exclusions cannot be applied here")
        print(f"[{tag}] {sum(len(v) for v in found.values())} {what} exclusion(s) on "
              f"{len(found)} primitives handed to SolverVBD")
        kwargs[name] = found
    return kwargs

def contact_source(tag, model, setup, recipe, declared, chosen, fixtures):
    """Replace the derived contact numbers with the setup's source, where it is not `derived`."""
    source = setups.contact_of(setup, recipe)
    if source is None:
        return
    # Stated in N*s/m; handed over the way this kernel reads it, as the derived damping is.
    kd = damping_as_the_kernel_reads_it(source["kd"], source["ke"])
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = source["ke"], kd, source["mu"]
    # The fixtures' own material: the source's where it states one, Newton's default shape
    # material where it does not -- never the particles' numbers, which are a different contact.
    default = newton.ModelBuilder.ShapeConfig()
    fixture = {}
    for name, stated, fallback in (("ke", source["shape_ke"], default.ke), ("kd", source["shape_kd"], default.kd),
                                   ("mu", source["shape_mu"], default.mu)):
        fixture[name] = (stated, "the source") if stated is not None else (fallback, "Newton's default shape material")
    fixture_kd = damping_as_the_kernel_reads_it(fixture["kd"][0], fixture["ke"][0])
    for array, value in ((model.shape_material_ke, fixture["ke"][0]),
                         (model.shape_material_kd, fixture_kd),
                         (model.shape_material_mu, fixture["mu"][0])):
        values = array.numpy()
        values[fixtures] = value
        array.assign(wp.array(values, dtype=float))
    for key, value in (("soft_contact_ke", source["ke"]), ("soft_contact_kd", kd),
                       ("soft_contact_mu", source["mu"])):
        chosen[key] = (value, f"{setup['contact']} contact source: {source['why']}"
                       + (f"; {source['kd']:g} N*s/m, as this kernel reads it" if key == "soft_contact_kd" else ""))
    for name in ("ke", "kd", "mu"):
        chosen[f"fixture_{name}"] = (fixture[name][0], f"the scene's fixtures, from {fixture[name][1]}")
    if setup["contact"] == "asset":
        asset_properties.consume(declared, *(n for k in ("contact_ke", "contact_kd", "shape_ke", "shape_kd",
                                                         "shape_mu", "friction")
                                             for n in asset_properties.RECIPE[k]))
    print(f"[{tag}] contact numbers from the {setup['contact']} source: ke {source['ke']:g} "
          f"kd {source['kd']:g} N*s/m (this kernel is handed {kd:g}) mu {source['mu']:g}; fixtures ke "
          f"{fixture['ke'][0]:g} kd {fixture['kd'][0]:g} mu {fixture['mu'][0]:g}")


def elements_report(tag, builder):
    print(f"[{tag}] elements before finalize: {builder.particle_count} particles, "
          f"{builder.tri_count} triangles, {builder.tet_count} tetrahedra, "
          f"{len(builder.edge_indices)} bending edges, {builder.spring_count} springs")


def load(asset, setup=None, radius="auto", tag="load"):
    """Build `asset` into a fresh ModelBuilder with everything it declares; -> Loaded.

    `setup` (a parsed native/setups.py name) says which sources to use where the asset states
    them; the default is setups.CANON, everything the asset states. `radius` overrides the
    particle radius only when the asset states none, or when a number is given explicitly."""
    setup = setup or setups.resolve(setups.parse(setups.CANON), asset_properties.read(asset))[0]
    # add_usd stamps `default_particle_radius` onto every particle it imports, so the radius
    # has to be known first. Import once cheaply to measure the asset, then again to build it.
    measure = newton.ModelBuilder()
    measure.add_usd(Usd.Stage.Open(asset))
    if measure.particle_count == 0:
        raise SystemExit(f"{asset}: nothing deformable -- this Newton's importer produced no particles")
    points = np.asarray(measure.particle_q, dtype=np.float64)
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
    refusal = usd_deformable.why_not_runnable(stage, asset)   # any number of bodies
    if refusal:
        raise SystemExit(refusal)
    builder.add_usd(stage)
    usd_deformable.check_imported(builder, stage)             # every declared body, point for point
    # The radius the run uses is the one the builder ended up with, not the one asked for. A
    # volume deformable takes `default_particle_radius`; a cloth's constructor sets its own from
    # the declared shell thickness and ignores it.
    sizes = usd_deformable.assign_particle_radii(builder, stage, radius, chosen)
    # Before the damping: a Rayleigh kernel's damping is handed over relative to the stiffness.
    usd_deformable.read_surface_stiffness(builder, stage, setup["surface"], declared, chosen)
    usd_deformable.carry_material_damping(builder, stage,
                                          element_damping_as_the_kernel_reads_it(), chosen, declared)
    if setup["stepping"] == "asset":
        asset_properties.consume(declared, *asset_properties.RECIPE["dt"], *asset_properties.RECIPE["iterations"])
    seal, exclusions, bag = structure(builder, stage, setup, declared, chosen, sizes)
    # Every scene length -- a contact band, a plate, a landing tolerance -- is sized from the
    # coarsest body, so that no body's contact is narrower than its own particles.
    radius = max(sizes.values()) if sizes else float(np.median(np.asarray(builder.particle_radius, dtype=np.float64)))
    # Which prims the solver simulates and what each is, so a recording can hide the asset's
    # still copy of each and bind the right render mesh to the right body.
    simulated = usd_deformable.find(stage)
    # The USD import also brings the asset's render mesh in as a shape with no collision flags:
    # a still copy of the asset at its authored pose. Nothing reads it, so it is switched off.
    ghosts = [i for i in range(builder.shape_count)
              if not int(builder.shape_flags[i]) & (int(newton.ShapeFlags.COLLIDE_SHAPES)
                                                    | int(newton.ShapeFlags.COLLIDE_PARTICLES))]
    for i in ghosts:
        builder.shape_flags[i] = 0
    if ghosts:
        print(f"[{tag}] hid {len(ghosts)} visual-only shape(s) the USD import added: {ghosts}")
    return Loaded(asset=asset, tag=tag, setup=setup, builder=builder,
                  stage=stage, declared=declared, chosen=chosen, points=points, radius=radius,
                  sizes=sizes, seal=seal, exclusions=exclusions, bag=bag,
                  kinds=[kind for kind, _ in simulated],
                  sim_path=next((str(prim.GetPath()) for _, prim in simulated), None))


class Loaded(types.SimpleNamespace):
    """What `load` built: the builder (not finalized -- the scene adds its fixtures), the stage,
    what the asset declared and what we chose (`declared`, `chosen`), the particle radius every
    scene length should be sized from (`radius`, the coarsest body's), the seal springs and
    contact exclusions it added, and what `integrity` measures (`bag`)."""


def colour(asset):
    """SolverVBD's colouring, with the seal springs in the graph. Call after the scene's fixtures
    are added and before finalize."""
    colour_for_vbd(asset.builder, asset.seal)


def configure(asset, model, fixtures):
    """Give the finalized model the asset's contact: stiffness and damping from its own material
    (or the recipe the setup takes), its friction, the same numbers on the scene's `fixtures`
    (shape indices the scene added); then print the audit and refuse what nobody read."""
    tag, chosen, declared = asset.tag, asset.chosen, asset.declared
    model.soft_contact_ke = contact_stiffness(model)
    chosen["soft_contact_ke"] = (model.soft_contact_ke,
                                 "N/m per contact: the asset's own material at its own resolution "
                                 "(E * 2r for a volume, tri_ke for a membrane), stiffest body")
    damping = contact_damping(model)
    model.soft_contact_kd = damping_as_the_kernel_reads_it(damping, model.soft_contact_ke)
    chosen["soft_contact_kd"] = (model.soft_contact_kd,
                                 f"{damping:.4g} N*s/m, {CONTACT_DAMPING_RATIO:g}x critical for the "
                                 f"median particle at this contact stiffness, as this kernel reads it")
    friction, restitution = contact_material(declared, chosen)
    model.soft_contact_mu = friction
    model.soft_contact_restitution = restitution
    # Only the scene's fixtures are ours to give a material to. Filling every shape would
    # overwrite whatever the asset's own shapes were imported with.
    for array, value in ((model.shape_material_ke, model.soft_contact_ke),
                         (model.shape_material_kd, model.soft_contact_kd),
                         (model.shape_material_mu, friction)):
        values = array.numpy()
        values[list(fixtures)] = value
        array.assign(wp.array(values, dtype=float))
    chosen["particle_radius_used"] = (asset.radius, "read back from the model, whatever set it")
    chosen["fixture_ke"] = (model.soft_contact_ke, "the scene's fixtures are as stiff as the contact, "
                                                   "because it is the same contact")
    contact_source(tag, model, asset.setup, declared["recipe"], declared, chosen, list(fixtures))
    # The recipe's contact buffers are applied by `vbd_solver`, which the scene calls next.
    for key in ("rigid_particle_buffer", "rigid_buffer"):
        if declared["recipe"].get(key):
            asset_properties.consume(declared, *asset_properties.RECIPE[key])
    asset_properties.report(tag, declared, chosen, asset.setup)


def vbd_solver(asset, model, iterations, margin):
    """SolverVBD with the asset's self-contact and contact exclusions. `margin` is the scene's
    soft-contact band, which self-contact uses where the asset states none."""
    tag = asset.tag
    self_collision, self_radius, self_margin, why = self_contact(asset.setup, asset.declared["recipe"],
                                                                 asset.declared, asset.radius, margin)
    print(f"[{tag}] self-collision {'on' if self_collision else 'off'} -- {why}")
    if not self_collision and not model.tet_count:
        # Context for a cell that dies here. Every cloth example Newton ships that has a ground
        # plane -- bending, franka, hanging, poker_cards, rollers -- enables self-contact, so a
        # sheet run without it is outside anything the engine demonstrates. 1.5.0 handles it; 1.2.1
        # diverges on the first frame. We follow the asset either way and report what happened.
        print(f"[{tag}] no cloth example Newton ships runs a sheet over a ground plane with "
              f"self-contact off; if this cell diverges, that is the reason to look at first")
    # The contact buffers: the asset's recipe where it sizes them. Otherwise every particle could
    # touch a fixture at once, and the per-body list (256 by default, documented as never resizing)
    # would drop what did not fit without a word, so it is sized to the particle count.
    recipe, buffers = asset.declared["recipe"], {}
    stated = recipe.get("rigid_particle_buffer")
    buffers["rigid_body_particle_contact_buffer_size"] = int(stated[0]) if stated else max(256, model.particle_count)
    if recipe.get("rigid_buffer"):
        buffers["rigid_body_contact_buffer_size"] = int(recipe["rigid_buffer"][0])
    print(f"[{tag}] contact buffers: " + ", ".join(f"{k} {v}" for k, v in buffers.items())
          + (" (the asset's recipe)" if stated else " (sized to the particle count)"))
    return newton.solvers.SolverVBD(
        model, iterations=iterations,
        particle_enable_self_contact=self_collision,
        particle_self_contact_radius=self_radius, particle_self_contact_margin=self_margin,
        **buffers, **exclusion_kwargs(tag, asset.exclusions, self_collision))
