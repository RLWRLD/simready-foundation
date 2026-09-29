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
import pace
import setups
import stepping
import drop_shape
import usd_deformable
import recording
import loader
# Moved to loader.py (the one loading path); re-exported for the modules that import them from here.
from loader import (CONTACT_DAMPING_RATIO, contact_damping, element_damping_as_the_kernel_reads_it,  # noqa: F401
                    damping_as_the_kernel_reads_it, contact_stiffness, solver_reads, colour_for_vbd,
                    auto_radius, contact_material, solver_class, structure, self_contact,
                    exclusion_kwargs, contact_source, particle_contact_report, elements_report)


def full_surface_contact(solver_name, wanted):
    """Whether this run generates edge/face soft contacts, not only particle ones.

    They exist where the pipeline offers the switch (Newton 1.4.0+), and only standalone
    SolverVBD consumes them: the 1.5.0 changelog says every other solver rejects them, and
    SolverXPBD raises NotImplementedError on one. That is a capability of the solver, stated in
    one place with its reason, and the run says which it used.
    """
    return bool(wanted and solver_name == "vbd" and _pipeline_takes_full_surface())


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


def _pipeline_takes_full_surface():
    import inspect
    return "enable_rigid_soft_full_surface_contact" in inspect.signature(newton.CollisionPipeline.__init__).parameters


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
    asset_ = loader.load(asset, solver_name, setup, radius, tag="baseline")
    builder, points, radius = asset_.builder, asset_.points, asset_.radius
    sim_path, kinds, bag = asset_.sim_path, asset_.kinds, asset_.bag

    lift = drop - float(points[:, 2].min())          # lowest point starts `drop` above the plane
    q = np.asarray(builder.particle_q, dtype=np.float64)
    q[:, 2] += lift
    builder.particle_q = [wp.vec3(*p) for p in q]
    ground_shape = builder.shape_count
    builder.add_ground_plane()

    membrane_as_springs(builder, solver_name, 1.0 / (fps * substeps), asset_.chosen)
    elements_report("baseline", builder)
    loader.colour(asset_)
    model = builder.finalize()
    loader.configure(asset_, model, fixtures=[ground_shape])

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

    if solver_name == "vbd":
        solver = loader.vbd_solver(asset_, model, iterations, margin)
    else:
        print(f"[baseline] SolverXPBD has no particle self-collision switch, so this run has none")
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


def resolve_stepping(args):
    """-> (setup, substeps, iterations) from `--setup`, with `--substeps`/`--iterations` allowed
    only over the canon stepping: a number given twice has two owners."""
    setup = setups.parse(args.setup)
    declared = asset_properties.read(args.asset)
    setup, resolved = setups.resolve(setup, declared)
    for line in resolved:
        print(f"[baseline] setup {args.setup}: {line}")
    recipe = declared["recipe"]
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
    clock = pace.Pace("baseline", args.fps)

    def substep_loop():
        nonlocal state_0, state_1
        for _ in range(substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    # Newton's examples record the substep loop once as a CUDA graph and replay it every frame
    # (e.g. multiphysics/example_softbody_dropping_to_cloth.py); stepped from Python instead, each
    # substep's dozens of kernel launches cost more than the kernels on a small asset, and the
    # pace a run reports would be this runner's, not the engine's. Replaying needs the state
    # buffers back where they started after one frame, i.e. an even number of swaps.
    graph = None
    if wp.get_device().is_cuda and substeps % 2 == 0:
        with wp.ScopedCapture() as capture:
            substep_loop()
        graph = capture.graph
        print(f"[baseline] substep loop captured as a CUDA graph ({substeps} substeps), replayed per frame")
    else:
        print(f"[baseline] substep loop stepped from Python: "
              + ("not a CUDA device" if not wp.get_device().is_cuda else f"{substeps} substeps is odd, "
                 "so the state buffers end a frame swapped and a replay would read the wrong one"))
    for frame in range(frames):
        # SolverVBD keeps a bounding-volume hierarchy for collision and it does not notice the
        # scene moving on its own: Newton's grasping example rebuilds it once per frame, right
        # before the substep loop, and we never did. The cost of not doing it is invisible until
        # something moves a long way -- measured, a plate held 5 mm inside the banana while only
        # 553 of its 3074 particles were ever in contact, because the tree still described where
        # everything had been at the start.
        clock.start(frame)
        if hasattr(solver, "rebuild_bvh"):
            solver.rebuild_bvh(state_0)
        if graph is not None:
            wp.capture_launch(graph)
        else:
            substep_loop()
        clock.stop()
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
          + (" " + extra if extra else "") + (" " + clock.tokens() if clock.tokens() else ""))
    if tape is not None:
        tape.close()
        print(f"[baseline] wrote {args.usd}")


if __name__ == "__main__":
    wp.config.quiet = True
    main()
