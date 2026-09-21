"""Drop a deformable USD on a ground plane in plain Newton, and report what happens.

    <venv>/bin/python tools/newton_baseline.py <asset.usda> [--solver vbd|xpbd]
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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import warp as wp

import newton
from pxr import Usd

import asset_properties
import usd_deformable
import recording

# Penalty constants, quoted from newton/examples/multiphysics/example_rigid_soft_contact.py. They
# are numerics, not material: the two solvers need different numbers to express the same contact,
# and `ke` without its matching `kd` is a different simulation.
#
# `mu` is deliberately NOT here. Friction is a property of the materials in the scene, and giving
# VBD 0.3 and XPBD 1.0 because two examples happened to use those would mean the two solvers were
# never rubbing the same banana on the same floor. It comes from the asset, or from one stated
# default shared by every solver.
CONTACT = {
    "xpbd": {"soft_contact_ke": 75.0, "soft_contact_kd": 1.0, "soft_contact_kf": 1.0e3},
    "vbd": {"soft_contact_ke": 1.0e5, "soft_contact_kd": 1.0e-4, "soft_contact_kf": 1.0e3},
}
GROUND_CONTACT_KE = 2.0e5   # example_rigid_soft_contact.GROUND_CONTACT_KE
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
SUBSTEPS = {"xpbd": 32, "vbd": 10}   # rigid_soft_contact uses 32; the softbody examples use 10
ITERATIONS = 10             # every official soft-body example is 5-10
XPBD_MAX_RELAXATION = 0.9   # SolverXPBD's own default; never raise it, only lower it
# What counts as a pass, as fractions of the asset's own size and its own drop.
MIN_FALL_OF_DROP = 0.5
TUNNEL_DEPTH_OF_HEIGHT = 0.05
SETTLED_SPEED_OF_HEIGHT = 2.0   # of the asset's height per second
# "Settled" is judged on the 99th percentile of node speed, not the maximum. A maximum over a
# few thousand nodes is decided by whichever single node is jittering, so an asset that has not
# moved a tenth of a millimetre in a second still reads as moving; the percentile asks whether
# the body is at rest, which is the question.


def solver_elements(model):
    """The elements this model is actually made of: tetrahedra for a soft body, triangles for a
    cloth. Asking the model beats assuming, and it is the same question for any asset."""
    if model.tet_count:
        return model.tet_indices.numpy()
    if model.tri_count:
        return model.tri_indices.numpy()
    raise SystemExit("the solver built particles but no elements; there is no surface to draw")


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
    friction = declared.get("friction")
    if friction is None:
        friction = asset_properties.DEFAULT_FRICTION
        chosen["soft_contact_mu"] = (friction, "the asset declares no friction; one value for every "
                                               "solver so they rub the same floor")
    restitution = declared.get("restitution")
    if restitution is None:
        restitution = 0.0
        chosen["soft_contact_restitution"] = (restitution, "the asset declares no restitution")
    return friction, restitution


def build(asset, solver_name, iterations, radius, drop, margin, full_surface):
    """`margin` and `radius` of 0/"auto" mean: take it from the asset, and say where it came from."""
    """`margin` of 0 means: take it from the asset."""
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
    declared, chosen = asset_properties.read(asset), {}
    if radius != "auto":
        radius = float(radius)
        chosen["requested_particle_radius"] = (radius, "asked for on the command line")
    elif declared.get("particle_radius") is not None:
        # Newton's importer does not read this attribute, so an asset that spells out its own
        # particle size gets Newton's 0.1 m default instead. Read it here or it is simply lost.
        radius = declared["particle_radius"]
    else:
        radius = auto_radius(points)
        chosen["derived_particle_radius"] = (radius, "the asset declares none; half the median "
                                                      "distance between neighbouring nodes")

    builder = newton.ModelBuilder()
    builder.default_particle_radius = radius
    # The stage has to be held in a name: a traversal of one opened inline outlives the stage
    # itself and the iteration dies on an expired prim.
    stage = Usd.Stage.Open(asset)
    builder.add_usd(stage)
    built = usd_deformable.add_missing(builder, asset, chosen)
    # The radius the run uses is the one the builder ended up with, not the one asked for. A
    # volume deformable takes `default_particle_radius`; a cloth's constructor sets its own from
    # the declared shell thickness and ignores it. Reading it back is the only way the contact
    # margin, the plate's size and the landing tolerance are all talking about the same number.
    radius = float(np.median(np.asarray(builder.particle_radius, dtype=np.float64)))
    if built:
        print(f"[baseline] this Newton's importer produced nothing; built from the asset's "
              f"declaration instead: {built}")
    # Which prim the solver simulates, so the recording can hide the asset's still copy of it.
    sim_path = next((str(prim.GetPath()) for _, prim in usd_deformable.find(stage)), None)

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

    if solver_name == "vbd":
        builder.color()  # SolverVBD refuses a model without particle colour groups
    model = builder.finalize()
    for name, value in CONTACT[solver_name].items():
        setattr(model, name, value)
    for name in CONTACT[solver_name]:
        chosen[name] = (CONTACT[solver_name][name],
                        f"penalty numerics for {solver_name}, from Newton's example_rigid_soft_contact.py")
    friction, restitution = contact_material(declared, chosen)
    model.soft_contact_mu = friction
    model.soft_contact_restitution = restitution
    # Only the floor we added is ours to give a material to. Filling every shape would overwrite
    # whatever the asset's own shapes were imported with.
    for array, value in ((model.shape_material_ke, GROUND_CONTACT_KE),
                         (model.shape_material_kd, CONTACT[solver_name]["soft_contact_kd"]),
                         (model.shape_material_mu, friction)):
        values = array.numpy()
        values[ground_shape] = value
        array.assign(wp.array(values, dtype=float))
    chosen["particle_radius_used"] = (radius, "read back from the model, whatever set it")
    chosen["ground_ke"] = (GROUND_CONTACT_KE, "the floor is the experiment's, not the asset's")
    asset_properties.report("baseline", declared, chosen)

    # Pipeline first, then the solver: SolverVBD sizes its per-body contact state from the
    # contacts that already exist, and Newton's own message says to construct CollisionPipeline
    # before SolverVBD. A static ground survives the wrong order; a rigid body does not.
    margin = margin or radius * CONTACT_MARGIN_OF_RADIUS
    print(f"[baseline] soft contact margin {margin * 1000:.2f} mm")
    margin = margin or radius * CONTACT_MARGIN_OF_RADIUS
    print(f"[baseline] soft contact margin {margin * 1000:.2f} mm")
    kwargs = {"broad_phase": "nxn", "soft_contact_margin": margin}
    if full_surface and solver_name == "vbd" and _pipeline_takes_full_surface():
        # Newton 1.5, and VBD only: SolverXPBD raises NotImplementedError on an edge/face soft
        # contact, saying so in as many words. Without the flag a rigid shape only ever meets the
        # particles, never the surface between them, and the gripper example's docstring says the
        # mesh then slips out of the jaws. It is a capability of this engine+solver pair, so it is
        # on where it exists and reported where it does not.
        kwargs["enable_rigid_soft_full_surface_contact"] = True
    pipeline = newton.CollisionPipeline(model, **kwargs)
    print(f"[baseline] full-surface soft contact: {kwargs.get('enable_rigid_soft_full_surface_contact', False)}")

    if solver_name == "vbd":
        solver = newton.solvers.SolverVBD(model, iterations=iterations, particle_enable_self_contact=False,
                                          rigid_body_particle_contact_buffer_size=max(256, model.particle_count))
    else:
        if relaxation_is_a_jacobi_factor():
            relaxation = xpbd_relaxation(model.tet_count, model.particle_count)
            print(f"[baseline] soft_body_relaxation {relaxation:.4f} from "
                  f"{model.tet_count} tets over {model.particle_count} particles")
        else:
            relaxation = XPBD_MAX_RELAXATION
            print(f"[baseline] soft_body_relaxation left at {relaxation} -- this Newton's tet kernel "
                  f"spends it as the compliance and never reads the material")
        solver = newton.solvers.SolverXPBD(model, iterations=iterations, soft_body_relaxation=relaxation)
    return model, solver, pipeline, radius, lift, sim_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asset")
    ap.add_argument("--solver", default="vbd", choices=("vbd", "xpbd"))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--substeps", type=int, default=0, help="0 = the official count for this solver")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--radius", default="auto")
    ap.add_argument("--drop", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="soft_contact_margin; 0 derives it from the asset's particle radius")
    ap.add_argument("--no-full-surface", action="store_true")
    ap.add_argument("--usd", default=None, help="write an animated USD of the run here")
    args = ap.parse_args()

    substeps = args.substeps or SUBSTEPS[args.solver]
    model, solver, pipeline, radius, lift, sim_path = build(
        args.asset, args.solver, args.iterations, args.radius, args.drop, args.margin,
        not args.no_full_surface)
    frames = int(args.seconds * args.fps)
    # The recording is written here rather than by newton.viewer.ViewerUSD: the two Newton
    # versions lay their viewer output out differently, and neither carries the asset's render
    # mesh, so there would be nothing to photograph the textured banana from.
    tape = None
    if args.usd:
        tape = recording.Recording(args.usd, int(args.fps), frames,
                                   np.asarray(model.particle_q.numpy()), solver_elements(model),
                                   asset=args.asset, sim_prim_path=sim_path)

    state_0, state_1, control = model.state(), model.state(), model.control()
    contacts = pipeline.contacts()
    dt = 1.0 / (args.fps * substeps)

    start = np.asarray(state_0.particle_q.numpy())
    print(f"[baseline] {args.asset.split('/')[-1]} on {args.solver}: {model.particle_count} particles, "
          f"radius {radius * 1000:.2f} mm, lifted {lift * 100:.1f} cm, {args.iterations} iterations x "
          f"{substeps} substeps at {args.fps:g} fps")
    print(f"[baseline] contact {CONTACT[args.solver]}")
    print(f"[baseline] starts z [{start[:, 2].min():.4f}, {start[:, 2].max():.4f}]")
    for frame in range(frames):
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
        if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
            speed = float(np.abs(np.asarray(state_0.particle_qd.numpy())).max())
            print(f"[baseline] t={frame / args.fps:5.2f}s  z [{q[:, 2].min():8.4f}, {q[:, 2].max():8.4f}]  "
                  f"max|v| {speed:8.3f}")
    q = np.asarray(state_0.particle_q.numpy())
    qd = np.asarray(state_0.particle_qd.numpy())
    fell = float(start[:, 2].min() - q[:, 2].min())
    below = float(max(0.0, -q[:, 2].min() - radius))
    speed = float(np.percentile(np.abs(qd), 99))
    peak = float(np.abs(qd).max())
    height = float(start[:, 2].max() - start[:, 2].min())
    # Thresholds as fractions of the asset's own size and its own drop, so they mean the same
    # thing for any asset. `below` allows for the radius because a particle centre is what is
    # measured and a resting particle sits one radius above the floor.
    if not np.isfinite(q).all():
        verdict = "diverged"
    elif fell < MIN_FALL_OF_DROP * lift:
        verdict = "never-fell"
    elif below > TUNNEL_DEPTH_OF_HEIGHT * height:
        verdict = "through-the-floor"
    elif speed > SETTLED_SPEED_OF_HEIGHT * height:
        verdict = "never-settled"
    else:
        verdict = "pass"
    print(f"[baseline] RESULT fell_mm={fell * 1000:.1f} rest_low_m={q[:, 2].min():.4f} "
          f"rest_high_m={q[:, 2].max():.4f} thickness_mm={(q[:, 2].max() - q[:, 2].min()) * 1000:.1f} "
          f"below_floor_mm={below * 1000:.1f} p99_speed={speed:.3f} max_speed={peak:.3f} verdict={verdict}")
    if tape is not None:
        tape.close()
        print(f"[baseline] wrote {args.usd}")


if __name__ == "__main__":
    wp.config.quiet = True
    main()
