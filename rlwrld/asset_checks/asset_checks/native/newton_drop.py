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

import numpy as np
import warp as wp

import newton
from pxr import Usd

# Per-solver contact constants, quoted from newton/examples/multiphysics/example_rigid_soft_contact.py.
# They are a set: `ke` without its matching `kd` is a different simulation.
CONTACT = {
    "xpbd": {"soft_contact_ke": 75.0, "soft_contact_kd": 1.0, "soft_contact_kf": 1.0e3, "soft_contact_mu": 1.0},
    "vbd": {"soft_contact_ke": 1.0e5, "soft_contact_kd": 1.0e-4, "soft_contact_kf": 1.0e3, "soft_contact_mu": 0.3},
}
GROUND_CONTACT_KE = 2.0e5   # example_rigid_soft_contact.GROUND_CONTACT_KE
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


def build(asset, solver_name, iterations, radius, drop, margin, full_surface):
    # add_usd stamps `default_particle_radius` onto every particle it imports, so the radius
    # has to be known first. Import once cheaply to measure the asset, then again to build it.
    measure = newton.ModelBuilder()
    measure.add_usd(Usd.Stage.Open(asset))
    if measure.particle_count == 0:
        raise SystemExit(f"{asset}: no particles -- this Newton did not import it as a deformable")
    points = np.asarray(measure.particle_q, dtype=np.float64)
    radius = auto_radius(points) if radius == "auto" else float(radius)

    builder = newton.ModelBuilder()
    builder.default_particle_radius = radius
    builder.add_usd(Usd.Stage.Open(asset))

    lift = drop - float(points[:, 2].min())          # lowest point starts `drop` above the plane
    q = np.asarray(builder.particle_q, dtype=np.float64)
    q[:, 2] += lift
    builder.particle_q = [wp.vec3(*p) for p in q]
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
    # The ground is a rigid shape; its own material is what the particles push against.
    model.shape_material_ke.fill_(GROUND_CONTACT_KE)
    model.shape_material_kd.fill_(CONTACT[solver_name]["soft_contact_kd"])
    model.shape_material_mu.fill_(CONTACT[solver_name]["soft_contact_mu"])

    # Pipeline first, then the solver: SolverVBD sizes its per-body contact state from the
    # contacts that already exist, and Newton's own message says to construct CollisionPipeline
    # before SolverVBD. A static ground survives the wrong order; a rigid body does not.
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
    return model, solver, pipeline, radius, lift


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
    ap.add_argument("--margin", type=float, default=0.01, help="soft_contact_margin; the examples use 0.01 m")
    ap.add_argument("--no-full-surface", action="store_true")
    ap.add_argument("--usd", default=None, help="write an animated USD of the run here")
    args = ap.parse_args()

    substeps = args.substeps or SUBSTEPS[args.solver]
    model, solver, pipeline, radius, lift = build(args.asset, args.solver, args.iterations, args.radius,
                                                  args.drop, args.margin, not args.no_full_surface)
    frames = int(args.seconds * args.fps)
    viewer = None
    if args.usd:
        from newton.viewer import ViewerUSD
        viewer = ViewerUSD(output_path=args.usd, fps=int(args.fps), num_frames=frames)
        viewer.set_model(model)

    state_0, state_1, control = model.state(), model.state(), model.control()
    contacts = pipeline.contacts()
    dt = 1.0 / (args.fps * substeps)

    start = np.asarray(state_0.particle_q.numpy())
    print(f"[baseline] {args.asset.split('/')[-1]} on {args.solver}: {model.particle_count} particles, "
          f"radius {radius * 1000:.2f} mm, lifted {lift * 100:.1f} cm, {args.iterations} iterations x "
          f"{substeps} substeps at {args.fps:g} fps, margin {args.margin * 1000:.0f} mm")
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
        if viewer is not None:
            viewer.begin_frame(frame / args.fps)
            viewer.log_state(state_0)
            viewer.end_frame()
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
    if viewer is not None:
        viewer.close()
        print(f"[baseline] wrote {args.usd}")


if __name__ == "__main__":
    wp.config.quiet = True
    main()
