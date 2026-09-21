"""Press a deformable USD with a driven plate in plain Newton, and report how far it gives.

    <venv>/bin/python tools/newton_press.py <asset.usda> [--solver vbd|xpbd] [--usd out.usda]
                                            [--press-to 0.6] [--seconds 4]

The drop test asks whether an asset falls and settles. This asks what it does when something
pushes into it, which is the question a gripper actually poses. Newton's own
`vbd/example_vbd_gripper_soft_grid.py` drives its jaws the same way this drives the plate: a
prismatic joint to the world with a position target and a PD gain, stepped inside the same
substep loop as the soft body, so the plate's contact with the asset is solved rather than
imposed. A transform written straight into the state would push through anything.

The plate is kinematic: its pose is prescribed each substep rather than solved. The gripper
example drives its jaws through a prismatic joint with a PD gain, which is right for a gripper --
you want to know what force the jaw can apply. It is wrong here for two reasons. Newton 1.2.1's
VBD does not move a jointed body at all (measured: the plate stays where it started and never
produces a single contact with the asset), so the experiment simply could not run there. And even
where it does move, a spring-driven plate stops at a depth that depends on how hard the asset
pushes back, so each engine presses to a different depth and the comparison is no longer of one
motion applied to one asset. Prescribing the motion applies the same press everywhere and leaves
the asset's response as the only variable.

Verdict, all measured from the solver's own particles:
  * it must give: the asset's top has to drop by at least `min_compression` of its height;
  * it must not be crushed through the floor;
  * it must come back: after the plate lifts, the top has to return most of what it lost;
  * and every point must stay finite.
"""
import argparse

import numpy as np
import warp as wp

import newton
from pxr import Usd

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asset_properties
import press_shape
import recording
import usd_deformable
from newton_drop import (CONTACT, CONTACT_MARGIN_OF_RADIUS, GROUND_CONTACT_KE, ITERATIONS,
                         SELF_CONTACT, SUBSTEPS, XPBD_MAX_RELAXATION, auto_radius,
                         contact_material, deformable_kind, relaxation_is_a_jacobi_factor,
                         solver_elements, xpbd_relaxation)

PLATE_BODY = 0        # the only body in the scene
# What counts as a pass, as fractions of the asset's own settled height rather than absolute
# millimetres, so the same thresholds mean the same thing for a grape and for a melon.
MIN_COMPRESSION = 0.05
MIN_RECOVERY = 0.5
TUNNEL_DEPTH_OF_HEIGHT = 0.05


def build(asset, solver_name, iterations, radius, press_to, margin, full_surface=True):
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
    elif asset_properties.contact_size(declared)[0] is not None:
        radius = asset_properties.contact_size(declared)[0]
    else:
        radius = auto_radius(points)
        chosen["derived_particle_radius"] = (radius, "the asset declares none; half the median "
                                                      "distance between neighbouring nodes")
    height = float(points[:, 2].max() - points[:, 2].min())
    span = float(max(points[:, 0].max() - points[:, 0].min(), points[:, 1].max() - points[:, 1].min()))

    builder = newton.ModelBuilder()
    builder.default_particle_radius = radius
    stage = Usd.Stage.Open(asset)
    builder.add_usd(stage)
    built = usd_deformable.add_missing(builder, asset, chosen)
    # The radius the run uses is the one the builder ended up with, not the one asked for. A
    # volume deformable takes `default_particle_radius`; a cloth's constructor sets its own from
    # the declared shell thickness and ignores it. Reading it back is the only way the contact
    # margin, the plate's size and the landing tolerance are all talking about the same number.
    radius = float(np.median(np.asarray(builder.particle_radius, dtype=np.float64)))
    if built:
        print(f"[press] this Newton's importer produced nothing; built from the asset's "
              f"declaration instead: {built}")
    sim_path = next((str(prim.GetPath()) for _, prim in usd_deformable.find(stage)), None)
    # Rest on the floor rather than fall onto it: this test is about the plate, not the drop.
    q = np.asarray(builder.particle_q, dtype=np.float64)
    q[:, 2] += radius - q[:, 2].min()
    builder.particle_q = [wp.vec3(*p) for p in q]
    top = float(q[:, 2].max())
    centre = (float(q[:, 0].mean()), float(q[:, 1].mean()))
    ground_shape = builder.shape_count
    builder.add_ground_plane()

    for i in range(builder.shape_count):                       # visual-only import leftovers
        if not int(builder.shape_flags[i]) & (int(newton.ShapeFlags.COLLIDE_SHAPES)
                                              | int(newton.ShapeFlags.COLLIDE_PARTICLES)):
            builder.shape_flags[i] = 0

    # The plate covers the asset's own footprint with a small margin, rather than a square of
    # its longest side: a square plate on a long thin asset is mostly plate, and it hides the
    # thing being measured.
    footprint = (float(q[:, 0].max() - q[:, 0].min()), float(q[:, 1].max() - q[:, 1].min()))
    # Thicker than the contact margin, always. A plate thinner than the distance at which
    # contacts are generated has both of its faces inside the same margin, and the asset gets
    # pushed from underneath as hard as from above: measured, the same press went from 13.9 mm
    # of compression to 3.1 mm when the plate was thinned below it.
    margin = margin or radius * CONTACT_MARGIN_OF_RADIUS
    thickness = press_shape.plate_thickness(height, margin)
    start_z = top + thickness / 2.0 + radius * 2.0
    plate = builder.add_body(xform=wp.transform(wp.vec3(centre[0], centre[1], start_z), wp.quat_identity()),
                             mass=1.0, is_kinematic=True)
    plate_shape = builder.shape_count
    builder.add_shape_box(plate, hx=footprint[0] * 0.6, hy=footprint[1] * 0.6, hz=thickness / 2.0,
                          cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
                          color=(0.25, 0.45, 0.85))

    if solver_name == "vbd":
        builder.color()
    model = builder.finalize()
    kind = deformable_kind(model)
    for name, value in CONTACT[kind][solver_name].items():
        setattr(model, name, value)
        chosen[name] = (value, f"penalty numerics for a {kind} deformable on {solver_name}, from "
                               f"Newton's own examples for that pair")
    friction, restitution = contact_material(declared, chosen)
    model.soft_contact_mu = friction
    model.soft_contact_restitution = restitution
    # The floor and the plate are the experiment's; the asset's own shapes keep what they came
    # in with. Friction is the same number the asset (or, failing that, this run) set globally,
    # so the plate does not grip differently from the ground.
    fixtures = [ground_shape, plate_shape]
    for array, value in ((model.shape_material_ke, GROUND_CONTACT_KE),
                         (model.shape_material_kd, CONTACT[kind][solver_name]["soft_contact_kd"]),
                         (model.shape_material_mu, friction)):
        values = array.numpy()
        values[fixtures] = value
        array.assign(wp.array(values, dtype=float))
    chosen["particle_radius_used"] = (radius, "read back from the model, whatever set it")
    chosen["fixture_ke"] = (GROUND_CONTACT_KE, "the floor and plate are the experiment's")
    asset_properties.report("press", declared, chosen)

    # The pipeline is built BEFORE the solver, on purpose. SolverVBD sizes its per-body
    # contact state from the contacts that already exist, and Newton's own error message says
    # so in as many words: "Pre-size before capture by constructing CollisionPipeline before
    # SolverVBD". The official grasping example (softbody/example_softbody_franka.py) does it in
    # this order too. Built the other way round, the static ground still stops the asset -- a
    # plane hangs off body -1 and needs no per-body list -- but the plate, which is a body,
    # generates no contact at all and slides straight through the deformable.
    kwargs = {"broad_phase": "nxn", "soft_contact_margin": margin}
    import inspect
    # Newton 1.5's VBD can meet the surface between particles, not only the particles; 1.2.1 has
    # no such parameter and SolverXPBD refuses it. It is the engines' real difference and it is
    # on by default, but it changes the answer enough that a like-for-like comparison needs to be
    # able to switch it off -- so the run always says which it used.
    full = (full_surface and solver_name == "vbd"
            and "enable_rigid_soft_full_surface_contact" in inspect.signature(
                newton.CollisionPipeline.__init__).parameters)
    if full:
        kwargs["enable_rigid_soft_full_surface_contact"] = True
    print(f"[press] full-surface soft contact: {full}")
    pipeline = newton.CollisionPipeline(model, **kwargs)

    if solver_name == "vbd":
        # Every particle of the asset could touch the plate at once. The per-body list is fixed
        # at 256 by default and documented as never resizing, so anything past it is dropped
        # without a word.
        solver = newton.solvers.SolverVBD(
            model, iterations=iterations,
            particle_enable_self_contact=SELF_CONTACT[kind],
            particle_self_contact_radius=radius, particle_self_contact_margin=radius * 2.0,
            rigid_body_particle_contact_buffer_size=max(256, model.particle_count))
    else:
        relaxation = (xpbd_relaxation(model.tet_count, model.particle_count)
                      if relaxation_is_a_jacobi_factor() else XPBD_MAX_RELAXATION)
        print(f"[press] soft_body_relaxation {relaxation:.4f}")
        solver = newton.solvers.SolverXPBD(model, iterations=iterations, soft_body_relaxation=relaxation)
    return (model, solver, pipeline, radius, height, start_z, thickness, sim_path,
            (footprint[0] * 0.6, footprint[1] * 0.6, thickness / 2.0), margin)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asset")
    ap.add_argument("--solver", default="vbd", choices=("vbd", "xpbd"))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--substeps", type=int, default=0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--radius", default="auto")
    ap.add_argument("--press-to", type=float, default=0.6, help="plate stops at this much of the asset's height")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="soft_contact_margin; 0 derives it from the asset's particle radius")
    ap.add_argument("--no-full-surface", action="store_true",
                    help="meet the asset's particles only, the way Newton 1.2.1 must")
    ap.add_argument("--usd", default=None)
    args = ap.parse_args()

    substeps = args.substeps or SUBSTEPS[args.solver]
    press_to_frac = args.press_to
    (model, solver, pipeline, radius, height, start_z, thickness, sim_path, plate_half,
     margin) = build(
        args.asset, args.solver, args.iterations, args.radius, args.press_to, args.margin,
        not args.no_full_surface)
    frames = int(args.seconds * args.fps)
    # settle, descend, hold, lift, watch -- in fifths of the run.
    phase = frames // 5
    tape = None
    if args.usd:
        tape = recording.Recording(args.usd, int(args.fps), frames,
                                   np.asarray(model.particle_q.numpy()), solver_elements(model),
                                   asset=args.asset, sim_prim_path=sim_path, plate=plate_half)

    state_0, state_1, control = model.state(), model.state(), model.control()
    contacts = pipeline.contacts()
    dt = 1.0 / (args.fps * substeps)
    plate_xy = np.asarray(state_0.body_q.numpy())[PLATE_BODY][:2]

    def place_plate(state, z, speed):
        """Put the kinematic plate where the script says it is, and tell the solver how fast it
        is going -- friction needs the velocity, not just the pose."""
        q = np.asarray(state.body_q.numpy())
        q[PLATE_BODY] = [plate_xy[0], plate_xy[1], z, 0.0, 0.0, 0.0, 1.0]
        state.body_q.assign(wp.array(q, dtype=wp.transform, device=model.device))
        qd = np.asarray(state.body_qd.numpy())
        qd[PLATE_BODY] = [0.0, 0.0, 0.0, 0.0, 0.0, speed] if qd.shape[1] == 6 else qd[PLATE_BODY]
        state.body_qd.assign(wp.array(qd, dtype=wp.spatial_vector, device=model.device))

    print(f"[press] {args.asset.split('/')[-1]} on {args.solver}: {model.particle_count} particles, "
          f"radius {radius * 1000:.2f} mm, authored height {height * 1000:.1f} mm, "
          f"plate {thickness * 1000:.1f} mm thick parked at {start_z:.4f}, "
          f"{args.iterations} iterations x {substeps} substeps")
    start_top = lowest_top = bottom_z = None
    deepest, recovered, contact_peak, settled_height = 0.0, None, 0, None
    last_plate_z = start_z
    for frame in range(frames):
        if frame < phase:
            plate_z = start_z
        elif frame < 2 * phase:
            plate_z = start_z + (bottom_z - start_z) * (frame - phase) / phase
        elif frame < 3 * phase:
            plate_z = bottom_z
        elif frame < 4 * phase:
            plate_z = bottom_z + (start_z - bottom_z) * (frame - 3 * phase) / phase
        else:
            plate_z = start_z
        speed = (plate_z - last_plate_z) * args.fps
        last_plate_z = plate_z
        for _ in range(substeps):
            state_0.clear_forces()
            place_plate(state_0, plate_z, speed)
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        q = np.asarray(state_0.particle_q.numpy())
        if not np.isfinite(q).all():
            print(f"[press] diverged at {frame / args.fps:.2f}s")
            return
        # A press that generated no contact is a different failure from a press the asset
        # resisted, and the two look identical in the heights alone. Count them.
        for name in ("soft_contact_count", "soft_contact_max"):
            counter = getattr(contacts, name, None)
            if counter is not None and hasattr(counter, "numpy"):
                contact_peak = max(contact_peak, int(np.asarray(counter.numpy()).max()))
                break
        top_now = float(q[:, 2].max())
        if frame == phase - 1:
            # Measure the asset only once it has settled under gravity. Reading its height at
            # frame zero is reading the pose its author happened to save: this banana loses
            # 16 mm just lying down, which a press measured from frame zero would report as
            # compression it never caused -- and it aims the plate at a height the asset no
            # longer has, so the plate stops in the air and presses nothing at all.
            start_top = lowest_top = top_now
            floor_now = float(q[:, 2].min())
            settled_height = top_now - floor_now
            bottom_z = floor_now + (top_now - floor_now) * press_to_frac + thickness / 2.0
            print(f"[press] settled to {top_now:.4f}; plate will go to {bottom_z:.4f} "
                  f"({press_to_frac * 100:.0f}% of the settled height)")
        if start_top is None:
            continue
        lowest_top = min(lowest_top, top_now)
        deepest = max(deepest, -float(q[:, 2].min()))
        if frame >= 4 * phase + phase // 2:
            recovered = top_now
        if tape is not None:
            tape.frame(frame, q, plate_z=plate_z)
        if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
            print(f"[press] t={frame / args.fps:5.2f}s  plate {plate_z:7.4f}  top {top_now:7.4f}  "
                  f"floor {float(q[:, 2].min()):7.4f}", flush=True)

    compressed = start_top - lowest_top
    recovery = 0.0 if recovered is None or compressed <= 1e-9 else (recovered - lowest_top) / compressed
    verdict = press_shape.verdict(contact_peak, compressed, height, deepest, recovery,
                                  settled_height=settled_height, margin=margin)
    print(f"[press] most soft contacts in any frame: {contact_peak}")
    # One token per measurement, no spaces inside a value: the runner reads this line, and a
    # value it has to guess the end of is a value it will read wrong.
    print(f"[press] RESULT start_top_m={start_top:.4f} lowest_top_m={lowest_top:.4f} "
          f"compressed_mm={compressed * 1000:.1f} compressed_frac={compressed / height:.3f} "
          f"recovered_frac={recovery:.2f} below_floor_mm={deepest * 1000:.1f} "
          f"soft_contacts={contact_peak} verdict={verdict}")
    if tape is not None:
        tape.close()
        print(f"[press] wrote {args.usd}")


if __name__ == "__main__":
    wp.config.quiet = True
    main()
