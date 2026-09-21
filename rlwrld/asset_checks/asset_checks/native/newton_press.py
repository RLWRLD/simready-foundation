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
from newton_drop import (CONTACT, GROUND_CONTACT_KE, ITERATIONS, SUBSTEPS, XPBD_MAX_RELAXATION,
                         auto_radius, relaxation_is_a_jacobi_factor, xpbd_relaxation)

PLATE_BODY = 0        # the only body in the scene


def build(asset, solver_name, iterations, radius, press_to, margin):
    measure = newton.ModelBuilder()
    measure.add_usd(Usd.Stage.Open(asset))
    if measure.particle_count == 0:
        raise SystemExit(f"{asset}: no particles -- this Newton did not import it as a deformable")
    points = np.asarray(measure.particle_q, dtype=np.float64)
    radius = auto_radius(points) if radius == "auto" else float(radius)
    height = float(points[:, 2].max() - points[:, 2].min())
    span = float(max(points[:, 0].max() - points[:, 0].min(), points[:, 1].max() - points[:, 1].min()))

    builder = newton.ModelBuilder()
    builder.default_particle_radius = radius
    builder.add_usd(Usd.Stage.Open(asset))
    # Rest on the floor rather than fall onto it: this test is about the plate, not the drop.
    q = np.asarray(builder.particle_q, dtype=np.float64)
    q[:, 2] += radius - q[:, 2].min()
    builder.particle_q = [wp.vec3(*p) for p in q]
    top = float(q[:, 2].max())
    centre = (float(q[:, 0].mean()), float(q[:, 1].mean()))
    builder.add_ground_plane()

    for i in range(builder.shape_count):                       # visual-only import leftovers
        if not int(builder.shape_flags[i]) & (int(newton.ShapeFlags.COLLIDE_SHAPES)
                                              | int(newton.ShapeFlags.COLLIDE_PARTICLES)):
            builder.shape_flags[i] = 0

    # The plate: wider than the asset so it cannot slide off, thin enough to see past.
    thickness = max(0.01, height * 0.3)
    start_z = top + thickness / 2.0 + radius * 2.0
    plate = builder.add_body(xform=wp.transform(wp.vec3(centre[0], centre[1], start_z), wp.quat_identity()),
                             mass=1.0, is_kinematic=True)
    builder.add_shape_box(plate, hx=span * 0.7, hy=span * 0.7, hz=thickness / 2.0,
                          cfg=newton.ModelBuilder.ShapeConfig(density=1000.0,
                                                              mu=CONTACT[solver_name]["soft_contact_mu"]),
                          color=(0.25, 0.45, 0.85))

    if solver_name == "vbd":
        builder.color()
    model = builder.finalize()
    for name, value in CONTACT[solver_name].items():
        setattr(model, name, value)
    model.shape_material_ke.fill_(GROUND_CONTACT_KE)
    model.shape_material_kd.fill_(CONTACT[solver_name]["soft_contact_kd"])
    model.shape_material_mu.fill_(CONTACT[solver_name]["soft_contact_mu"])

    # The pipeline is built BEFORE the solver, on purpose. SolverVBD sizes its per-body
    # contact state from the contacts that already exist, and Newton's own error message says
    # so in as many words: "Pre-size before capture by constructing CollisionPipeline before
    # SolverVBD". The official grasping example (softbody/example_softbody_franka.py) does it in
    # this order too. Built the other way round, the static ground still stops the asset -- a
    # plane hangs off body -1 and needs no per-body list -- but the plate, which is a body,
    # generates no contact at all and slides straight through the deformable.
    kwargs = {"broad_phase": "nxn", "soft_contact_margin": margin}
    import inspect
    if solver_name == "vbd" and "enable_rigid_soft_full_surface_contact" in inspect.signature(
            newton.CollisionPipeline.__init__).parameters:
        kwargs["enable_rigid_soft_full_surface_contact"] = True
    pipeline = newton.CollisionPipeline(model, **kwargs)

    if solver_name == "vbd":
        # Every particle of the asset could touch the plate at once. The per-body list is fixed
        # at 256 by default and documented as never resizing, so anything past it is dropped
        # without a word.
        solver = newton.solvers.SolverVBD(model, iterations=iterations, particle_enable_self_contact=False,
                                          rigid_body_particle_contact_buffer_size=max(256, model.particle_count))
    else:
        relaxation = (xpbd_relaxation(model.tet_count, model.particle_count)
                      if relaxation_is_a_jacobi_factor() else XPBD_MAX_RELAXATION)
        print(f"[press] soft_body_relaxation {relaxation:.4f}")
        solver = newton.solvers.SolverXPBD(model, iterations=iterations, soft_body_relaxation=relaxation)
    return model, solver, pipeline, radius, height, start_z, thickness


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
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--usd", default=None)
    args = ap.parse_args()

    substeps = args.substeps or SUBSTEPS[args.solver]
    press_to_frac = args.press_to
    model, solver, pipeline, radius, height, start_z, thickness = build(
        args.asset, args.solver, args.iterations, args.radius, args.press_to, args.margin)
    frames = int(args.seconds * args.fps)
    # settle, descend, hold, lift, watch -- in fifths of the run.
    phase = frames // 5
    viewer = None
    if args.usd:
        from newton.viewer import ViewerUSD
        viewer = ViewerUSD(output_path=args.usd, fps=int(args.fps), num_frames=frames)
        viewer.set_model(model)

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
          f"radius {radius * 1000:.2f} mm, authored height {height * 1000:.1f} mm, plate parked at {start_z:.4f}, "
          f"{args.iterations} iterations x {substeps} substeps")
    start_top = lowest_top = bottom_z = None
    deepest, recovered, contact_peak = 0.0, None, 0
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
            bottom_z = floor_now + (top_now - floor_now) * press_to_frac + thickness / 2.0
            print(f"[press] settled to {top_now:.4f}; plate will go to {bottom_z:.4f} "
                  f"({press_to_frac * 100:.0f}% of the settled height)")
        if start_top is None:
            continue
        lowest_top = min(lowest_top, top_now)
        deepest = max(deepest, -float(q[:, 2].min()))
        if frame >= 4 * phase + phase // 2:
            recovered = top_now
        if viewer is not None:
            viewer.begin_frame(frame / args.fps)
            viewer.log_state(state_0)
            viewer.end_frame()
        if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
            print(f"[press] t={frame / args.fps:5.2f}s  plate {plate_z:7.4f}  top {top_now:7.4f}  "
                  f"floor {float(q[:, 2].min()):7.4f}", flush=True)

    compressed = start_top - lowest_top
    recovery = 0.0 if recovered is None or compressed <= 1e-9 else (recovered - lowest_top) / compressed
    print(f"[press] most soft contacts in any frame: {contact_peak}")
    print(f"[press] RESULT start_top={start_top:.4f} lowest_top={lowest_top:.4f} "
          f"compressed={compressed * 1000:.1f}mm ({compressed / height * 100:.0f}% of height) "
          f"recovered={recovery * 100:.0f}% below_floor={deepest * 1000:.1f}mm")
    if viewer is not None:
        viewer.close()
        print(f"[press] wrote {args.usd}")


if __name__ == "__main__":
    wp.config.quiet = True
    main()
