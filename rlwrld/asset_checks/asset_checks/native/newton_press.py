"""Press a deformable USD with a driven plate in plain Newton, and report how far it gives.

    <venv>/bin/python native/newton_press.py <asset.usda> [--solver vbd|xpbd] [--usd out.usda]
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
import integrity
import loader
import pace
import setups
import stepping
import press_shape
import recording
import usd_deformable
from newton_drop import (ITERATIONS, CONTACT_DAMPING_RATIO, contact_stiffness, solver_reads,
                         contact_damping, damping_as_the_kernel_reads_it, membrane_as_springs,
                         element_damping_as_the_kernel_reads_it,
                         full_surface_contact,
                         colour_for_vbd,
                         XPBD_MAX_RELAXATION, auto_radius,
                         contact_material, contact_margin,
                         relaxation_is_a_jacobi_factor,
                         solver_elements, xpbd_relaxation,
                         structure, self_contact, exclusion_kwargs, contact_source,
                         particle_contact_report, elements_report, resolve_stepping)

PLATE_BODY = 0        # the only body in the scene


def build(asset, solver_name, iterations, radius, margin, full_surface=True,
          substeps=stepping.SUBSTEPS, fps=stepping.FPS, setup=None):
    setup = setup or setups.parse(setups.DEFAULT)
    asset_ = loader.load(asset, solver_name, setup, radius, tag="press")
    builder, points, radius = asset_.builder, asset_.points, asset_.radius
    sim_path, kinds, bag, chosen = asset_.sim_path, asset_.kinds, asset_.bag, asset_.chosen
    height = float(points[:, 2].max() - points[:, 2].min())
    # Rest on the floor rather than fall onto it: this test is about the plate, not the drop.
    q = np.asarray(builder.particle_q, dtype=np.float64)
    q[:, 2] += radius - q[:, 2].min()
    builder.particle_q = [wp.vec3(*p) for p in q]
    top = float(q[:, 2].max())
    centre = (float(q[:, 0].mean()), float(q[:, 1].mean()))
    ground_shape = builder.shape_count
    builder.add_ground_plane()

    # The plate covers the asset's own footprint with a small margin, rather than a square of
    # its longest side: a square plate on a long thin asset is mostly plate, and it hides the
    # thing being measured.
    footprint = (float(q[:, 0].max() - q[:, 0].min()), float(q[:, 1].max() - q[:, 1].min()))
    # Thicker than the contact margin, always. A plate thinner than the distance at which
    # contacts are generated has both of its faces inside the same margin, and the asset gets
    # pushed from underneath as hard as from above: measured, the same press went from 13.9 mm
    # of compression to 3.1 mm when the plate was thinned below it.
    # The press sets the asset down rather than dropping it, so gravity's stride does not apply;
    # what decides the band here is how deep the plate means to go, because a particle outside it
    # feels nothing at all.
    # The experiment decides how deep; the engine makes that depth representable. `height` here
    # is the asset's authored height, an upper bound on what it will settle to, so the band is
    # never narrower than the indentation that is coming.
    margin = margin or min(contact_margin(radius, substeps, fps, 0.0, press_shape.press_depth(height)),
                       press_shape.widest_usable_margin(height, radius))
    thickness = press_shape.plate_thickness(height, radius)
    start_z = top + thickness / 2.0 + radius * 2.0
    plate = builder.add_body(xform=wp.transform(wp.vec3(centre[0], centre[1], start_z), wp.quat_identity()),
                             mass=1.0, is_kinematic=True)
    plate_shape = builder.shape_count
    builder.add_shape_box(plate, hx=footprint[0] * press_shape.PLATE_FOOTPRINT,
                          hy=footprint[1] * press_shape.PLATE_FOOTPRINT, hz=thickness / 2.0,
                          cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
                          color=(0.25, 0.45, 0.85))

    membrane_as_springs(builder, solver_name, 1.0 / (fps * substeps), chosen)
    elements_report("press", builder)
    loader.colour(asset_)
    model = builder.finalize()
    # The floor and the plate are the experiment's; the asset's own shapes keep what they came
    # in with, and the plate does not grip differently from the ground.
    fixtures = [ground_shape, plate_shape]
    loader.configure(asset_, model, fixtures)

    # The pipeline is built BEFORE the solver, on purpose. SolverVBD sizes its per-body
    # contact state from the contacts that already exist, and Newton's own error message says
    # so in as many words: "Pre-size before capture by constructing CollisionPipeline before
    # SolverVBD". The official grasping example (softbody/example_softbody_franka.py) does it in
    # this order too. Built the other way round, the static ground still stops the asset -- a
    # plane hangs off body -1 and needs no per-body list -- but the plate, which is a body,
    # generates no contact at all and slides straight through the deformable.
    kwargs = {"broad_phase": "nxn", "soft_contact_margin": margin}
    # It changes the answer enough that a like-for-like comparison needs to be able to switch it
    # off, so the run always says which it used.
    full = full_surface_contact(solver_name, full_surface)
    if full:
        kwargs["enable_rigid_soft_full_surface_contact"] = True
    print(f"[press] full-surface soft contact: {full}")
    pipeline = newton.CollisionPipeline(model, **kwargs)

    if solver_name == "vbd":
        solver = loader.vbd_solver(asset_, model, iterations, margin)
    else:
        print(f"[press] SolverXPBD has no particle self-collision switch, so this run has none")
        particle_contact_report("press", model)
        relaxation = (xpbd_relaxation(model.tet_count, model.particle_count)
                      if relaxation_is_a_jacobi_factor() else XPBD_MAX_RELAXATION)
        print(f"[press] soft_body_relaxation {relaxation:.4f}")
        solver = newton.solvers.SolverXPBD(model, iterations=iterations, soft_body_relaxation=relaxation)
    return (model, solver, pipeline, radius, height, start_z, thickness, sim_path, kinds,
            (footprint[0] * press_shape.PLATE_FOOTPRINT,
             footprint[1] * press_shape.PLATE_FOOTPRINT, thickness / 2.0), margin, plate_shape,
            centre, bag)


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
    ap.add_argument("--margin", type=float, default=0.0,
                    help="soft_contact_margin; 0 derives it from the asset's particle radius")
    ap.add_argument("--no-full-surface", action="store_true",
                    help="meet the asset's particles only, the way Newton 1.2.1 must")
    ap.add_argument("--usd", default=None)
    args = ap.parse_args()

    setup, substeps, iterations = resolve_stepping(args)
    args.iterations = iterations
    (model, solver, pipeline, radius, height, start_z, thickness, sim_path, kinds, plate_half,
     margin, plate_shape, plate_centre, bag) = build(
        args.asset, args.solver, args.iterations, args.radius, args.margin,
        not args.no_full_surface, substeps, args.fps, setup)
    frames = int(args.seconds * args.fps)
    settle_at, recover_at = press_shape.settle_frame(frames), press_shape.recovery_frame(frames)
    tape = None
    if args.usd:
        tape = recording.Recording(args.usd, int(args.fps), frames,
                                   np.asarray(model.particle_q.numpy()),
                                   solver_elements(model, kinds), asset=args.asset,
                                   sim_prim_path=sim_path, plate=plate_half,
                                   plate_centre=plate_centre,
                                   ground_half=recording.ground_half(model.particle_q.numpy()))

    state_0, state_1, control = model.state(), model.state(), model.control()
    contacts = pipeline.contacts()
    dt = 1.0 / (args.fps * substeps)
    plate_xy = np.asarray(state_0.body_q.numpy())[PLATE_BODY][:2]   # until the asset settles

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
    deepest, recovered, contact_peak, plate_peak, settled_height = 0.0, None, 0, 0, None
    depth, settled, lowest_q, finite = None, None, None, True
    last_plate_z = start_z
    clock = pace.Pace("press", args.fps)
    for frame in range(frames):
        plate_z = press_shape.plate_height(frame, frames, start_z, bottom_z)
        speed = (plate_z - last_plate_z) * args.fps
        last_plate_z = plate_z
        # Newton's grasping example rebuilds SolverVBD's BVH once per frame, so this does too.
        # It is not, as was briefly claimed here, the reason a plate can sit inside this asset:
        # adding it changed the measured compression by 0.1 mm. The tree it rebuilds is for
        # particle self-contact; shape collision is refreshed by `pipeline.collide` every substep
        # regardless. It is kept because the example does it, not because it fixed anything.
        clock.start(frame)
        if hasattr(solver, "rebuild_bvh"):
            solver.rebuild_bvh(state_0)
        for _ in range(substeps):
            state_0.clear_forces()
            place_plate(state_0, plate_z, speed)
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        clock.stop()
        q = np.asarray(state_0.particle_q.numpy())
        if not np.isfinite(q).all():
            print(f"[press] diverged at {frame / args.fps:.2f}s")
            finite = False
            break
        # A press that generated no contact is a different failure from a press the asset
        # resisted, and the two look identical in the heights alone. Counting all soft contacts
        # is not enough either: an asset resting on the floor produces hundreds of them whatever
        # the plate does, so the plate's own are counted separately.
        counter = getattr(contacts, "soft_contact_count", None)
        if counter is not None and hasattr(counter, "numpy"):
            total = int(np.asarray(counter.numpy()).max())
            contact_peak = max(contact_peak, total)
            if total and hasattr(contacts, "soft_contact_shape"):
                shapes = np.asarray(contacts.soft_contact_shape.numpy())[:total]
                plate_peak = max(plate_peak, int((shapes == plate_shape).sum()))
        top_now = float(q[:, 2].max())
        if frame == settle_at:
            # Measure the asset only once it has settled under gravity. Reading its height at
            # frame zero is reading the pose its author happened to save: this banana loses
            # 16 mm just lying down, which a press measured from frame zero would report as
            # compression it never caused -- and it aims the plate at a height the asset no
            # longer has, so the plate stops in the air and presses nothing at all.
            start_top = lowest_top = top_now
            settled = lowest_q = q.copy()
            floor_now = float(q[:, 2].min())
            settled_height = top_now - floor_now
            # The asset has stopped moving; press where it actually is.
            plate_xy = np.asarray(press_shape.plate_over(q))
            if tape is not None:
                tape.plate_centre = (float(plate_xy[0]), float(plate_xy[1]))
            depth = press_shape.press_depth(settled_height)
            bottom_z = top_now - depth + thickness / 2.0
            print(f"[press] settled to {top_now:.4f} ({settled_height * 1000:.1f} mm tall); the "
                  f"plate will indent it {depth * 1000:.1f} mm, ")
        if start_top is None:
            continue
        if top_now < lowest_top:
            lowest_top, lowest_q = top_now, q.copy()
        deepest = max(deepest, press_shape.below_floor(float(q[:, 2].min()), radius))
        if frame >= recover_at:
            recovered = top_now
        if tape is not None:
            tape.frame(frame, q, plate_z=plate_z)
        if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
            # The plate's underside is what meets the asset; its centre is what the schedule
            # moves. Printing only the centre made every reading of this log arithmetic.
            print(f"[press] t={frame / args.fps:5.2f}s  plate_underside {plate_z - thickness / 2.0:7.4f}  "
                  f"top {top_now:7.4f}  floor {float(q[:, 2].min()):7.4f}", flush=True)

    compressed = (start_top - lowest_top) if start_top is not None else 0.0
    recovery = press_shape.recovery_fraction(recovered, lowest_top, compressed, radius)
    pressed = (press_shape.pressed_nodes(settled, lowest_q, plate_xy, plate_half[:2], radius)
               if settled is not None else 0)
    verdict = press_shape.verdict(finite, pressed, compressed, settled_height or 0.0, deepest,
                                  recovery, margin)
    print(f"[press] most soft contacts in any frame: {contact_peak}, of which {plate_peak} "
          f"were with the plate; {pressed} node(s) under the plate moved down a contact size")
    # Both runners print through press_shape, so the two engines' results are the same line with
    # the same names in the same order -- a table built from them compares like with like. The
    # line is printed whatever happened, a diverged run included: a run with no RESULT line is a
    # run the harness has to guess about.
    q_last = np.asarray(state_0.particle_q.numpy())
    extra = integrity.result_tokens(q_last, bag["seal"], bag["bodies"], bag["reach"]) if np.isfinite(q_last).all() else ""
    print(press_shape.result_line("press", start_top or 0.0, lowest_top or 0.0, compressed,
                                  settled_height or 0.0, recovery, deepest, pressed, verdict,
                                  indent=depth) + (" " + extra if extra else "")
          + (" " + clock.tokens() if clock.tokens() else ""))
    if tape is not None:
        tape.close()
        print(f"[press] wrote {args.usd}")


if __name__ == "__main__":
    wp.config.quiet = True
    main()
