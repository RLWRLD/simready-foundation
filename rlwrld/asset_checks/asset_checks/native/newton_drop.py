"""Drop a deformable USD on a ground plane in plain Newton 1.5 (SolverVBD), and report what happens.

    <venv>/bin/python native/newton_drop.py <asset.usda> --seconds 2 [--setup <setup>]
                                            [--fps 60] [--drop 0.05] [--radius auto|<m>]
                                            [--substeps N --iterations N] [--usd out.usda]

The asset is built by `loader.py` (everything it states, and only the rest ours); this file is
the drop: the asset lifted so its lowest point is `--drop` above a ground plane, stepped through
`newton_scene.Stepper`, and judged by `drop_shape` -- the experiment's own rules, which know no
engine. Every run prints a RESULT line of name=value tokens, the one thing `run.py` reads.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import warp as wp

import drop_shape
import integrity
import loader
import newton_scene
import pace
import recording
import setups
import stepping


def build(asset, iterations, radius, drop, margin, substeps, fps, setup):
    """-> (model, solver, pipeline, radius, lift, sim_path, kinds, margin, bag). `margin` and
    `radius` of 0/"auto" mean: take it from the asset, and say where it came from."""
    asset_ = loader.load(asset, setup, radius, tag="baseline")
    builder, points, radius = asset_.builder, asset_.points, asset_.radius

    lift = drop - float(points[:, 2].min())          # lowest point starts `drop` above the plane
    q = np.asarray(builder.particle_q, dtype=np.float64)
    q[:, 2] += lift
    builder.particle_q = [wp.vec3(*p) for p in q]
    ground_shape = builder.shape_count
    builder.add_ground_plane()

    loader.elements_report("baseline", builder)
    loader.colour(asset_)
    model = builder.finalize()
    loader.configure(asset_, model, fixtures=[ground_shape])

    margin = margin or newton_scene.contact_margin(radius, substeps, fps, drop)
    pipeline = newton_scene.collision_pipeline(model, margin, "baseline")
    solver = loader.vbd_solver(asset_, model, iterations, margin)
    return model, solver, pipeline, radius, lift, asset_.sim_path, asset_.kinds, margin, asset_.bag


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asset")
    ap.add_argument("--setup", default=setups.CANON, choices=setups.NAMES, metavar="SETUP",
                    help=f"{'-'.join(setups.FACTORS)}; see setups.py (default {setups.CANON})")
    ap.add_argument("--iterations", type=int, default=None, help="default: the setup's")
    ap.add_argument("--substeps", type=int, default=0, help="0 = the setup's")
    ap.add_argument("--fps", type=float, default=stepping.FPS)
    ap.add_argument("--seconds", type=float, required=True,
                    help="simulated seconds: the experiment's own SECONDS, which run.py passes")
    ap.add_argument("--radius", default="auto")
    ap.add_argument("--drop", type=float, default=drop_shape.DROP_HEIGHT)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="soft_contact_margin; 0 derives it from the asset's particle radius")
    ap.add_argument("--usd", default=None, help="write an animated USD of the run here")
    args = ap.parse_args()

    setup, substeps, iterations = newton_scene.resolve_stepping(
        args.asset, args.setup, args.fps, args.substeps, args.iterations, tag="baseline")
    model, solver, pipeline, radius, lift, sim_path, kinds, band, bag = build(
        args.asset, iterations, args.radius, args.drop, args.margin, substeps, args.fps, setup)
    frames = int(args.seconds * args.fps)
    # The recording is written here rather than by newton.viewer.ViewerUSD, whose output carries
    # none of the asset's render meshes: there would be nothing to photograph the textures from.
    tape = None
    if args.usd:
        tape = recording.Recording(args.usd, int(args.fps), frames,
                                   np.asarray(model.particle_q.numpy()),
                                   newton_scene.solver_elements(model, kinds), asset=args.asset,
                                   sim_prim_path=sim_path,
                                   ground_half=recording.ground_half(model.particle_q.numpy()))

    stepper = newton_scene.Stepper(model, solver, pipeline, substeps, args.fps, "baseline")
    start = np.asarray(stepper.state_0.particle_q.numpy())
    print(f"[baseline] {args.asset.split('/')[-1]}: {model.particle_count} particles, "
          f"radius {radius * 1000:.2f} mm, lifted {lift * 100:.1f} cm, {iterations} iterations x "
          f"{substeps} substeps at {args.fps:g} fps")
    print(f"[baseline] contact ke {model.soft_contact_ke:.4g} kd {model.soft_contact_kd:g}")
    print(f"[baseline] starts z [{start[:, 2].min():.4f}, {start[:, 2].max():.4f}]")
    first_frame = None
    clock = pace.Pace("baseline", args.fps)
    for frame in range(frames):
        clock.start(frame)
        stepper.frame()
        clock.stop()
        q = np.asarray(stepper.state_0.particle_q.numpy())
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
            speed = float(np.abs(np.asarray(stepper.state_0.particle_qd.numpy())).max())
            print(f"[baseline] t={frame / args.fps:5.2f}s  z [{q[:, 2].min():8.4f}, {q[:, 2].max():8.4f}]  "
                  f"max|v| {speed:8.3f}")
    q = np.asarray(stepper.state_0.particle_q.numpy())
    qd = np.asarray(stepper.state_0.particle_qd.numpy())
    fell = float(start[:, 2].min() - q[:, 2].min())
    below = drop_shape.below_floor(float(q[:, 2].min()), radius)
    # "Settled" is judged on the 99th percentile of node speed, not the maximum: a maximum over a
    # few thousand nodes is decided by whichever single node is jittering.
    speed = float(np.percentile(np.abs(qd), 99))
    peak = float(np.abs(qd).max())
    height = float(start[:, 2].max() - start[:, 2].min())
    # The verdict and the line it is printed on belong to the experiment (drop_shape).
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
