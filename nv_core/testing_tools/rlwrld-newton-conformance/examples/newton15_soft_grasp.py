"""Grasp-and-deform demo for SpaceAI's tetrahedral fruit assets on standalone Newton 1.5.2.

A gantry with two kinematic box pads (the same idea as NVIDIA's grasp_and_lift
benchmark, no robot) descends over the fruit resting on a table, closes on it by
a set squeeze depth, lifts, holds, shakes, holds and opens. The fruit is a Newton
VBD tet soft body built straight from the delivered USDZ (TetMesh prim + the
physics material it binds: Young's modulus, Poisson's ratio, density). Frames
come from the headless OpenGL viewer and go to an mp4.

    python outputs/newton15_soft_grasp.py --asset apple.usdz --out outputs/newton15_videos/apple.mp4

Newton 1.5.2 standalone (warp 1.17), not the Arena image (Newton 1.2.1).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
from newton import ModelBuilder
from newton.solvers import SolverVBD

TABLE_TOP_Z = 0.20


def load_tetmesh(usdz: Path):
    stage = Usd.Stage.Open(str(usdz))
    tet_prim = next(p for p in stage.Traverse() if p.GetTypeName() == "TetMesh")
    tet = newton.TetMesh.create_from_usd(tet_prim, compat_namespaces=())
    # primvars (displayColor etc.) ride along as custom attributes the builder wants registered; we only need geometry + material
    for attr in ("custom_attributes", "custom_attrs", "attributes"):
        store = getattr(tet, attr, None)
        if isinstance(store, dict):
            store.clear()
    render = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.GetName().endswith("_render")), None)
    pts = np.array(tet_prim.GetAttribute("points").Get(), dtype=np.float32)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return tet, tet_prim.GetPath().pathString, lo, hi, (tet.k_mu, tet.k_lambda, tet.density), render


class Demo:
    def __init__(self, usdz: Path, out: Path, squeeze_m: float, fps: int = 30, substeps: int = 20, iterations: int = 10):
        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / substeps
        self.substeps = substeps
        self.sim_time = 0.0
        tet, tet_path, lo, hi, mat, render = load_tetmesh(usdz)
        size = hi - lo
        self.size = size
        k_mu, k_lambda, density = mat
        density = 800.0 if density is None else density
        print(f"[demo] {usdz.name}: tet prim {tet_path}, size {np.round(size*1000,1)} mm, "
              f"k_mu {float(np.mean(k_mu)):.3g} k_lambda {float(np.mean(k_lambda)):.3g} density {float(np.mean(density)):.0f}")

        builder = ModelBuilder(gravity=(0.0, 0.0, -9.81))
        # table
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, TABLE_TOP_Z * 0.5), wp.quat_identity()), hx=0.3, hy=0.3, hz=TABLE_TOP_Z * 0.5)
        # fruit: authored with the bottom at z=0, centre at x,y ~ 0 -> put it on the table
        centre_xy = (lo[:2] + hi[:2]) * 0.5
        self.fruit_pos = wp.vec3(float(-centre_xy[0]), float(-centre_xy[1]), TABLE_TOP_Z - float(lo[2]) + 0.001)
        particle_radius = 0.002
        builder.add_soft_mesh(
            pos=self.fruit_pos, rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tet, density=density, k_mu=k_mu, k_lambda=k_lambda, k_damp=0.0,
            particle_radius=particle_radius,
        )
        # gantry pads: two kinematic boxes closing along x
        self.pad_half = wp.vec3(0.008, 0.02, 0.02)
        self.pad_bodies = []
        for sign in (-1.0, 1.0):
            b = builder.add_body(xform=wp.transform(wp.vec3(sign * 0.2, 0.0, 0.5), wp.quat_identity()), mass=0.0, is_kinematic=True)
            builder.add_shape_box(b, hx=self.pad_half[0], hy=self.pad_half[1], hz=self.pad_half[2])
            self.pad_bodies.append(b)
        builder.color()
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = 2.0e5
        self.model.soft_contact_kd = 2.0e1
        self.model.soft_contact_mu = 0.9
        self.model.shape_material_mu.fill_(1.2)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision = newton.CollisionPipeline(self.model, soft_contact_margin=0.01)
        self.contacts = self.collision.contacts()
        self.solver = SolverVBD(self.model, iterations=iterations, particle_enable_self_contact=False,
                                particle_collision_detection_interval=-1)

        # grasp plan (metres, seconds)
        width = float(size[0]); height = float(size[2])
        self.grasp_z = TABLE_TOP_Z + 0.55 * height
        self.open_x = width * 0.5 + 0.03 + self.pad_half[0]
        self.closed_x = max(width * 0.5 - squeeze_m + self.pad_half[0], 0.005)
        self.lift = max(0.2, 2.0 * float(size.max()))
        self.plan = [  # (duration, x_pad, z_pad) targets, linearly interpolated
            (0.8, self.open_x, self.grasp_z + 0.25),
            (0.8, self.open_x, self.grasp_z),
            (0.8, self.closed_x, self.grasp_z),
            (0.4, self.closed_x, self.grasp_z),
            (1.2, self.closed_x, self.grasp_z + self.lift),
            (0.8, self.closed_x, self.grasp_z + self.lift),
            (1.5, None, None),  # shake
            (0.8, self.closed_x, self.grasp_z + self.lift),
            (0.6, self.open_x, self.grasp_z + self.lift),
            (1.0, self.open_x, self.grasp_z + self.lift),
        ]
        self.total = sum(p[0] for p in self.plan)
        self.body_q_host = np.zeros((self.model.body_count, 7), dtype=np.float32)
        self.viewer = newton.viewer.ViewerGL(width=768, height=768, headless=True)
        self.viewer.set_model(self.model)
        # frame the whole motion: table top to the top of the lift
        mid_z = self.grasp_z + 0.5 * self.lift
        dist = max(0.5, 2.0 * self.lift)
        self.viewer.set_camera(wp.vec3(-dist * 0.7, -dist * 0.7, mid_z + 0.15 * dist), -16.0, 45.0)
        self.writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7)
        self.fruit_z0 = None

    def pad_targets(self, t):
        acc = 0.0; prev = (self.open_x, self.grasp_z + 0.25)
        for dur, x, z in self.plan:
            if x is None:  # shake: 1 cm at 2 Hz about the hold pose
                if t < acc + dur:
                    dz = 0.01 * math.sin(2.0 * math.pi * 2.0 * (t - acc))
                    return prev[0], prev[1] + dz
                acc += dur; continue
            if t < acc + dur:
                a = (t - acc) / dur
                return prev[0] + (x - prev[0]) * a, prev[1] + (z - prev[1]) * a
            acc += dur; prev = (x, z)
        return prev

    def set_pads(self, t, dt):
        x, z = self.pad_targets(t)
        x2, z2 = self.pad_targets(t + dt)
        q = self.state_0.body_q.numpy(); qd = self.state_0.body_qd.numpy()
        for i, sign in zip(self.pad_bodies, (-1.0, 1.0)):
            q[i, :3] = (sign * x, 0.0, z); q[i, 3:] = (0.0, 0.0, 0.0, 1.0)
            qd[i, :3] = ((sign * (x2 - x)) / dt, 0.0, (z2 - z) / dt); qd[i, 3:] = 0.0
        self.state_0.body_q.assign(q); self.state_0.body_qd.assign(qd)
        self.state_1.body_q.assign(q); self.state_1.body_qd.assign(qd)

    def step(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.substeps):
            self.state_0.clear_forces(); self.state_1.clear_forces()
            self.set_pads(self.sim_time, self.sim_dt)
            self.collision.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def fruit_stats(self):
        p = self.state_0.particle_q.numpy()
        return p.mean(axis=0), p.min(axis=0), p.max(axis=0)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        frame = self.viewer.get_frame().numpy()
        self.writer.append_data(frame)  # get_frame() is already top-down

    def run(self):
        log = []
        n = int(self.total / self.frame_dt)
        t0 = time.time()
        for i in range(n):
            self.step(); self.render()
            if i % 15 == 0:
                c, lo, hi = self.fruit_stats(); ext = hi - lo; q = self.state_0.body_q.numpy()
                log.append((round(self.sim_time, 2), round(float(c[2] - TABLE_TOP_Z), 4), [round(float(v) * 1000, 1) for v in ext],
                            "pads x/z mm", [round(float(q[b, 0]) * 1000, 1) for b in self.pad_bodies], round(float(q[self.pad_bodies[0], 2] - TABLE_TOP_Z) * 1000, 1)))
        self.writer.close(); self.viewer.close()
        print(f"[demo] {n} frames in {time.time()-t0:.0f}s; centre height above table / extents (mm) over time:")
        for row in log: print("   ", row)
        return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--squeeze_mm", type=float, default=8.0, help="how far each pad closes past the surface")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    Demo(args.asset, args.out, squeeze_m=args.squeeze_mm / 1000.0).run()


if __name__ == "__main__":
    main()
