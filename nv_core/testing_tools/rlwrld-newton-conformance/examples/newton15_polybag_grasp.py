#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinch-and-lift demo for SpaceAI's polybag USDZ on standalone Newton 1.5.2.

The delivered bag is a formed VBD cloth film (triangle mesh, Newton stiffness
values on its physics material, seal pairs that keep the bag closed) and, for
the loaded variant, a tet soft body of cotton inside. This adapter builds both
into one Newton model: cloth from the film mesh, springs across the seal pairs,
the cotton as a soft mesh, VBD self-contact so film and filling collide. Two
kinematic pads then pinch the bag's sealed lip from above and below, lift it,
hold, shake, hold and open, the way a bag is picked off a bench.

    python scripts/tools/newton15_polybag_grasp.py --asset polybag_green_bubble_cotton_loaded.usdz --out out.mp4

Newton 1.5.2 standalone (warp 1.17), not the Arena image (Newton 1.2.1). Needs a venv with
``newton==1.5.2 usd-core imageio imageio-ffmpeg "pyglet>=2.0"`` and a display for the headless GL viewer
(``DISPLAY=:1`` on the workstation).
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


def read_bag(usdz: Path):
    stage = Usd.Stage.Open(str(usdz))
    film = stage.GetPrimAtPath("/Polybag/Film/Simulation")
    mesh = UsdGeom.Mesh(film)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    fvc = np.array(mesh.GetFaceVertexCountsAttr().Get()); assert (fvc == 3).all(), "film must be triangles"
    tris = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    seal = np.array(film.GetAttribute("rlwrld:sealPairs").Get(), dtype=np.int32).reshape(-1, 2)
    lip = np.array(film.GetAttribute("rlwrld:foldLipIndices").Get(), dtype=np.int32)
    mat = stage.GetPrimAtPath("/Polybag/Film/PhysicsMaterial")
    g = lambda n, d=None: (mat.GetAttribute(n).Get() if mat.GetAttribute(n) and mat.GetAttribute(n).HasAuthoredValue() else d)
    film_params = dict(density=float(g("newton:density", 0.133)), tri_ke=float(g("newton:triKe", 5e4)), tri_ka=float(g("newton:triKa", 0.0)),
                       tri_kd=float(g("newton:triKd", 10.0)), edge_ke=float(g("newton:edgeKe", 1.0)), edge_kd=float(g("newton:edgeKd", 2e-4)),
                       particle_radius=float(g("newton:particleRadius", 0.003)))
    root = stage.GetPrimAtPath("/Polybag")
    r = lambda n, d: (root.GetAttribute(n).Get() if root.GetAttribute(n) and root.GetAttribute(n).HasAuthoredValue() else d)
    contact = dict(seal_ke=float(r("rlwrld:contact:seal_spring_ke_n_m", 1e5)), soft_ke=float(r("rlwrld:contact:soft_contact_ke", 1e5)),
                   soft_kd=float(r("rlwrld:contact:soft_contact_kd", 100.0)), mu=float(r("rlwrld:simulation:soft_contact_mu", 0.8)),
                   self_radius=float(r("rlwrld:contact:self_contact_radius_m", 5e-4)), self_margin=float(r("rlwrld:contact:self_contact_margin_m", 1e-3)))
    contents = stage.GetPrimAtPath("/Polybag/Contents/Simulation")
    tet = None
    if contents and contents.GetTypeName() == "TetMesh":
        tet = newton.TetMesh.create_from_usd(contents, compat_namespaces=())
        for attr in ("custom_attributes",):
            store = getattr(tet, attr, None)
            if isinstance(store, dict):
                store.clear()
    return pts, tris, seal, lip, film_params, contact, tet


class Demo:
    def __init__(self, usdz: Path, out: Path, fps: int = 30, substeps: int = 40, iterations: int = 20, float_start: bool = False):
        self.float_start = float_start
        self.frame_dt = 1.0 / fps; self.sim_dt = self.frame_dt / substeps; self.substeps = substeps; self.sim_time = 0.0
        pts, tris, seal, lip, fp, contact, tet = read_bag(usdz)
        lo, hi = pts.min(0), pts.max(0); size = hi - lo
        centre = (lo + hi) * 0.5
        # put the bag flat on the table, centred in x,y
        # a flat empty film cannot be pinched off a table by cube pads (the lower pad would be inside the
        # table); with --float the bag starts 0.15 m up at zero g, is pinched, and gravity returns before the
        # lift, the free-space variant of the rigid benchmark
        start_z = TABLE_TOP_Z - lo[2] + (0.15 if float_start else 0.002)
        offset = np.array([-centre[0], -centre[1], start_z], dtype=np.float32)
        print(f"[bag] {usdz.name}: film {len(pts)} verts / {len(tris)//3} tris, size {np.round(size*1000,1)} mm, seal pairs {len(seal)}, lip verts {len(lip)}, "
              f"film {fp}, contents {'tets %d' % tet.tet_count if tet else 'none'}")
        builder = ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, TABLE_TOP_Z * 0.5), wp.quat_identity()), hx=0.4, hy=0.4, hz=TABLE_TOP_Z * 0.5)
        p0 = builder.particle_count
        builder.add_cloth_mesh(pos=wp.vec3(*offset.tolist()), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                               vertices=[wp.vec3(*p) for p in pts], indices=tris.tolist(), density=fp["density"],
                               tri_ke=fp["tri_ke"], tri_ka=fp["tri_ka"], tri_kd=fp["tri_kd"], edge_ke=fp["edge_ke"], edge_kd=fp["edge_kd"],
                               particle_radius=fp["particle_radius"])
        for i, j in seal:
            builder.add_spring(int(p0 + i), int(p0 + j), contact["seal_ke"], 1.0, 0.0)
        if tet is not None:
            builder.add_soft_mesh(pos=wp.vec3(*offset.tolist()), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                                  mesh=tet, density=tet.density if tet.density is not None else 75.0,
                                  k_mu=tet.k_mu if tet.k_mu is not None else 900.0, k_lambda=tet.k_lambda if tet.k_lambda is not None else 230.0,
                                  k_damp=10.0, particle_radius=0.002)
        # lip pinch point: middle of the lip vertices, in world
        lip_world = pts[lip] + offset
        self.pinch = lip_world.mean(axis=0); lip_span = lip_world.max(0) - lip_world.min(0)
        print(f"[bag] lip centre {np.round(self.pinch,3)}, lip span {np.round(lip_span*1000,1)} mm, bag top z {hi[2]+offset[2]:.3f}")
        self.pad_half = wp.vec3(0.08, 0.03, 0.006)  # a wide flat pinch along the sealed lip
        self.pad_bodies = []
        for sign in (-1.0, 1.0):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 + sign * 0.1), wp.quat_identity()), mass=0.0, is_kinematic=True)
            builder.add_shape_box(b, hx=self.pad_half[0], hy=self.pad_half[1], hz=self.pad_half[2])
            self.pad_bodies.append(b)
        builder.color()
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = contact["soft_ke"]; self.model.soft_contact_kd = contact["soft_kd"]; self.model.soft_contact_mu = contact["mu"]
        self.model.shape_material_mu.fill_(1.0)
        self.state_0 = self.model.state(); self.state_1 = self.model.state(); self.control = self.model.control()
        self.collision = newton.CollisionPipeline(self.model, soft_contact_margin=0.006)
        self.contacts = self.collision.contacts()
        # detect contacts every substep: a bag dropped from 0.36 m moves ~9 cm per frame and tunnels
        # through the table when detection runs once per frame
        self.solver = SolverVBD(self.model, iterations=iterations, particle_enable_self_contact=True,
                                particle_self_contact_radius=max(contact["self_radius"], 0.002), particle_self_contact_margin=max(contact["self_margin"], 0.004),
                                particle_collision_detection_interval=1)
        # pinch plan along z at the lip: pads close on the film thickness, lift 25 cm
        px, py, pz = (float(v) for v in self.pinch)
        open_dz = 0.05; closed_dz = self.pad_half[2] + 0.0005  # pad faces 1 mm apart on the two film layers
        self.pinch_xy = (px, py); self.lift = 0.36  # the bag is 32 cm long: clear the table when hanging
        self.plan = [  # (duration, pad half-gap, pad z, pad y offset away from the bag)
            (0.8, open_dz, pz + 0.15, 0.0), (0.8, open_dz, pz, 0.0), (0.6, closed_dz, pz, 0.0), (0.4, closed_dz, pz, 0.0),
            (1.5, closed_dz, pz + self.lift, 0.0), (0.8, closed_dz, pz + self.lift, 0.0), (1.5, None, None, None),
            (0.8, closed_dz, pz + self.lift, 0.0), (0.6, open_dz, pz + self.lift, 0.0),
            # an opened lower pad is a shelf under the fold: slide both pads out from under the lip
            (0.8, open_dz, pz + self.lift, 0.22), (0.8, open_dz, pz + self.lift, 0.22),
        ]
        self.total = sum(p[0] for p in self.plan)
        self.close_done_t = sum(p[0] for p in self.plan[:4])
        self.gravity_zero = wp.zeros(self.model.gravity.shape[0], dtype=wp.vec3, device=self.model.device)
        self.gravity_earth = wp.full(self.model.gravity.shape[0], wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3, device=self.model.device)
        if self.float_start:
            self.model.gravity.assign(self.gravity_zero)
        self.viewer = newton.viewer.ViewerGL(width=768, height=768, headless=True); self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(px - 0.75, py - 0.75, pz + 0.30), -14.0, 45.0)
        self.writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7)
        self.film_slice = slice(p0, p0 + len(pts))

    def pad_targets(self, t):
        acc = 0.0; prev = (0.05, self.pinch[2] + 0.15, 0.0)
        for dur, dz, z, dy in self.plan:
            if dz is None:
                if t < acc + dur:
                    return prev[0], prev[1] + 0.01 * math.sin(2.0 * math.pi * 2.0 * (t - acc)), prev[2]
                acc += dur; continue
            if t < acc + dur:
                a = (t - acc) / dur; return prev[0] + (dz - prev[0]) * a, prev[1] + (z - prev[1]) * a, prev[2] + (dy - prev[2]) * a
            acc += dur; prev = (dz, z, dy)
        return prev

    def set_pads(self, t, dt):
        dz, z, dy = self.pad_targets(t); dz2, z2, dy2 = self.pad_targets(t + dt)
        q = self.state_0.body_q.numpy(); qd = self.state_0.body_qd.numpy(); px, py = self.pinch_xy
        for i, sign in zip(self.pad_bodies, (-1.0, 1.0)):
            q[i, :3] = (px, py + dy, z + sign * dz); q[i, 3:] = (0.0, 0.0, 0.0, 1.0)
            qd[i, :3] = (0.0, (dy2 - dy) / dt, ((z2 + sign * dz2) - (z + sign * dz)) / dt); qd[i, 3:] = 0.0
        for st in (self.state_0, self.state_1):
            st.body_q.assign(q); st.body_qd.assign(qd)

    def step(self):
        for _ in range(self.substeps):
            self.solver.rebuild_bvh(self.state_0)
            self.state_0.clear_forces(); self.state_1.clear_forces()
            self.set_pads(self.sim_time, self.sim_dt)
            if self.float_start and self.sim_time >= self.close_done_t:
                self.model.gravity.assign(self.gravity_earth); self.float_start = False
            self.collision.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time); self.viewer.log_state(self.state_0); self.viewer.end_frame()
        self.writer.append_data(self.viewer.get_frame().numpy())

    def run(self):
        n = int(self.total / self.frame_dt); log = []; t0 = time.time()
        for i in range(n):
            self.step(); self.render()
            if i % 15 == 0:
                p = self.state_0.particle_q.numpy(); film = p[self.film_slice]
                log.append((round(self.sim_time, 2), "film z min/mean/max mm above table", [round(float(v - TABLE_TOP_Z) * 1000, 1) for v in (film[:, 2].min(), film[:, 2].mean(), film[:, 2].max())],
                            "nan" if np.isnan(p).any() else "ok"))
        self.writer.close(); self.viewer.close()
        print(f"[bag] {n} frames in {time.time()-t0:.0f}s")
        for row in log: print("   ", row)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--asset", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--substeps", type=int, default=40); ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--float", action="store_true", help="start the bag floating at zero g and restore gravity after the pinch (flat empty film)")
    args = ap.parse_args(); args.out.parent.mkdir(parents=True, exist_ok=True); wp.init()
    Demo(args.asset, args.out, substeps=args.substeps, iterations=args.iterations, float_start=args.float).run()


if __name__ == "__main__":
    main()
