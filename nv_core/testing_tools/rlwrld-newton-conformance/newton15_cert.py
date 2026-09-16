#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Newton 1.5.2 standalone (MuJoCo-Warp) conformance pass over DexBench rigid props.

Mirrors the three NVIDIA simready-benchmark runtime tests that every handled prop
passed (or failed) on Isaac Sim 6.0.1 / PhysX, re-run on the Newton 1.5.2 wheel with
the MuJoCo-Warp rigid solver -- NOT the Isaac Lab-Arena image:

  parse          load the package USD with Newton's own USD importer (rigid body,
                 MassAPI, mesh colliders with the authored ``physics:approximation``)
  ground_drop    drop from 1 cm above a flat floor, settle, no tunnelling / explosion /
                 jitter
  slope_drop     same on a 15 deg slope (with a stop wall at the bottom, like NVIDIA's
                 walled room)
  grasp_and_lift settle the asset on the floor, rebuild a gantry two-pad gripper at the
                 authored ``grasp_identifier_01`` line (pads = cubes, edge 10 % of the
                 bbox geometric mean / 30 % of the min dim when aspect > 5, capped at
                 30 % of the gap; grip force 5x weight; friction 5 on the pads), close,
                 lift by max(0.3 m, 2 x longest edge), hold 1 s, shake 1 cm @ 2 Hz for
                 1.5 s, hold 1 s, open.  Verdict messages follow NVIDIA's wording
                 ("pads touched (no object)", "did not rise", "dropped", ...).

Driver mode loops over a manifest and runs every asset in its own subprocess (a
crash or a hang is a finding, not the end of the run); worker mode (``--single``)
does one asset and writes ``results/<name>.json`` after every stage.

    python newton15_cert.py --assets-root object_library --manifest pkg_paths.txt --out out/
    python newton15_cert.py --assets-root object_library --single 005_tomato_soup_can/usd/005_tomato_soup_can.usd --out out/

Engine label used everywhere: "Newton 1.5.2 standalone (MuJoCo-Warp), not the Arena image".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
import warnings

ENGINE_LABEL = "Newton 1.5.2 standalone (MuJoCo-Warp), not the Arena image"

# --------------------------------------------------------------------------------------
# benchmark parameters (NVIDIA kit-suite semantics; see the module docstring)
# --------------------------------------------------------------------------------------
SIM_DT = 1.0 / 1000.0  # MuJoCo-Warp step
FRAME_HZ = 100  # control / observation rate
SUBSTEPS = int(round(1.0 / (SIM_DT * FRAME_HZ)))
VIDEO_FPS = 5
VIDEO_PX = 512

DROP_HEIGHT = 0.01  # m above the surface
DROP_MAX_T = 6.0  # s
SETTLE_LIN = 2e-2  # m/s  (MuJoCo soft contacts creep a few mm/s on a slope)
SETTLE_ANG = 5e-1  # rad/s
SETTLE_HOLD = 0.5  # s of quiet before "settled"
TUNNEL_DEPTH = 0.01  # m below the support surface counts as tunnelling
EXPLODE_SPEED = 10.0  # m/s

SLOPE_DEG = 15.0

PAD_MU = 5.0  # NVIDIA GripperMaterial static/dynamic friction
PAD_MU_TORSIONAL = 0.05  # NVIDIA pads carry physxCollision:torsionalPatchRadius=1 (rigid pivot); MuJoCo condim 4 + this coefficient [m]
PAD_MASS = 0.02  # kg (NVIDIA used 0.003; heavier pads keep the stiff drive stable at 1 kHz)
GRIP_FORCE_FACTOR = 5.0  # x asset weight
FINGER_KP, FINGER_KD = 2943.0, 100.0
GANTRY_KP, GANTRY_KD, GANTRY_FMAX = 50000.0, 447.2136, 10000.0
LIFT_MIN = 0.3
LIFT_T = 1.0
HOLD_T = 1.0
SHAKE_T, SHAKE_AMP, SHAKE_HZ = 1.5, 0.01, 2.0
RELEASE_T = 1.0
CLOSE_MAX_T = 1.5
RISE_MIN = 0.02  # NVIDIA: "did not rise 0.020m"
DEFAULT_MU = 0.5  # PhysX default material when no PhysicsMaterialAPI is bound

# MuJoCo contact softness.  Newton exports a shape's (ke, kd) as MuJoCo solref
# (timeconst = 2/kd, dampratio = kd/2*sqrt(1/ke)); the builder default (2.5e3, 100) is
# MuJoCo's default solref (0.02 s, 1), whose stiffness scales with the effective mass of
# the contact pair -- a 20 g pad pushed with 5x an asset's weight then sinks through the
# asset.  The cert uses a 4 ms time constant everywhere (asset, floor, pads) and 1 kg of
# armature on the finger DOFs; both are recorded in results and can be changed on the CLI.
CONTACT_TIMECONST = 0.004
CONTACT_DAMPRATIO = 1.0
FINGER_ARMATURE = 1.0


def contact_gains(timeconst=None, dampratio=None):
    """(ke, kd) that Newton's MuJoCo export turns into solref (timeconst, dampratio)."""
    tc = CONTACT_TIMECONST if timeconst is None else timeconst
    dr = CONTACT_DAMPRATIO if dampratio is None else dampratio
    kd = 2.0 / tc
    ke = (kd / (2.0 * dr)) ** 2
    return ke, kd


def apply_contact_defaults(builder, args=None):
    tc = getattr(args, "contact_timeconst", None) if args is not None else None
    ke, kd = contact_gains(tc)
    builder.default_shape_cfg.mu = DEFAULT_MU
    builder.default_shape_cfg.ke = ke
    builder.default_shape_cfg.kd = kd
    return ke, kd


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------------------
# small numpy transform helpers (x, y, z, w quaternions like warp)
# --------------------------------------------------------------------------------------
def q_mul(a, b):
    import numpy as np

    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def q_rot(q, v):
    import numpy as np

    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = q[:3]
    s = q[3]
    if v.ndim == 1:
        return 2.0 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)
    return 2.0 * (v @ u)[:, None] * u[None, :] + (s * s - np.dot(u, u)) * v + 2.0 * s * np.cross(u[None, :], v)


def q_inv(q):
    import numpy as np

    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def tf_apply(tf, v):
    import numpy as np

    tf = np.asarray(tf, dtype=np.float64)
    return q_rot(tf[3:7], v) + tf[:3]


def tf_mul(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.concatenate([q_rot(a[3:7], b[:3]) + a[:3], q_mul(a[3:7], b[3:7])])


def tf_inv(a):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    qi = q_inv(a[3:7])
    return np.concatenate([-q_rot(qi, a[:3]), qi])


def q_axis_angle(axis, ang):
    import numpy as np

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * math.sin(ang / 2)), math.cos(ang / 2)])


def q_from_to(a, b):
    """Quaternion rotating unit vector a onto unit vector b."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -1.0 + 1e-9:
        # 180 deg: pick any perpendicular axis
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return q_axis_angle(perp, math.pi)
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def q_angle_between(qa, qb):
    import numpy as np

    d = abs(float(np.dot(np.asarray(qa), np.asarray(qb))))
    return 2.0 * math.degrees(math.acos(min(1.0, d)))


# --------------------------------------------------------------------------------------
# USD side information (sidecar, grasp line, render bbox)
# --------------------------------------------------------------------------------------
_STAGE_CACHE = {}


def open_stage(usd_path):
    """Open the package stage once per process and reuse it: opening it a second time for
    add_usd made UsdPhysics.LoadUsdPhysicsFromRange corrupt the heap on polybag_3.usda."""
    from pxr import Usd

    st = _STAGE_CACHE.get(usd_path)
    if st is None:
        st = Usd.Stage.Open(usd_path)
        _STAGE_CACHE[usd_path] = st
    return st


def read_stage_info(usd_path):
    from pxr import Usd, UsdGeom

    st = open_stage(usd_path)
    dp = st.GetDefaultPrim()
    info = {"default_prim": str(dp.GetPath()), "mpu": UsdGeom.GetStageMetersPerUnit(st), "up": UsdGeom.GetStageUpAxis(st)}
    bb = UsdGeom.Imageable(dp).ComputeWorldBound(0, "default").ComputeAlignedRange()
    info["render_bbox"] = [list(map(float, bb.GetMin())), list(map(float, bb.GetMax()))]
    grasp = None
    for pr in Usd.PrimRange(dp):
        if pr.IsA(UsdGeom.BasisCurves) and pr.GetName().startswith("grasp_identifier"):
            pts = UsdGeom.BasisCurves(pr).GetPointsAttr().Get()
            xf = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(0)
            parent = pr.GetParent()
            body_anc = None
            anc = parent
            while anc and anc.GetPath() != "/":
                if "PhysicsRigidBodyAPI" in anc.GetAppliedSchemas():
                    body_anc = str(anc.GetPath())
                    break
                anc = anc.GetParent()
            grasp = {
                "path": str(pr.GetPath()),
                "points_world": [list(map(float, xf.Transform(p))) for p in pts],
                "parent": str(parent.GetPath()),
                "rigid_body_ancestor": body_anc,
            }
            break
    info["grasp"] = grasp
    approx = {}
    for pr in Usd.PrimRange(dp):
        if "PhysicsCollisionAPI" in pr.GetAppliedSchemas():
            a = pr.GetAttribute("physics:approximation")
            d = {"approx": a.Get() if a and a.HasAuthoredValue() else None}
            for k in ("physxCollision:contactOffset", "physxCollision:restOffset"):
                at = pr.GetAttribute(k)
                if at and at.HasAuthoredValue():
                    d[k.split(":")[1]] = float(at.Get())
            if pr.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(pr)
                d["nverts"] = len(m.GetPointsAttr().Get() or [])
                d["nfaces"] = len(m.GetFaceVertexCountsAttr().Get() or [])
            approx[str(pr.GetPath())] = d
    info["colliders_authored"] = approx
    return info


def read_sidecar(usd_path):
    base = os.path.splitext(usd_path)[0]
    for cand in (base + ".meta.json",):
        if os.path.exists(cand):
            try:
                with open(cand) as f:
                    d = json.load(f)
                a = d.get("asset", {})
                gl = (a.get("grasp_lines") or [{}])[0]
                return {
                    "mass_kg": a.get("mass_kg"),
                    "version": a.get("version"),
                    "grasp_points": gl.get("points"),
                    "grasp_rule": gl.get("rule"),
                    "physx_benchmark": gl.get("benchmark"),
                }
            except Exception as e:  # noqa: BLE001
                return {"error": repr(e)}
    return None


# --------------------------------------------------------------------------------------
# Newton side
# --------------------------------------------------------------------------------------
def newton_setup():
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    try:
        import pyglet

        pyglet.options["headless"] = True
    except Exception:  # noqa: BLE001
        pass
    import warp as wp

    wp.config.quiet = True
    wp.init()
    import newton

    newton.use_coord_layout_targets = True
    try:
        import coacd

        coacd.set_log_level("off")
    except Exception:  # noqa: BLE001
        pass


def versions():
    out = {}
    from importlib import metadata

    for m, dist in (("newton", "newton"), ("warp", "warp-lang"), ("mujoco", "mujoco"), ("mujoco_warp", "mujoco-warp"), ("coacd", "coacd"), ("scipy", "scipy"), ("pyglet", "pyglet"), ("usd-core", "usd-core")):
        try:
            out[m] = metadata.version(dist)
        except Exception:  # noqa: BLE001
            try:
                out[m] = getattr(__import__(m), "__version__", "?")
            except Exception as e:  # noqa: BLE001
                out[m] = f"missing ({e.__class__.__name__})"
    return out


def geo_name(t):
    from newton import GeoType

    for n in ("PLANE", "BOX", "SPHERE", "CAPSULE", "CYLINDER", "CONE", "MESH", "CONVEX_MESH", "SDF", "HEIGHTFIELD", "ELLIPSOID"):
        if hasattr(GeoType, n) and int(getattr(GeoType, n)) == int(t):
            return n
    return str(int(t))


def shape_vertices_local(b, i):
    """Vertices of collision shape i in its body frame (None for planes / unknown)."""
    import numpy as np
    from newton import GeoType

    t = int(b.shape_type[i])
    xf = np.array(b.shape_transform[i], dtype=np.float64)
    sc = np.array(b.shape_scale[i], dtype=np.float64)
    if t in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
        src = b.shape_source[i]
        v = np.asarray(src.vertices, dtype=np.float64) * sc
    elif t == int(GeoType.BOX):
        hx, hy, hz = sc
        v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == int(GeoType.SPHERE):
        r = sc[0]
        v = np.array([[0, 0, -r], [0, 0, r], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]])
    else:
        return None
    return tf_apply(xf, v)


def _is_planar(vertices):
    """Same test MuJoCo-Warp's exporter applies (it rejects planar mesh colliders)."""
    import numpy as np

    try:
        from newton._src.solvers.mujoco.solver_mujoco import _mujoco_mesh_vertices_are_planar

        return bool(_mujoco_mesh_vertices_are_planar(np.asarray(vertices, dtype=np.float64)))
    except Exception:  # noqa: BLE001
        v = np.asarray(vertices, dtype=np.float64)
        if len(v) < 3:
            return False
        c = v - v.mean(axis=0)
        sv = np.linalg.svd(c, compute_uv=False)
        return bool(sv[-1] <= 1e-6 * max(sv[0], 1e-6))


def thicken_planar_colliders(b, min_thickness=0.001):
    """Give zero-thickness collision meshes a 1 mm slab so MuJoCo-Warp accepts them.

    Returns a list of (shape index, label, thickness) that were modified."""
    import numpy as np
    import newton
    from newton import GeoType, ShapeFlags

    fixed = []
    for i in range(b.shape_count):
        if not (int(b.shape_flags[i]) & int(ShapeFlags.COLLIDE_SHAPES)):
            continue
        t = int(b.shape_type[i])
        if t not in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
            continue
        src = b.shape_source[i]
        sc = np.array(b.shape_scale[i], dtype=np.float64)
        v = np.asarray(src.vertices, dtype=np.float64) * sc
        c = v - v.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        n = vt[-1]
        thin_extent = float((c @ n).max() - (c @ n).min())
        ext = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
        # planar (what MuJoCo-Warp rejects outright) or a sliver thinner than 0.5 mm /
        # 1e-3 of its size (what MuJoCo's compiler rejects as "mesh volume is too small")
        if not (_is_planar(v) or thin_extent < max(5e-4, 1e-3 * ext)):
            continue
        th = max(min_thickness, 1e-3 * ext)
        idx = np.asarray(src.indices, dtype=np.int32).reshape(-1, 3)
        nv = len(v)
        v2 = np.concatenate([v + 0.5 * th * n, v - 0.5 * th * n])
        idx2 = np.concatenate([idx, idx[:, ::-1] + nv])
        mesh = newton.Mesh(v2.astype(np.float32), idx2.flatten(), compute_inertia=False, maxhullvert=getattr(src, "maxhullvert", None))
        b.shape_source[i] = mesh
        b.shape_scale[i] = type(b.shape_scale[i])(1.0, 1.0, 1.0) if not isinstance(b.shape_scale[i], (list, tuple)) else (1.0, 1.0, 1.0)
        fixed.append((i, b.shape_label[i] if i < len(b.shape_label) else str(i), th))
    return fixed


def parse_asset(usd_path, stage_info, sidecar, args=None):
    """Load the package into its own ModelBuilder. Returns (builder|None, parse_info, geom)."""
    import numpy as np
    import newton
    from newton import ShapeFlags

    info = {"ok": False, "engine": ENGINE_LABEL}
    t0 = time.time()
    b = newton.ModelBuilder()
    apply_contact_defaults(b, args)
    caught = []
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            r = b.add_usd(open_stage(usd_path), verbose=False)
        caught = [str(w.message)[:300] for w in ws]
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{e.__class__.__name__}: {e}"
        info["traceback"] = traceback.format_exc()[-3000:]
        info["time_s"] = round(time.time() - t0, 2)
        info["warnings"] = caught
        return None, info, None
    info["time_s"] = round(time.time() - t0, 2)
    info["warnings"] = sorted(set(caught))
    try:
        fixed = thicken_planar_colliders(b)
    except Exception as e:  # noqa: BLE001
        fixed = []
        info["warnings"].append(f"planar-collider check failed: {e.__class__.__name__}: {e}")
    info["planar_colliders_thickened"] = [{"shape": i, "label": lbl, "thickness": round(th, 4)} for i, lbl, th in fixed]
    info["body_count"] = b.body_count
    info["shape_count"] = b.shape_count
    info["joint_count"] = b.joint_count
    jt = []
    for j in range(b.joint_count):
        jt.append(str(newton.JointType(int(b.joint_type[j])).name))
    info["joint_types"] = jt
    path_body = r.get("path_body_map", {})
    path_shape = r.get("path_shape_map", {})
    body_paths = {v: k for k, v in path_body.items()}
    bodies = []
    for bi in range(b.body_count):
        bodies.append(
            {
                "path": body_paths.get(bi, f"body_{bi}"),
                "mass": float(b.body_mass[bi]),
                "com": [float(x) for x in b.body_com[bi]],
                "inertia_diag": [float(np.array(b.body_inertia[bi], dtype=np.float64).reshape(3, 3)[k, k]) for k in range(3)],
                "xform": [float(x) for x in b.body_q[bi]],
            }
        )
    info["bodies"] = bodies
    info["mass_total"] = round(float(sum(b.body_mass)), 6)
    info["sidecar_mass_kg"] = sidecar.get("mass_kg") if sidecar else None

    # shapes: what Newton actually built
    per_body = {}
    collide_shapes = []
    for i in range(b.shape_count):
        fl = int(b.shape_flags[i])
        collide = bool(fl & int(ShapeFlags.COLLIDE_SHAPES))
        body = int(b.shape_body[i])
        d = per_body.setdefault(body, {"CONVEX_MESH": 0, "MESH": 0, "BOX": 0, "OTHER": 0, "visual_only": 0})
        if not collide:
            d["visual_only"] += 1
            continue
        collide_shapes.append(i)
        n = geo_name(b.shape_type[i])
        d[n if n in d else "OTHER"] += 1
    info["collide_shape_count"] = len(collide_shapes)
    info["shapes_per_body"] = {bodies[k]["path"] if k >= 0 else "static": v for k, v in per_body.items()}
    mus = [float(b.shape_material_mu[i]) for i in collide_shapes]
    info["collider_mu"] = {"min": min(mus) if mus else None, "max": max(mus) if mus else None}

    # per authored collider: what happened to its approximation
    colliders = []
    replaced = []
    authored = stage_info.get("colliders_authored", {})
    for path, ad in authored.items():
        sid = path_shape.get(path)
        c = {"path": path, "approx_authored": ad.get("approx"), "nverts": ad.get("nverts")}
        for k in ("contactOffset", "restOffset"):
            if k in ad:
                c[k] = ad[k]
        if sid is None:
            c["newton"] = "not imported"
            c["honoured"] = False
            c["note"] = "no shape created for this prim"
            colliders.append(c)
            replaced.append(f"{path}: {ad.get('approx')} -> not imported")
            continue
        n = geo_name(b.shape_type[sid])
        fl = int(b.shape_flags[sid])
        collide = bool(fl & int(ShapeFlags.COLLIDE_SHAPES))
        body = int(b.shape_body[sid])
        pieces = per_body.get(body, {}).get("CONVEX_MESH", 0)
        a = (ad.get("approx") or "none").lower()
        c["newton"] = n + ("" if collide else " (visual only)")
        if a == "convexhull":
            c["honoured"] = n == "CONVEX_MESH"
            c["note"] = "convex hull (MuJoCo re-hulls it, max 64 vertices)"
        elif a == "convexdecomposition":
            c["honoured"] = n == "CONVEX_MESH" and pieces > 1
            c["note"] = f"CoACD decomposition, {pieces} convex pieces on the body" if c["honoured"] else "decomposition fell back to a single convex hull"
        elif a == "sdf":
            c["honoured"] = False
            c["note"] = "sdf is not a Newton approximation: kept as a triangle mesh, MuJoCo-Warp collides its convex hull (max 64 vertices)"
        elif a == "boundingcube":
            c["honoured"] = n == "BOX"
            c["note"] = "bounding box"
        elif a == "none":
            c["honoured"] = False
            c["note"] = "no approximation authored: triangle mesh, MuJoCo-Warp collides its convex hull"
        else:
            c["honoured"] = False
            c["note"] = f"unknown approximation {a}"
        if not c["honoured"]:
            replaced.append(f"{path}: {ad.get('approx')} -> {c['newton']}")
        colliders.append(c)
    info["colliders"] = colliders
    for i, lbl, th in fixed:
        replaced.append(f"{lbl}: planar / sliver collision mesh thickened to {th * 1000:.1f} mm for MuJoCo-Warp")
    info["approximations_replaced"] = replaced

    # collision extents in the stage frame (authored pose)
    pts = []
    for i in collide_shapes:
        v = shape_vertices_local(b, i)
        if v is None:
            continue
        body = int(b.shape_body[i])
        bq = np.array(b.body_q[body], dtype=np.float64) if body >= 0 else np.array([0, 0, 0, 0, 0, 0, 1.0])
        pts.append(tf_apply(bq, v))
    geom = None
    if pts:
        allp = np.concatenate(pts)
        geom = {"coll_min": allp.min(axis=0), "coll_max": allp.max(axis=0), "coll_pts": allp}
        info["collision_bbox"] = [list(map(float, allp.min(axis=0))), list(map(float, allp.max(axis=0)))]
        info["collision_vertex_count"] = int(len(allp))
    info["render_bbox"] = stage_info.get("render_bbox")
    info["ok"] = True
    if b.body_count == 0:
        info["ok"] = False
        info["error"] = "no rigid body imported"
    elif geom is None:
        info["ok"] = False
        info["error"] = "no collision shape imported"
    return b, info, geom


class Sim:
    """One finalized scene + MuJoCo-Warp solver + captured substep graph."""

    def __init__(self, builder, args, label=""):
        import warp as wp
        import newton

        self.model = builder.finalize()
        kw = {"nconmax": args.nconmax, "njmax": args.njmax, "enable_multiccd": not args.no_multiccd}
        self.solver_settings = dict(kw)
        if args.cone:
            kw["cone"] = args.cone
        if args.impratio is not None:
            kw["impratio"] = args.impratio
        self.solver_settings = dict(kw)
        self.solver = newton.solvers.SolverMuJoCo(self.model, **kw)
        self.s0 = self.model.state()
        self.s1 = self.model.state()
        self.control = self.model.control()
        self.t = 0.0
        self.graph = None
        self.use_graph = wp.get_device().is_cuda and not args.no_graph
        if self.use_graph:
            try:
                with wp.ScopedCapture() as cap:
                    self._substeps()
                self.graph = cap.graph
            except Exception as e:  # noqa: BLE001
                log(f"[warn] graph capture failed ({e}); stepping eagerly")
                self.graph = None

    def _substeps(self):
        for _ in range(SUBSTEPS):
            self.s0.clear_forces()
            self.solver.step(self.s0, self.s1, self.control, None, SIM_DT)
            self.s0, self.s1 = self.s1, self.s0

    def frame(self):
        import warp as wp

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._substeps()
        self.t += 1.0 / FRAME_HZ

    def body_q(self):
        return self.s0.body_q.numpy().astype("float64")

    def body_qd(self):
        return self.s0.body_qd.numpy().astype("float64")

    def joint_q(self):
        return self.s0.joint_q.numpy().astype("float64")

    def joint_qd(self):
        return self.s0.joint_qd.numpy().astype("float64")


def add_support(scene, slope_deg=None, args=None, half=1.2):
    """Flat floor, or a 15 deg slope box (top face through the origin) walled on four sides.

    ``half`` is the half-size of the walled slope; the drop test sizes it to the asset
    (1.5 x its longest edge, at least 0.25 m) so rolling props stop at a wall instead of
    bouncing between distant walls, like NVIDIA's small walled drop room."""
    import numpy as np
    import warp as wp
    import newton

    ke, kd = contact_gains(getattr(args, "contact_timeconst", None) if args is not None else None)
    cfg = newton.ModelBuilder.ShapeConfig(mu=DEFAULT_MU, ke=ke, kd=kd)
    if not slope_deg:
        scene.add_ground_plane(cfg=cfg)
        return np.array([0.0, 0.0, 1.0]), np.zeros(3)
    th = math.radians(slope_deg)
    q = q_axis_angle([0, 1, 0], th)  # +x goes downhill
    hz = 0.05
    centre = q_rot(q, [0.0, 0.0, -hz])
    scene.add_shape_box(-1, xform=wp.transform(wp.vec3(*centre), wp.quat(*q)), hx=half, hy=half, hz=hz, cfg=cfg, label="slope")
    # walls on all four sides (NVIDIA's drop room is walled); 0.3 m tall
    for k, (cx, cy, hx, hy) in enumerate(((half, 0.0, 0.02, half), (-half, 0.0, 0.02, half), (0.0, half, half, 0.02), (0.0, -half, half, 0.02))):
        wc = q_rot(q, [cx, cy, 0.15])
        scene.add_shape_box(-1, xform=wp.transform(wp.vec3(*wc), wp.quat(*q)), hx=hx, hy=hy, hz=0.15, cfg=cfg, label=f"wall_{k}")
    n = q_rot(q, [0.0, 0.0, 1.0])
    return n, np.zeros(3)


def place_offset(geom, normal, surface_pt, height):
    """Translation so the lowest collision vertex sits `height` above the support plane."""
    import numpy as np

    pts = geom["coll_pts"]
    d = (pts - surface_pt) @ normal
    need = height - d.min()
    centre_xy = 0.5 * (geom["coll_min"] + geom["coll_max"])
    off = np.array([-centre_xy[0], -centre_xy[1], 0.0]) + normal * need
    # keep the xy centring exact on the surface plane: correct the normal shift's xy drift
    return off


def asset_body_ids(scene_map, n_asset_bodies):
    return list(range(n_asset_bodies))


def transformed_points(pts_local_by_body, body_q):
    """World-frame collision points given per-body local points."""
    import numpy as np

    out = []
    for bi, pl in pts_local_by_body.items():
        out.append(tf_apply(body_q[bi], pl))
    return np.concatenate(out) if out else np.zeros((0, 3))


def local_points_by_body(b, collide_only=True):
    """Per-body collision vertices in body frame (for penetration / extents at runtime)."""
    import numpy as np
    from newton import ShapeFlags

    out = {}
    for i in range(b.shape_count):
        if collide_only and not (int(b.shape_flags[i]) & int(ShapeFlags.COLLIDE_SHAPES)):
            continue
        body = int(b.shape_body[i])
        if body < 0:
            continue
        v = shape_vertices_local(b, i)
        if v is None:
            continue
        out.setdefault(body, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# --------------------------------------------------------------------------------------
# drop tests
# --------------------------------------------------------------------------------------
def run_drop(asset_b, geom, args, slope_deg=None):
    import numpy as np
    import warp as wp
    import newton

    t0 = time.time()
    res = {"engine": ENGINE_LABEL, "slope_deg": slope_deg or 0.0}
    scene = newton.ModelBuilder()
    apply_contact_defaults(scene, args)
    longest = float(np.max(geom["coll_max"] - geom["coll_min"]))
    half = min(1.2, max(0.25, 1.5 * longest))
    normal, spt = add_support(scene, slope_deg, args, half=half)
    res["slope_half_size"] = round(half, 3) if slope_deg else None
    off = place_offset(geom, normal, spt, DROP_HEIGHT)
    scene.add_builder(asset_b, xform=wp.transform(wp.vec3(*off), wp.quat_identity()))
    nb = asset_b.body_count
    pts_local = local_points_by_body(asset_b)
    try:
        sim = Sim(scene, args)
    except Exception as e:  # noqa: BLE001
        res["verdict"] = "fail"
        res["message"] = f"solver setup failed: {e.__class__.__name__}: {str(e)[:400]}"
        res["time_s"] = round(time.time() - t0, 1)
        return res

    q0 = sim.body_q()[:nb]
    quiet_since = None
    settled_t = None
    max_speed = 0.0
    min_height = 1e9
    below_frames = 0
    nan = False
    fly = False
    hist_speed = []
    hist_ang = []
    nframes = int(DROP_MAX_T * FRAME_HZ)
    for f in range(nframes):
        sim.frame()
        bq = sim.body_q()[:nb]
        bqd = sim.body_qd()[:nb]
        if not (np.all(np.isfinite(bq)) and np.all(np.isfinite(bqd))):
            nan = True
            break
        lin = np.linalg.norm(bqd[:, :3], axis=1).max()
        ang = np.linalg.norm(bqd[:, 3:], axis=1).max()
        max_speed = max(max_speed, float(lin))
        hist_speed.append(float(lin))
        hist_ang.append(float(ang))
        pw = transformed_points(pts_local, bq)
        h = float(((pw - spt) @ normal).min())
        min_height = min(min_height, h)
        if h < -TUNNEL_DEPTH:
            below_frames += 1
        if np.linalg.norm(bq[:, :3], axis=1).max() > 3.0 or lin > EXPLODE_SPEED:
            fly = True
            break
        if lin < SETTLE_LIN and ang < SETTLE_ANG:
            if quiet_since is None:
                quiet_since = sim.t
            elif sim.t - quiet_since >= SETTLE_HOLD and settled_t is None:
                settled_t = quiet_since
                break
        else:
            quiet_since = None
    bq = sim.body_q()[:nb]
    pw = transformed_points(pts_local, bq)
    rest_h = float(((pw - spt) @ normal).min()) if len(pw) else float("nan")
    res.update(
        {
            "settle_time_s": None if settled_t is None else round(settled_t, 2),
            "sim_time_s": round(sim.t, 2),
            "rest_body_z": [round(float(z), 4) for z in bq[:, 2]],
            "rest_min_height": round(rest_h, 4),
            "max_penetration": round(max(0.0, -min_height), 4),
            "max_speed": round(max_speed, 3),
            "residual_speed": round(float(np.max(hist_speed[-int(0.5 * FRAME_HZ) :])) if hist_speed else 0.0, 4),
            "residual_ang_speed": round(float(np.max(hist_ang[-int(0.5 * FRAME_HZ) :])) if hist_ang else 0.0, 4),
            "tilt_deg": round(max(q_angle_between(q0[i, 3:7], bq[i, 3:7]) for i in range(nb)), 1),
            "displacement_xy": round(float(np.linalg.norm((bq[0, :2] - q0[0, :2]))), 3),
            "nan": nan,
            "fly_away": fly,
        }
    )
    left_slope = bool(slope_deg) and (abs(bq[0, 0]) > half + 0.05 or abs(bq[0, 1]) > half + 0.05 or bq[0, 2] < -(half + 0.3))
    res["left_slope"] = left_slope
    res["reached_wall"] = bool(slope_deg) and float(np.linalg.norm(bq[0, :2] - q0[0, :2])) > 0.6 * half
    msg = []
    verdict = "pass"
    if nan:
        verdict, msg = "fail", ["NaN in body state"]
    elif fly:
        verdict, msg = "fail", [f"flew away / exploded (max speed {max_speed:.1f} m/s)"]
    elif below_frames > 5 or rest_h < -TUNNEL_DEPTH:
        verdict, msg = "fail", [f"tunnelled into the support ({-min_height * 1000:.1f} mm)"]
    elif settled_t is None:
        if left_slope:
            verdict, msg = "fail", ["left the walled slope (went over / through a wall)"]
        else:
            verdict, msg = "fail", [f"did not come to rest within {DROP_MAX_T:.0f} s (residual {res['residual_speed']:.3f} m/s, {res['residual_ang_speed']:.2f} rad/s)"]
    if verdict == "pass" and res.get("reached_wall"):
        msg.append("rolled / slid to the wall")
    if verdict == "pass" and slope_deg and res["residual_speed"] > 0.003:
        msg.append(f"creep {res['residual_speed'] * 1000:.1f} mm/s while 'at rest' (MuJoCo soft contact)")
    if verdict == "pass" and res["max_penetration"] > 0.003:
        msg.append(f"soft contact: mesh vertices dip {res['max_penetration'] * 1000:.1f} mm below the surface (MuJoCo hull < mesh)")
    if verdict == "pass" and res["tilt_deg"] > 20:
        msg.append(f"tipped over ({res['tilt_deg']:.0f} deg from the authored pose)")
    res["verdict"] = verdict
    res["message"] = "; ".join(msg)
    res["time_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------------------
# grasp_and_lift
# --------------------------------------------------------------------------------------
def settle_on_floor(asset_b, geom, args):
    """Scene 1: drop from 1 cm, settle, return the settled body poses (or None)."""
    import numpy as np
    import warp as wp
    import newton

    scene = newton.ModelBuilder()
    apply_contact_defaults(scene, args)
    normal, spt = add_support(scene, None, args)
    off = place_offset(geom, normal, spt, DROP_HEIGHT)
    scene.add_builder(asset_b, xform=wp.transform(wp.vec3(*off), wp.quat_identity()))
    nb = asset_b.body_count
    sim = Sim(scene, args)
    quiet_since = None
    for _ in range(int(3.0 * FRAME_HZ)):
        sim.frame()
        bqd = sim.body_qd()[:nb]
        lin = np.linalg.norm(bqd[:, :3], axis=1).max()
        ang = np.linalg.norm(bqd[:, 3:], axis=1).max()
        if lin < SETTLE_LIN and ang < SETTLE_ANG:
            if quiet_since is None:
                quiet_since = sim.t
            elif sim.t - quiet_since >= SETTLE_HOLD:
                break
        else:
            quiet_since = None
    bq = sim.body_q()[:nb]
    if not np.all(np.isfinite(bq)):
        return None, off
    return bq, off


def pad_edge_for(dims, gap):
    dims = [max(float(d), 1e-4) for d in dims]
    gm = (dims[0] * dims[1] * dims[2]) ** (1.0 / 3.0)
    aspect = max(dims) / min(dims)
    e = 0.30 * min(dims) if aspect > 5.0 else 0.10 * gm
    e = min(e, 0.30 * gap)
    return max(e, 0.004)


def run_grasp(asset_b, geom, parse_info, stage_info, args, video_path=None):
    import numpy as np
    import warp as wp
    import newton

    t0 = time.time()
    res = {"engine": ENGINE_LABEL, "phases": []}
    g = stage_info.get("grasp")
    if not g:
        res.update({"verdict": "fail", "phase": "Setup", "message": "no grasp_identifier prims found"})
        return res
    nb = asset_b.body_count
    settled, off = settle_on_floor(asset_b, geom, args)
    if settled is None:
        res.update({"verdict": "fail", "phase": "Settle", "message": "NaN while settling on the floor"})
        return res
    # grasp line placement.  NVIDIA's kit benchmark rebuilds the gantry after the asset
    # settles but at the *authored* line pose plus the AssetRoot offset (its log for
    # styrofoam_box shows gp z = authored z + 0.01 while the body had dropped 8 mm), i.e.
    # the line stays in the stage frame.  That is the default here ("stage"); "body"
    # attaches the line to the nearest rigid-body ancestor of the grasp Xform (or the
    # root body) and follows its settled pose.
    root_authored = np.array(asset_b.body_q[0], dtype=np.float64)
    line_frame = args.line_frame
    if line_frame == "body":
        body_idx = 0
        anc = g.get("rigid_body_ancestor")
        if anc:
            for bi, bd in enumerate(parse_info.get("bodies", [])):
                if bd.get("path") == anc:
                    body_idx = bi
        ref_authored = np.array(asset_b.body_q[body_idx], dtype=np.float64)
        p_local = [tf_apply(tf_inv(ref_authored), np.array(p)) for p in g["points_world"]]
        gp = [tf_apply(settled[body_idx], p) for p in p_local]
        line_frame_used = f"body:{parse_info['bodies'][body_idx]['path']}"
    else:
        gp = [np.array(p, dtype=np.float64) + off for p in g["points_world"]]
        line_frame_used = "stage (authored pose + placement offset, NVIDIA semantics)"
    gp1, gp2 = gp
    settle_shift = float(np.linalg.norm(settled[0][:3] - (root_authored[:3] + off)))
    axis = gp2 - gp1
    gap = float(np.linalg.norm(axis))
    if gap < 1e-4:
        res.update({"verdict": "fail", "phase": "Setup", "message": "degenerate grasp line"})
        return res
    u = axis / gap
    centre = 0.5 * (gp1 + gp2)
    rb = parse_info.get("render_bbox") or parse_info.get("collision_bbox")
    dims = [rb[1][k] - rb[0][k] for k in range(3)]
    e = pad_edge_for(dims, gap)
    mass = float(parse_info["mass_total"])
    f_grip = GRIP_FORCE_FACTOR * mass * 9.81
    lift = max(LIFT_MIN, 2.0 * max(dims))
    obj_mu = parse_info.get("collider_mu", {}).get("max") or DEFAULT_MU
    pad_mu = 0.5 * (PAD_MU + obj_mu)  # PhysX averages the pair; MuJoCo takes the max
    q_close_max = 0.5 * gap - 0.5 * e  # pads meet
    res["params"] = {
        "gap": round(gap, 4),
        "pad_edge": round(e, 4),
        "grip_force_N": round(f_grip, 3),
        "lift_height": round(lift, 3),
        "pad_mu_effective": round(pad_mu, 2),
        "pad_mass": PAD_MASS,
        "pad_condim": int(args.pad_condim),
        "pad_mu_torsional": PAD_MU_TORSIONAL,
        "finger_armature": args.finger_armature,
        "contact_timeconst": args.contact_timeconst,
        "gp1": [round(float(x), 4) for x in gp1],
        "gp2": [round(float(x), 4) for x in gp2],
        "axis": [round(float(x), 3) for x in u],
        "line_vertical_component": round(abs(float(u[2])), 3),
        "settled_root_pose": [round(float(x), 4) for x in settled[0]],
        "settle_tilt_deg": round(q_angle_between(root_authored[3:7], settled[0][3:7]), 1),
        "settle_shift_m": round(settle_shift, 4),
        "line_frame": line_frame_used,
        "grasp_xform_parent": g.get("parent"),
        "grasp_rigid_body_ancestor": g.get("rigid_body_ancestor"),
    }
    notes = []
    if min(gp1[2], gp2[2]) - 0.5 * e < 0.0:
        notes.append(f"pad bottom starts {(0.5 * e - min(gp1[2], gp2[2])) * 1000:.1f} mm below the floor (line z={min(gp1[2], gp2[2]):.3f} m)")

    # scene 2: asset at the settled pose + gantry
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    ke, kd = apply_contact_defaults(scene, args)
    add_support(scene, None, args)
    xf = tf_mul(settled[0], tf_inv(root_authored))
    scene.add_builder(asset_b, xform=wp.transform(wp.vec3(*xf[:3]), wp.quat(*xf[3:7])))
    pts_local = local_points_by_body(asset_b)

    I3 = np.eye(3) * 1e-2
    gz = scene.add_link(xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), mass=1.0, inertia=wp.mat33(*I3.flatten()), label="gantry_z")
    gx = scene.add_link(xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), mass=1.0, inertia=wp.mat33(*I3.flatten()), label="gantry_x")
    gy = scene.add_link(xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), mass=1.0, inertia=wp.mat33(*I3.flatten()), label="gantry_y")
    pad_cfg = newton.ModelBuilder.ShapeConfig(mu=pad_mu, mu_torsional=PAD_MU_TORSIONAL, density=PAD_MASS / (e**3), restitution=0.0, ke=ke, kd=kd)
    pads = []
    for k, (p, sgn) in enumerate(((gp1, 1.0), (gp2, -1.0))):
        qk = q_from_to([1.0, 0.0, 0.0], sgn * u)  # local +X = closing direction
        pb = scene.add_link(xform=wp.transform(wp.vec3(*p), wp.quat(*qk)), mass=0.0, label=f"pad_{k}")
        scene.add_shape_box(pb, hx=e / 2, hy=e / 2, hz=e / 2, cfg=pad_cfg, label=f"pad_{k}_cube", custom_attributes={"mujoco:condim": int(args.pad_condim)})
        pads.append((pb, qk))
    gkw = dict(target_ke=GANTRY_KP, target_kd=GANTRY_KD, effort_limit=GANTRY_FMAX, limit_lower=-10.0, limit_upper=10.0, actuator_mode=newton.JointTargetMode.POSITION, target_pos=0.0)
    jz = scene.add_joint_prismatic(-1, gz, parent_xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), axis=newton.Axis.Z, label="joint_z", **gkw)
    jx = scene.add_joint_prismatic(gz, gx, axis=newton.Axis.X, label="joint_x", **gkw)
    jy = scene.add_joint_prismatic(gx, gy, axis=newton.Axis.Y, label="joint_y", **gkw)
    fkw = dict(target_ke=FINGER_KP, target_kd=FINGER_KD, effort_limit=f_grip, limit_lower=0.0, limit_upper=q_close_max, actuator_mode=newton.JointTargetMode.POSITION, target_pos=0.0, armature=args.finger_armature)
    jp = []
    for k, ((pb, qk), p) in enumerate(zip(pads, (gp1, gp2))):
        rel = p - centre
        j = scene.add_joint_prismatic(gy, pb, parent_xform=wp.transform(wp.vec3(*rel), wp.quat(*qk)), axis=newton.Axis.X, label=f"finger_{k}", **fkw)
        jp.append(j)
    scene.add_articulation([jz, jx, jy, *jp], label="gantry")

    try:
        sim = Sim(scene, args)
    except Exception as ex:  # noqa: BLE001
        res.update({"verdict": "fail", "phase": "Setup", "message": f"solver setup failed: {ex.__class__.__name__}: {str(ex)[:400]}"})
        res["time_s"] = round(time.time() - t0, 1)
        return res
    model = sim.model
    qd_start = model.joint_qd_start.numpy()
    tq = model.joint_target_q_start.numpy() if model.joint_target_q_start is not None else qd_start
    n_t = sim.control.joint_target_q.shape[0]
    targets = np.zeros(n_t, dtype=np.float32)
    idx = {"z": int(tq[jz]), "x": int(tq[jx]), "y": int(tq[jy]), "p0": int(tq[jp[0]]), "p1": int(tq[jp[1]])}
    q_start = model.joint_q_start.numpy()
    jq_idx = {"p0": int(q_start[jp[0]]), "p1": int(q_start[jp[1]])}
    pad_bodies = [pads[0][0], pads[1][0]]

    def set_targets(z=0.0, x=0.0, y=0.0, p=0.0):
        targets[idx["z"]] = z
        targets[idx["x"]] = x
        targets[idx["y"]] = y
        targets[idx["p0"]] = p
        targets[idx["p1"]] = p
        sim.control.joint_target_q.assign(targets)

    # video
    viewer = None
    frames = []
    if video_path:
        try:
            import newton.viewer

            viewer = newton.viewer.ViewerGL(width=VIDEO_PX, height=VIDEO_PX, headless=True)
            viewer.set_model(model)
            look = centre + np.array([0.0, 0.0, 0.5 * lift])
            dist = 1.7 * max(lift, 2.0 * max(dims), 0.35)
            az = math.atan2(u[1], u[0]) + math.pi / 2
            cam = look + dist * np.array([math.cos(az) * 0.85, math.sin(az) * 0.85, 0.45])
            d = look - cam
            d /= np.linalg.norm(d)
            viewer.set_camera(pos=wp.vec3(*cam), pitch=math.degrees(math.asin(d[2])), yaw=math.degrees(math.atan2(d[1], d[0])))
        except Exception as ex:  # noqa: BLE001
            notes.append(f"video disabled: {ex.__class__.__name__}: {str(ex)[:120]}")
            viewer = None
    every = FRAME_HZ // VIDEO_FPS

    state = {"f": 0}

    def capture():
        if viewer is None:
            return
        if state["f"] % every == 0:
            viewer.begin_frame(sim.t)
            viewer.log_state(sim.s0)
            viewer.end_frame()
            frames.append(viewer.get_frame().numpy().copy())
        state["f"] += 1

    def obs():
        bq = sim.body_q()
        return bq, bq[:nb], bq[pad_bodies[0]], bq[pad_bodies[1]]

    def obj_min_z(bq):
        pw = transformed_points(pts_local, bq)
        return float(pw[:, 2].min())

    def finite(bq):
        return bool(np.all(np.isfinite(bq)))

    phase_log = []
    verdict = None
    phase = None
    message = ""

    def fail(ph, msg):
        nonlocal verdict, phase, message
        verdict, phase, message = "fail", ph, msg

    def run_for(seconds, ph, ctrl_fn=None, check_fn=None):
        nonlocal verdict
        n = int(round(seconds * FRAME_HZ))
        for i in range(n):
            if ctrl_fn:
                ctrl_fn(i / FRAME_HZ)
            sim.frame()
            capture()
            bq, aq, p0, p1 = obs()
            if not finite(bq):
                fail(ph, "NaN in body state")
                return False
            if check_fn and not check_fn(bq, aq, p0, p1):
                return False
        return True

    # Phase: Positioning / settle 0.5 s with the pads open at the endpoints
    set_targets()
    bq, aq, p0, p1 = obs()
    z_pad0 = 0.5 * (p0[2] + p1[2])
    if not run_for(0.5, "Positioning"):
        pass
    else:
        bq, aq, p0, p1 = obs()
        obj_z0 = float(aq[0, 2])
        obj_xy0 = aq[0, :2].copy()
        z_pad0 = 0.5 * (p0[2] + p1[2])
        moved = float(np.linalg.norm(aq[0, :3] - settled[0][:3]))
        if moved > 0.02:
            notes.append(f"asset moved {moved * 1000:.0f} mm while the gripper was positioned")
        phase_log.append(f"t={sim.t:.1f}s Positioning obj_z={obj_z0:.4f} pads_z={z_pad0:.4f} gap={gap:.3f} pad={e:.4f}")

        # Phase: Grasping (close until both pads stop or timeout)
        set_targets(p=q_close_max)
        quiet = 0
        closed_ok = False
        n = int(CLOSE_MAX_T * FRAME_HZ)
        for i in range(n):
            sim.frame()
            capture()
            jqd = sim.joint_qd()
            v0 = abs(jqd[int(qd_start[jp[0]])])
            v1 = abs(jqd[int(qd_start[jp[1]])])
            if v0 < 2e-3 and v1 < 2e-3 and i > 10:
                quiet += 1
                if quiet >= int(0.2 * FRAME_HZ):
                    closed_ok = True
                    break
            else:
                quiet = 0
        bq, aq, p0, p1 = obs()
        if not finite(bq):
            fail("Grasping", "NaN in body state")
        else:
            sep = float(np.linalg.norm(p0[:3] - p1[:3])) - e
            jq = sim.joint_q()
            res["grasp_state"] = {"pad_separation": round(sep, 4), "finger_q": [round(float(jq[jq_idx["p0"]]), 4), round(float(jq[jq_idx["p1"]]), 4)], "closed_settled": closed_ok}
            pushed = float(np.linalg.norm(aq[0, :2] - obj_xy0))
            phase_log.append(f"t={sim.t:.1f}s Grasping sep={sep:.4f} finger_q={jq[jq_idx['p0']]:.4f},{jq[jq_idx['p1']]:.4f} obj_moved_xy={pushed:.4f}")
            if sep < 0.001:
                fail("Grasping", "Grasping failed: pads touched (no object)")
            else:
                if pushed > 0.02:
                    notes.append(f"asset pushed {pushed * 1000:.0f} mm sideways while closing")
                if not closed_ok:
                    notes.append("fingers were still moving when the closing timeout hit")

    if verdict is None:
        # Phase: Lifting (ramp z over LIFT_T)
        bq, aq, p0, p1 = obs()
        obj_z_grasp = float(aq[0, 2])
        # the material point of the root body that sits between the pads at grasp time;
        # slip = how far that point has moved away from the pad centre (rotation about
        # the pinch axis is not slip)
        pad_c0 = 0.5 * (p0[:3] + p1[:3])
        grasp_pt_local = tf_apply(tf_inv(aq[0]), pad_c0)
        slip_thr = 0.03
        res["slip_threshold_m"] = slip_thr

        def slip_now(aq, p0, p1):
            pc = 0.5 * (p0[:3] + p1[:3])
            gp_now = tf_apply(aq[0], grasp_pt_local)
            return float(np.linalg.norm(gp_now - pc)), float(pc[2] - gp_now[2])

        def held_check(bq, aq, p0, p1, ph):
            slip, vslip = slip_now(aq, p0, p1)
            res["max_slip_m"] = round(max(res.get("max_slip_m", 0.0), slip), 4)
            if slip > slip_thr:
                oz = float(aq[0, 2])
                fail(ph, f"{ph} failed: object dropped" if (oz < obj_z0 + RISE_MIN or vslip > 0.5 * lift) else f"{ph} failed: object slipped {slip:.3f} m in the jaws")
                return False
            return True

        def lift_ctrl(t):
            set_targets(z=lift * min(1.0, t / LIFT_T), p=q_close_max)

        ok = run_for(LIFT_T + 0.2, "Lifting", lift_ctrl, lambda bq, aq, p0, p1: held_check(bq, aq, p0, p1, "Lifting"))
        if ok:
            bq, aq, p0, p1 = obs()
            dz = float(aq[0, 2]) - obj_z0
            phase_log.append(f"t={sim.t:.1f}s Lifting obj_z={aq[0, 2]:.4f} dz={dz:.4f} pads_z={0.5 * (p0[2] + p1[2]):.4f} slip={slip_now(aq, p0, p1)[0]:.4f}")
            res["obj_z_after_lift"] = round(float(aq[0, 2]), 4)
            if dz < RISE_MIN:
                fail("Lifting", f"Lifting failed: object did not rise {RISE_MIN:.3f}m (actual_dz={dz:.4f}, obj_z={obj_z0:.4f}->{aq[0, 2]:.4f})")
        if verdict is None:
            ok = run_for(HOLD_T, "HoldBeforeShake", lambda t: set_targets(z=lift, p=q_close_max), lambda bq, aq, p0, p1: held_check(bq, aq, p0, p1, "HoldBeforeShake"))
            if ok:
                bq, aq, p0, p1 = obs()
                phase_log.append(f"t={sim.t:.1f}s HoldBeforeShake obj_z={aq[0, 2]:.4f}")
        if verdict is None:
            ok = run_for(SHAKE_T, "Shake", lambda t: set_targets(z=lift, x=SHAKE_AMP * math.sin(2 * math.pi * SHAKE_HZ * t), p=q_close_max), lambda bq, aq, p0, p1: held_check(bq, aq, p0, p1, "Shake"))
            if ok:
                bq, aq, p0, p1 = obs()
                phase_log.append(f"t={sim.t:.1f}s Shake obj_z={aq[0, 2]:.4f}")
        if verdict is None:
            ok = run_for(HOLD_T, "HoldAfterShake", lambda t: set_targets(z=lift, p=q_close_max), lambda bq, aq, p0, p1: held_check(bq, aq, p0, p1, "HoldAfterShake"))
            if ok:
                bq, aq, p0, p1 = obs()
                mz = obj_min_z(bq)
                phase_log.append(f"t={sim.t:.1f}s HoldAfterShake obj_z={aq[0, 2]:.4f} obj_min_z={mz:.4f} slip={slip_now(aq, p0, p1)[0]:.4f}")
                if "obj_z_after_lift" in res:
                    res["creep_in_jaws_m"] = round(res["obj_z_after_lift"] - float(aq[0, 2]), 4)
                if mz < 0.005:
                    fail("HoldAfterShake", f"HoldAfterShake failed: object touched ground (z={mz:.4f})")
        if verdict is None:
            bq, aq, p0, p1 = obs()
            z_before = float(aq[0, 2])
            ok = run_for(RELEASE_T, "Dropping", lambda t: set_targets(z=lift, p=0.0))
            if ok:
                bq, aq, p0, p1 = obs()
                fell = z_before - float(aq[0, 2])
                needed = 0.5 * (z_before - obj_z0)
                phase_log.append(f"t={sim.t:.1f}s Dropping fell={fell:.4f} needed={needed:.4f}")
                if fell < needed:
                    fail("Dropping", f"Dropping failed: fell {fell:.4f}m, needed {needed:.4f}m")
        if verdict is None:
            verdict, phase, message = "pass", "", "authored line held through lift, hold and shake"

    res["verdict"] = verdict
    res["phase"] = phase
    res["message"] = message
    res["solver"] = sim.solver_settings
    res["notes"] = notes
    res["log"] = phase_log
    res["sim_time_s"] = round(sim.t, 2)
    if viewer is not None:
        try:
            # a few extra frames after release so the drop is visible
            for _ in range(FRAME_HZ // 2):
                sim.frame()
                capture()
            import imageio

            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimwrite(video_path, frames, fps=VIDEO_FPS, codec="libx264", quality=6, macro_block_size=None)
            res["video"] = video_path
        except Exception as ex:  # noqa: BLE001
            res["video_error"] = f"{ex.__class__.__name__}: {str(ex)[:200]}"
        try:
            viewer.close()
        except Exception:  # noqa: BLE001
            pass
    res["time_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------
def worker(args):
    rel = args.single
    name = rel.split("/")[0]
    usd_path = os.path.join(args.assets_root, rel)
    out_json = os.path.join(args.out, "results", f"{name}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    result = {"name": name, "usd": rel, "engine": ENGINE_LABEL, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tests_requested": tests}
    if args.skip_done and os.path.exists(out_json):
        try:
            with open(out_json) as f:
                result = json.load(f)
            result["tests_requested"] = tests
        except Exception:  # noqa: BLE001
            pass

    def flush():
        tmp = out_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=1, default=float)
        os.replace(tmp, out_json)

    t_all = time.time()
    newton_setup()
    result["versions"] = versions()
    sidecar = read_sidecar(usd_path) or {}
    result["sidecar"] = sidecar
    try:
        stage_info = read_stage_info(usd_path)
    except Exception as e:  # noqa: BLE001
        stage_info = {"error": f"{e.__class__.__name__}: {e}"}
    result["stage"] = {k: v for k, v in stage_info.items() if k != "colliders_authored"}
    flush()

    asset_b, pinfo, geom = parse_asset(usd_path, stage_info, sidecar, args)
    pinfo["contact_timeconst"] = args.contact_timeconst
    pinfo["contact_ke_kd"] = list(contact_gains(args.contact_timeconst))
    result["parse"] = pinfo
    flush()
    log(f"[{name}] parse ok={pinfo.get('ok')} {pinfo.get('time_s')}s bodies={pinfo.get('body_count')} shapes={pinfo.get('collide_shape_count')} mass={pinfo.get('mass_total')} replaced={len(pinfo.get('approximations_replaced', []))}")
    if not pinfo.get("ok"):
        for t in ("ground_drop", "slope_drop", "grasp_and_lift"):
            if t in tests:
                result[t] = {"verdict": "skip", "message": "parse failed", "engine": ENGINE_LABEL}
        result["elapsed_s"] = round(time.time() - t_all, 1)
        flush()
        return

    def done(t):
        return args.skip_done and t in result and result[t].get("verdict") not in (None, "skip", "crash")

    if "ground_drop" in tests and not done("ground_drop"):
        try:
            result["ground_drop"] = run_drop(asset_b, geom, args, None)
        except Exception as e:  # noqa: BLE001
            result["ground_drop"] = {"verdict": "fail", "message": f"exception: {e.__class__.__name__}: {str(e)[:300]}", "traceback": traceback.format_exc()[-2000:]}
        log(f"[{name}] ground_drop {result['ground_drop'].get('verdict')} {result['ground_drop'].get('message')}")
        flush()
    if "slope_drop" in tests and not done("slope_drop"):
        try:
            result["slope_drop"] = run_drop(asset_b, geom, args, SLOPE_DEG)
        except Exception as e:  # noqa: BLE001
            result["slope_drop"] = {"verdict": "fail", "message": f"exception: {e.__class__.__name__}: {str(e)[:300]}", "traceback": traceback.format_exc()[-2000:]}
        log(f"[{name}] slope_drop {result['slope_drop'].get('verdict')} {result['slope_drop'].get('message')}")
        flush()
    if "grasp_and_lift" in tests and not done("grasp_and_lift"):
        video = None if args.no_video else os.path.join(args.out, "videos", f"{name}_grasp_and_lift.mp4")
        try:
            result["grasp_and_lift"] = run_grasp(asset_b, geom, pinfo, stage_info, args, video)
        except Exception as e:  # noqa: BLE001
            result["grasp_and_lift"] = {"verdict": "fail", "phase": "exception", "message": f"exception: {e.__class__.__name__}: {str(e)[:300]}", "traceback": traceback.format_exc()[-2000:]}
        g = result["grasp_and_lift"]
        log(f"[{name}] grasp_and_lift {g.get('verdict')} [{g.get('phase')}] {g.get('message')}")
        flush()
    result["elapsed_s"] = round(time.time() - t_all, 1)
    flush()


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def driver(args):
    with open(args.manifest) as f:
        rels = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        rels = [r for r in rels if r.split("/")[0] in keep]
    if args.limit:
        rels = rels[: args.limit]
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    summary_path = os.path.join(args.out, "results.json")
    t_start = time.time()
    for k, rel in enumerate(rels):
        name = rel.split("/")[0]
        out_json = os.path.join(args.out, "results", f"{name}.json")
        if args.skip_done and os.path.exists(out_json):
            try:
                with open(out_json) as f:
                    prev = json.load(f)
                if all(t in prev and prev[t].get("verdict") not in (None, "crash") for t in tests) and prev.get("parse", {}).get("ok") is not None:
                    log(f"[{k + 1}/{len(rels)}] {name}: done, skipping")
                    continue
            except Exception:  # noqa: BLE001
                pass
        cmd = [sys.executable, os.path.abspath(__file__), "--single", rel] + passthrough(args)
        logp = os.path.join(args.out, "logs", f"{name}.log")
        t0 = time.time()
        log(f"[{k + 1}/{len(rels)}] {name}: start ({time.strftime('%H:%M:%S')})")
        rc = None
        timed_out = False
        with open(logp, "w") as lf:
            try:
                p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.timeout)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
        dt = time.time() - t0
        if rc != 0 or timed_out:
            prev = {}
            if os.path.exists(out_json):
                try:
                    with open(out_json) as f:
                        prev = json.load(f)
                except Exception:  # noqa: BLE001
                    prev = {}
            tail = ""
            try:
                with open(logp, errors="replace") as f:
                    tail = "".join(f.readlines()[-25:])[-2500:]
            except Exception:  # noqa: BLE001
                pass
            prev.setdefault("name", name)
            prev.setdefault("usd", rel)
            prev.setdefault("engine", ENGINE_LABEL)
            prev["crash"] = {"returncode": rc, "timed_out": timed_out, "log_tail": tail, "elapsed_s": round(dt, 1)}
            for t in tests:
                if t not in prev or prev[t].get("verdict") is None:
                    prev[t] = {"verdict": "crash", "message": "worker timed out" if timed_out else f"worker exited {rc}", "engine": ENGINE_LABEL}
            if "parse" not in prev:
                prev["parse"] = {"ok": False, "error": "worker crashed before parse finished"}
            with open(out_json, "w") as f:
                json.dump(prev, f, indent=1)
            log(f"    CRASH rc={rc} timed_out={timed_out} ({dt:.0f}s)")
        else:
            try:
                with open(out_json) as f:
                    r = json.load(f)
                vs = " ".join(f"{t}={r.get(t, {}).get('verdict')}" for t in tests)
                log(f"    parse_ok={r.get('parse', {}).get('ok')} {vs} ({dt:.0f}s)")
            except Exception as e:  # noqa: BLE001
                log(f"    done ({dt:.0f}s) but result unreadable: {e}")
        if (k + 1) % 5 == 0 or k + 1 == len(rels):
            aggregate(args.out, summary_path)
            el = time.time() - t_start
            log(f"    progress {k + 1}/{len(rels)} elapsed {el / 60:.1f} min, eta {(el / (k + 1)) * (len(rels) - k - 1) / 60:.1f} min")
    aggregate(args.out, summary_path)
    log("DRIVER_DONE")


def passthrough(args):
    out = ["--assets-root", args.assets_root, "--out", args.out, "--tests", args.tests, "--nconmax", str(args.nconmax), "--njmax", str(args.njmax)]
    if args.no_video:
        out.append("--no-video")
    if args.no_graph:
        out.append("--no-graph")
    if args.no_multiccd:
        out.append("--no-multiccd")
    if args.cone:
        out += ["--cone", args.cone]
    if args.impratio is not None:
        out += ["--impratio", str(args.impratio)]
    if args.skip_done:
        out.append("--skip-done")
    out += ["--line-frame", args.line_frame, "--contact-timeconst", str(args.contact_timeconst), "--finger-armature", str(args.finger_armature), "--pad-condim", str(args.pad_condim)]
    return out


def aggregate(out_dir, summary_path):
    rdir = os.path.join(out_dir, "results")
    allr = {}
    for fn in sorted(os.listdir(rdir)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(rdir, fn)) as f:
                    allr[fn[:-5]] = json.load(f)
            except Exception:  # noqa: BLE001
                pass
    tmp = summary_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"engine": ENGINE_LABEL, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "count": len(allr), "assets": allr}, f, indent=1)
    os.replace(tmp, summary_path)


def main():
    # usd-core's UsdPhysics parser races on a body with many collision prims (a fifth to a
    # quarter of the loads corrupt the heap with its default thread pool); single-threaded it
    # never does. Set before any pxr import, and inherited by the worker subprocesses.
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-root", required=True, help="directory holding the package folders")
    ap.add_argument("--manifest", help="text file, one package USD path (relative to --assets-root) per line")
    ap.add_argument("--single", help="worker mode: relative USD path of one package")
    ap.add_argument("--out", required=True, help="output directory (results/, logs/, videos/, results.json)")
    ap.add_argument("--tests", default="ground_drop,slope_drop,grasp_and_lift")
    ap.add_argument("--only", help="comma-separated package names to run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-done", action="store_true", help="skip assets whose result JSON already carries the requested tests")
    ap.add_argument("--timeout", type=int, default=900, help="per-asset worker timeout [s]")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-graph", action="store_true", help="do not CUDA-graph the substeps")
    ap.add_argument("--cone", default="elliptic", help="MuJoCo friction cone (pyramidal|elliptic). Newton's default is pyramidal; the cert uses elliptic with impratio 10 so held objects do not creep out of the jaws")
    ap.add_argument("--impratio", type=float, default=10.0)
    ap.add_argument("--line-frame", default="stage", choices=("stage", "body"), help="grasp line placement after settling: stage (NVIDIA semantics, default) or body")
    ap.add_argument("--contact-timeconst", type=float, default=CONTACT_TIMECONST, help="MuJoCo solref time constant applied to every shape [s]")
    ap.add_argument("--finger-armature", type=float, default=FINGER_ARMATURE, help="armature on the two finger DOFs [kg]")
    ap.add_argument("--pad-condim", type=int, default=4, help="MuJoCo condim of the pad geoms (4 = torsional friction, like NVIDIA's torsional patch; 3 = stock)")
    ap.add_argument("--no-multiccd", action="store_true", help="single contact point per geom pair (MuJoCo-Warp default); the cert enables multi-CCD (up to 4 points)")
    ap.add_argument("--nconmax", type=int, default=4000)
    ap.add_argument("--njmax", type=int, default=8000)
    args = ap.parse_args()
    if args.single:
        worker(args)
    else:
        if not args.manifest:
            ap.error("--manifest is required in driver mode")
        driver(args)


if __name__ == "__main__":
    main()
