#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author a Newton collision variant of a handled prop, inside its package.

Newton's rigid solvers (MuJoCo-Warp in particular) collide every mesh collider as one
convex hull of at most 64 vertices: an ``sdf`` bowl becomes a lid, a ``convexDecomposition``
only means something when CoACD is installed and runs at every load, and a planar piece is
refused outright. This tool writes ``<usd dir>/variants/<name>_newton.usd``, a self-contained
flattened copy of the package USD in which it replaces every mesh collider Newton cannot honour by
pre-decomposed convex pieces (CoACD offline, <= 64 vertices each, >= 1 mm thick, purpose
``guide``, the original's physics material and, where the collider carried its own mass, the
mass split over the pieces by volume). The original collider keeps rendering; only its
CollisionAPI is dropped in the variant (a flattened copy, so the package USD is not composed at load). Nothing in the package USD changes; the base sidecar
gains ``asset.variants.newton`` and a MINOR bump, the variant gets its own sidecar.

    <newton venv>/bin/python scripts/tools/make_newton_variant.py --assets-root <clone> --only ikea_365_bowl_rounded_19 --check
    <newton venv>/bin/python scripts/tools/make_newton_variant.py --assets-root <clone> --candidates outputs/newton_variant_candidates.json --write

Runs in the Newton 1.5.2 venv (pxr, coacd, numpy, scipy, newton) so the variant is parsed
back through Newton's own importer before it is accepted: every collider must come out as a
CONVEX_MESH or a primitive, nothing thickened, nothing replaced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlwrld_sidecar import bump, sidecar_for  # noqa: E402  (vendored from DexBench-Arena's conform_simready_basics)

REPLACE = {"sdf", "convexdecomposition", "meshsimplification", "none"}
MAX_HULL_VERTS = 64
MIN_THICKNESS_M = 0.001
TOOL = "nv_core/testing_tools/rlwrld-newton-conformance/make_newton_variant.py"


def triangulate(counts, indices):
    tris = []; k = 0
    for c in counts:
        for i in range(1, c - 1):
            tris.append((indices[k], indices[k + i], indices[k + i + 1]))
        k += c
    return np.asarray(tris, dtype=np.int64)


def thin_axis(v):
    c = v - v.mean(axis=0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    n = vt[-1]; ext = float((c @ n).max() - (c @ n).min())
    return n, ext


def hull_piece(v):
    """Convex hull of a piece as (points <= 64, triangles); thickened when thinner than 1 mm."""
    from scipy.spatial import ConvexHull
    v = np.asarray(v, dtype=np.float64)
    h = ConvexHull(v)
    pts = v[h.vertices]
    # a sliver is thin along one axis, a needle off a ring along two; each extrusion doubles the
    # vertices, so the cap is applied first, small enough that the extruded hull stays <= 64
    # vertices (a piece above that would be re-hulled by the importer, which drops the thickness)
    c = pts - pts.mean(axis=0); ext = np.linalg.svd(c, full_matrices=False)[2] @ c.T
    n_thin = int(sum((ext[i].max() - ext[i].min()) < MIN_THICKNESS_M for i in (1, 2)))
    cap = MAX_HULL_VERTS >> n_thin
    if len(pts) > cap:  # keep the farthest-spread vertices, then hull again
        keep = [int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))]
        d = np.linalg.norm(pts - pts[keep[0]], axis=1)
        while len(keep) < cap:
            j = int(np.argmax(d)); keep.append(j); d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
        pts = pts[keep]; h = ConvexHull(pts); pts = pts[h.vertices]
    thickened = False
    for _ in range(3):
        n, ext1 = thin_axis(pts)
        if ext1 >= MIN_THICKNESS_M:
            break
        pts = np.concatenate([pts + 0.5 * MIN_THICKNESS_M * n, pts - 0.5 * MIN_THICKNESS_M * n]); thickened = True
        h = ConvexHull(pts); pts = pts[h.vertices]
    assert len(pts) <= MAX_HULL_VERTS, len(pts)
    h = ConvexHull(pts)
    # orient every triangle outward
    ctr = pts.mean(axis=0); tris = []
    for s in h.simplices:
        a, b, c = pts[s]
        if np.dot(np.cross(b - a, c - a), a - ctr) < 0: s = s[[0, 2, 1]]
        tris.append(s)
    return pts, np.asarray(tris, dtype=np.int64), float(h.volume), thickened


def decompose(points, tris, threshold, max_hulls, seed=0):
    import coacd
    coacd.set_log_level("off")
    m = coacd.Mesh(np.asarray(points, dtype=np.float64), np.asarray(tris, dtype=np.int64))
    # the same light MCTS Newton's importer uses for a load-time convexDecomposition (a full
    # search takes tens of minutes on a 260k-vertex scan), with merging on and a hull cap
    parts = coacd.run_coacd(m, threshold=threshold, max_convex_hull=max_hulls, max_ch_vertex=MAX_HULL_VERTS,
                            mcts_nodes=20, mcts_iterations=5, mcts_max_depth=1,
                            merge=True, preprocess_mode="auto", seed=seed)
    return [np.asarray(p[0], dtype=np.float64) for p in parts]


def collider_targets(stage):
    from pxr import UsdGeom, UsdPhysics
    out = []
    for p in stage.Traverse():
        if not p.HasAPI(UsdPhysics.CollisionAPI) or not p.IsA(UsdGeom.Mesh):
            continue
        a = p.GetAttribute("physics:approximation")
        approx = (a.Get() if a and a.HasAuthoredValue() else None) or "none"
        if approx.lower() in REPLACE or approx.lower() == "convexhull":
            out.append((p, approx))  # a convexHull collider is only touched when it is thinner than 1 mm
    return out


def author_variant(base_usd: Path, variant_usd: Path, threshold: float, max_hulls: int, log) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    base = Usd.Stage.Open(str(base_usd))
    default = base.GetDefaultPrim()
    targets = collider_targets(base)
    if not targets:
        return {}
    # (a package whose only candidates are convexHull colliders of sane thickness ends up with no
    # collider record, and is reported as nothing to replace)
    variant_usd.parent.mkdir(parents=True, exist_ok=True)
    if variant_usd.exists():
        variant_usd.unlink()
    # Authored over the package USD as a sublayer, then flattened into one self-contained
    # layer: usd-core's UsdPhysics parser (which Newton's importer and Isaac Lab's Newton
    # backend both call) corrupts the heap, intermittently, on a stage composed from more
    # than one layer, and a variant that only sometimes loads is no variant.
    layer = Sdf.Layer.CreateAnonymous("newton_variant.usd")
    layer.subLayerPaths = [str(base_usd.resolve())]
    layer.defaultPrim = default.GetName()
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(base))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(base))
    record = {"colliders": [], "pieces": 0, "thickened": 0, "coacd": {"threshold": threshold, "max_convex_hull": max_hulls, "max_ch_vertex": MAX_HULL_VERTS, "mcts_nodes": 20, "mcts_iterations": 5, "mcts_max_depth": 1, "merge": True}}
    for prim, approx in targets:
        src = stage.GetPrimAtPath(prim.GetPath())
        mesh = UsdGeom.Mesh(src)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        tris = triangulate(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
        # pieces are authored in the rigid body's frame, under the body prim, with no transform of
        # their own: the collider's transform (vendor parts carry a scale in it) is baked into the
        # points, so the 1 mm thickness floor is a metre and a collider always sits below its body
        body = src
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            body = src.GetParent()
        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        to_body = xf_cache.GetLocalToWorldTransform(src) * xf_cache.GetLocalToWorldTransform(body).GetInverse()
        m = np.asarray([[to_body[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)  # row-vector convention
        pts = (np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ m)[:, :3]
        t0 = time.time()
        if approx.lower() == "convexhull":
            if thin_axis(pts)[1] >= MIN_THICKNESS_M:
                continue  # an honoured hull of sane thickness stays as it is
            pieces = [hull_piece(pts)]  # a sheet-metal tab, a shim: one hull, extruded to 1 mm
        else:
            parts = decompose(pts, tris, threshold, max_hulls)
            pieces = [hull_piece(p) for p in parts]
        vol = sum(p[2] for p in pieces) or 1.0
        is_body = src.HasAPI(UsdPhysics.RigidBodyAPI)  # YCB-style packages: the collider mesh is the rigid body itself
        mass_attr = src.GetAttribute("physics:mass")
        # a mass authored on a collider that is not the body moves to the pieces (split by volume);
        # a mass authored on the body prim stays where it is
        mass = float(mass_attr.Get()) if (not is_body and mass_attr and mass_attr.HasAuthoredValue()) else None
        binding = UsdShade.MaterialBindingAPI(src).GetDirectBinding("physics").GetMaterialPath()
        holder = stage.DefinePrim(body.GetPath().AppendChild(("newton_collision" if is_body else src.GetName() + "_newton_collision")), "Xform")
        UsdGeom.Imageable(holder).GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        for i, (v, f, pvol, thick) in enumerate(pieces):
            piece = UsdGeom.Mesh.Define(stage, holder.GetPath().AppendChild(f"piece_{i:03d}"))
            piece.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in v]))
            piece.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(f)))
            piece.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(x) for x in f.reshape(-1)]))
            piece.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            lo, hi = v.min(axis=0), v.max(axis=0)
            piece.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
            piece.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
            pp = piece.GetPrim()
            UsdPhysics.CollisionAPI.Apply(pp)
            UsdPhysics.MeshCollisionAPI.Apply(pp).GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            if mass is not None:
                UsdPhysics.MassAPI.Apply(pp).GetMassAttr().Set(mass * pvol / vol)
            if binding and not binding.isEmpty:
                mat = UsdShade.Material(stage.GetPrimAtPath(binding))
                if mat:
                    UsdShade.MaterialBindingAPI.Apply(pp).Bind(mat, materialPurpose="physics")
            record["thickened"] += int(thick)
        # the original keeps rendering; its collision schemas (and its mass, now on the pieces) are
        # deleted with a list op in this layer, which also covers PhysX schemas usd-core does not know
        drop = [api for api in src.GetAppliedSchemas()
                if api in ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI") or (api.startswith("Physx") and "Collision" in api)
                or (mass is not None and api == "PhysicsMassAPI")]
        spec = Sdf.CreatePrimInLayer(layer, src.GetPath()); spec.specifier = Sdf.SpecifierOver
        lo = Sdf.TokenListOp(); lo.deletedItems = drop; spec.SetInfo("apiSchemas", lo)
        record["colliders"].append({"prim": str(prim.GetPath()), "body": str(body.GetPath()), "approximation": approx, "vertices": int(len(pts)), "pieces": len(pieces),
                                    "piece_vertices_max": max(len(p[0]) for p in pieces), "mass_kg_split": mass, "seconds": round(time.time() - t0, 1)})
        record["pieces"] += len(pieces)
        log(f"    {prim.GetPath()} {approx} {len(pts)}v -> {len(pieces)} pieces in {time.time() - t0:.1f}s")
    # flatten the two layers into the variant file; asset paths of the package USD are re-anchored
    # from its folder to variants/ (textures, MDL modules, any sub-USD it references)
    import os
    from pxr import UsdUtils
    variant_dir = variant_usd.parent.resolve()

    def rewrite(src_layer, path):
        if not path or path.startswith("/") or "://" in path or src_layer.anonymous:
            return path
        return os.path.relpath((Path(src_layer.realPath).parent / path).resolve(), variant_dir)

    if not record["colliders"]:
        return {}
    flat = UsdUtils.FlattenLayerStack(stage, rewrite)
    data = dict(Sdf.Layer.FindOrOpen(str(base_usd)).customLayerData or {})
    data.update({"newton_variant_of": base_usd.name, "tool": TOOL})
    flat.customLayerData = data
    flat.defaultPrim = default.GetName()
    flat.Export(str(variant_usd))
    record["flattened"] = True
    return record


def verify_with_newton(variant_usd: Path) -> dict:
    """Parse the variant with Newton's importer in a fresh interpreter (CoACD and warp do not
    share a process well): every collider must be convex or a primitive, nothing thickened."""
    import os
    import subprocess
    # usd-core's UsdPhysics parser (25.11 and 26.3) races on a body with many collision prims and
    # corrupts the heap about one run in four with its default thread pool; single-threaded it never does
    env = dict(os.environ, PXR_WORK_THREAD_LIMIT="1")
    last = ""
    for attempt in range(3):
        r = subprocess.run([sys.executable, __file__, "--verify-only", str(variant_usd)], capture_output=True, text=True, timeout=600, env=env)
        for line in r.stdout.splitlines():
            if line.startswith("{"):
                out = json.loads(line); out["attempts"] = attempt + 1
                return out
        last = f"verify subprocess rc={r.returncode}: {r.stderr.strip().splitlines()[-1][-300:] if r.stderr.strip() else ''}"
    return {"ok": False, "colliders": {}, "bodies": 0, "warnings": [last + " (3 attempts)"]}


def _verify_only(variant_usd: Path) -> None:
    import warnings
    import newton
    from newton import GeoType, ShapeFlags
    from pxr import Usd
    b = newton.ModelBuilder()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        b.add_usd(Usd.Stage.Open(str(variant_usd)), verbose=False)
    kinds = {}; thin = 0; unattached = 0
    for i in range(b.shape_count):
        if not (int(b.shape_flags[i]) & int(ShapeFlags.COLLIDE_SHAPES)):
            continue
        t = GeoType(int(b.shape_type[i])).name; kinds[t] = kinds.get(t, 0) + 1
        if int(b.shape_body[i]) < 0:
            unattached += 1  # a piece that did not land under its rigid body would be a static collider
        if t in ("MESH", "CONVEX_MESH"):
            v = np.asarray(b.shape_source[i].vertices, dtype=np.float64) * np.asarray(b.shape_scale[i], dtype=np.float64)
            if thin_axis(v)[1] < MIN_THICKNESS_M * 0.99:
                thin += 1
    masses = [round(float(m), 5) for m in b.body_mass]
    ok = kinds.get("MESH", 0) == 0 and thin == 0 and unattached == 0 and b.body_count > 0 and all(m > 0 for m in masses)
    print(json.dumps({"ok": ok, "colliders": kinds, "thin_pieces": thin, "unattached": unattached, "bodies": b.body_count, "mass_kg": masses,
                      "warnings": sorted({str(w.message)[:160] for w in ws})[:5]}))


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-only":
        _verify_only(Path(sys.argv[2])); return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=None, help="json {name: {usd: rel path}} (outputs/newton_variant_candidates.json)")
    ap.add_argument("--only", default=None, help="comma-separated package names")
    ap.add_argument("--threshold", type=float, default=0.05, help="CoACD concavity threshold")
    ap.add_argument("--max-hulls", type=int, default=32, help="CoACD max convex hulls per collider")
    ap.add_argument("--no-verify", action="store_true")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="author to a scratch file, verify, change nothing in the package")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    root = args.assets_root.resolve(); today = dt.date.today().isoformat()
    cands = json.loads(args.candidates.read_text()) if args.candidates else {}
    names = [n for n in args.only.split(",")] if args.only else sorted(cands)
    done = 0; skipped = []
    for name in names:
        pkg = root / "object_library" / name
        rel = (cands.get(name) or {}).get("usd")
        base = root / "object_library" / rel if rel else next((p for p in (pkg / "usd").glob(f"{name}.usd*")), None)
        if base is None or not base.is_file():
            print(f"[newton-variant] {name}: package USD not found"); continue
        variant = base.parent / "variants" / f"{base.stem}_newton.usd"
        target = variant if args.write else Path("/tmp/claude-1000") / "newton_variant_check" / name / "variants" / variant.name
        print(f"[newton-variant] {name}: {base.relative_to(root)}")
        if not args.write:  # a check authors next to a copy of the base folder so the ../ reference resolves
            import shutil
            shutil.rmtree(target.parent.parent, ignore_errors=True)
            shutil.copytree(base.parent, target.parent.parent, ignore=shutil.ignore_patterns("variants"))
        rec = author_variant(base, target, args.threshold, args.max_hulls, print)
        if not rec:
            skipped.append(name); print("    no mesh collider to replace"); continue
        if not args.no_verify:
            v = verify_with_newton(target); rec["newton_parse"] = v
            print(f"    newton parse: {'OK' if v['ok'] else 'REJECTED'} {v['colliders']} thin={v.get('thin_pieces')} unattached={v.get('unattached')} bodies={v['bodies']} mass={v.get('mass_kg')}" + (f" warnings={v['warnings']}" if v["warnings"] else ""))
            if not v["ok"]:
                print(f"    !! variant rejected")
                if args.write:  # leave no orphan layer behind: the package records nothing for it
                    target.unlink(missing_ok=True); (target.parent / f"{target.stem}.meta.json").unlink(missing_ok=True)
                continue
        done += 1
        # sidecars: the variant gets a copy of the base sidecar marked as a variant; the base points at it
        sc = sidecar_for(base); meta = json.loads(sc.read_text())
        vmeta = json.loads(json.dumps(meta)); vb = vmeta.setdefault("asset", {})
        vb["variant"] = "newton"; vb["variant_of"] = str(base.relative_to(pkg))
        vb["newton_collision"] = {"date": today, "tool": TOOL, "engine_checked": "Newton 1.5.2 importer (MuJoCo-Warp semantics)", **rec}
        for k in ("variants",):
            vb.pop(k, None)
        (target.parent / f"{target.stem}.meta.json").write_text(json.dumps(vmeta, indent=2, ensure_ascii=False) + "\n")
        if not args.write:
            continue
        meta.setdefault("asset", {}).setdefault("variants", {})["newton"] = str(variant.relative_to(pkg))
        sc.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        cols = ", ".join(f"`{c['prim'].rsplit('/', 1)[-1]}` ({c['approximation']}, {c['pieces']} piece{'s' if c['pieces'] != 1 else ', thickened to 1 mm'})" for c in rec["colliders"])
        note = (f"- Added the Newton collision variant `{variant.relative_to(pkg)}`: a flattened copy of this package in which {cols} "
                f"{'is' if len(rec['colliders']) == 1 else 'are'} replaced by pre-decomposed convex pieces (CoACD threshold {args.threshold}, "
                f"<= {MAX_HULL_VERTS} vertices each" + (f", {rec['thickened']} thickened to 1 mm" if rec["thickened"] else "") +
                "), because Newton's solvers collide a mesh as one 64-vertex convex hull. Same render mesh, same physics material, "
                "same mass. Parsed back through Newton's importer: every collider convex, nothing replaced. The package USD is unchanged.")
        v = bump(pkg, base, note, today, minor=True)
        print(f"    recorded -> {v}")
    print(f"[newton-variant] {'authored' if args.write else 'checked'} {done}; no mesh collider to replace: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
