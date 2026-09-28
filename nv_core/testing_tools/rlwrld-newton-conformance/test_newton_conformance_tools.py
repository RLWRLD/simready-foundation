#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Self-checks for the RLWRLD Newton conformance tools on synthetic stages.

Needs usd-core, numpy and scipy only (no Newton, no CoACD, no GPU): the variant checks use
convexHull colliders, which never reach CoACD, and skip the Newton parse. Runs under pytest or
as a plain script:

    python test_newton_conformance_tools.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_newton_variant as mnv  # noqa: E402
import newton15_cert as cert  # noqa: E402


def _thickness(v):
    return mnv.thin_axis(np.asarray(v, dtype=np.float64))[1]


# ------------------------------------------------------------------ hull_piece on degenerate input
def test_hull_piece_planar_quad():
    quad = np.array([[0, 0, 0], [0.05, 0, 0], [0.05, 0.03, 0], [0, 0.03, 0]], dtype=np.float64)
    pts, tris, vol, thick = mnv.hull_piece(quad)
    assert thick and len(pts) <= mnv.MAX_HULL_VERTS and len(tris) > 0 and vol > 0
    assert _thickness(pts) >= mnv.MIN_THICKNESS_M


def test_hull_piece_planar_disc_is_capped():
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    disc = np.stack([0.1 * np.cos(a), 0.1 * np.sin(a), np.zeros_like(a)], axis=1)
    disc = disc @ cert.q_rot(cert.q_axis_angle([1, 1, 0], 0.7), np.eye(3)).T  # tilted plane
    pts, _, _, thick = mnv.hull_piece(disc)
    assert thick and len(pts) <= mnv.MAX_HULL_VERTS
    assert _thickness(pts) >= mnv.MIN_THICKNESS_M


def test_hull_piece_collinear():
    line = np.array([[0, 0, 0], [0.02, 0.01, 0.0], [0.04, 0.02, 0.0]], dtype=np.float64)
    pts, _, vol, thick = mnv.hull_piece(line)
    assert thick and len(pts) <= mnv.MAX_HULL_VERTS and vol > 0
    assert _thickness(pts) >= mnv.MIN_THICKNESS_M


def test_hull_piece_solid_unchanged():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(500, 3)) * [0.03, 0.02, 0.01]
    pts, _, _, thick = mnv.hull_piece(cloud)
    assert not thick and len(pts) <= mnv.MAX_HULL_VERTS


# ------------------------------------------------------------------ variant authoring on a synthetic package
QUAD = "[(-0.02, -0.01, 0), (0.02, -0.01, 0), (0.02, 0.01, 0), (-0.02, 0.01, 0)]"


def _collider(name, xyz, enabled=True):
    en = "" if enabled else "\n            bool physics:collisionEnabled = 0"
    return f'''
        def Xform "{name}"
        {{
            double3 xformOp:translate = {xyz}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Mesh "mesh" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
            )
            {{
                int[] faceVertexCounts = [4]
                int[] faceVertexIndices = [0, 1, 2, 3]
                point3f[] points = {QUAD}
                uniform token physics:approximation = "convexHull"{en}
            }}
        }}'''


def _package(root: Path, name="pkg") -> Path:
    usd = root / "object_library" / name / "usd" / f"{name}.usda"
    usd.parent.mkdir(parents=True)
    usd.write_text(f'''#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = 0.1
    def Scope "Looks"
    {{
        def Material "Rubber" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = 0.9
            float physics:dynamicFriction = 0.8
        }}
    }}
    def Xform "parts" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </Root/Looks/Rubber>{_collider("a", "(0, 0, 0)")}{_collider("b", "(0, 0, 0.05)")}{_collider("c", "(0, 0, 0.10)", enabled=False)}
    }}
}}
''')
    (usd.parent / f"{name}.meta.json").write_text(json.dumps({"asset": {"version": "1.0.0"}}) + "\n")
    return usd


def test_same_named_colliders_get_their_own_holders():
    from pxr import Usd, UsdPhysics, UsdShade
    with tempfile.TemporaryDirectory() as d:
        base = _package(Path(d))
        out = base.parent / "variants" / "pkg_newton.usd"
        rec = mnv.author_variant(base, out, 0.05, 32, lambda *a: None)
        prims = sorted(c["prim"] for c in rec["colliders"])
        # the disabled collider (c) is not replaced; a and b, both leaf-named "mesh", each get one
        assert prims == ["/Root/parts/a/mesh", "/Root/parts/b/mesh"], prims
        holders = [c["holder"] for c in rec["colliders"]]
        assert len(set(holders)) == 2, holders
        assert mnv.check_holders(out, rec) == []
        st = Usd.Stage.Open(str(out))
        for c in rec["colliders"]:
            pieces = st.GetPrimAtPath(c["holder"]).GetChildren()
            assert len(pieces) == c["pieces"] == 1
            for q in pieces:
                assert q.HasAPI(UsdPhysics.CollisionAPI)
                # the material bound on an ancestor between body and collider (inherited by the collider,
                # not by the holder, which sits directly under the body) is carried to the piece
                mat = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial(materialPurpose="physics")[0]
                assert mat and str(mat.GetPath()) == "/Root/Looks/Rubber"
            assert c["physics_material"] == "/Root/Looks/Rubber"
        # the disabled collider keeps its CollisionAPI and gets no holder
        assert st.GetPrimAtPath("/Root/parts/c/mesh").HasAPI(UsdPhysics.CollisionAPI)
        # the acceptance check notices a holder shared by two colliders
        shared = json.loads(json.dumps(rec))
        shared["colliders"][1]["holder"] = shared["colliders"][0]["holder"]
        assert any("shared" in p for p in mnv.check_holders(out, shared))


def _run_main(argv):
    old = sys.argv
    sys.argv = ["make_newton_variant.py", *argv]
    try:
        return mnv.main()
    finally:
        sys.argv = old


def test_rejected_regeneration_keeps_the_accepted_variant():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        base = _package(root)
        variant = base.parent / "variants" / "pkg_newton.usd"
        variant.parent.mkdir()
        variant.write_text("#usda 1.0\n# previously accepted variant\n")
        orig = mnv.verify_with_newton
        try:
            mnv.verify_with_newton = lambda p: {"ok": False, "colliders": {}, "bodies": 0, "warnings": ["forced"]}
            _run_main(["--assets-root", str(root), "--only", "pkg", "--write"])
            assert variant.read_text().endswith("previously accepted variant\n")
            assert not list(variant.parent.glob(".*.tmp.usd"))
            meta = json.loads((base.parent / "pkg.meta.json").read_text())
            assert "variants" not in meta["asset"] and meta["asset"]["version"] == "1.0.0"
            mnv.verify_with_newton = lambda p: {"ok": True, "colliders": {"CONVEX_MESH": 2}, "bodies": 1, "warnings": []}
            _run_main(["--assets-root", str(root), "--only", "pkg", "--write"])
            assert "previously accepted" not in variant.read_bytes().decode(errors="replace")
            assert not list(variant.parent.glob(".*.tmp.usd"))
            meta = json.loads((base.parent / "pkg.meta.json").read_text())
            assert meta["asset"]["variants"]["newton"] == "usd/variants/pkg_newton.usd"
            assert meta["asset"]["version"] == "1.1.0"
        finally:
            mnv.verify_with_newton = orig


# ------------------------------------------------------------------ newton15_cert: placement and driver
def test_slope_clearance_is_measured_after_centring():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.05, 0.05, size=(200, 3)) + [0.4, -0.2, 0.03]  # off-centre asset
    geom = {"coll_pts": pts, "coll_min": pts.min(axis=0), "coll_max": pts.max(axis=0)}
    for deg in (0.0, 15.0, 45.0):
        q = cert.q_axis_angle([0, 1, 0], np.radians(deg))
        n = cert.q_rot(q, [0.0, 0.0, 1.0])
        off = cert.place_offset(geom, n, np.zeros(3), cert.DROP_HEIGHT)
        clearance = float(((pts + off) @ n).min())
        assert abs(clearance - cert.DROP_HEIGHT) < 1e-9, (deg, clearance)


def test_driver_does_not_keep_a_stale_result_when_the_worker_dies_early():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        out = d / "out"
        (out / "results").mkdir(parents=True)
        stale = {"name": "pkg", "parse": {"ok": True}, "ground_drop": {"verdict": "pass"}, "slope_drop": {"verdict": "pass"}, "grasp_and_lift": {"verdict": "pass"}}
        (out / "results" / "pkg.json").write_text(json.dumps(stale))
        (d / "manifest.txt").write_text("pkg/usd/pkg.usda\n")
        fake = d / "fake"
        fake.mkdir()
        (fake / "warp.py").write_text("raise SystemExit(3)  # a worker that dies before its first write\n")
        env = dict(os.environ, PYTHONPATH=str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        subprocess.run([sys.executable, str(HERE / "newton15_cert.py"), "--assets-root", str(d), "--manifest", str(d / "manifest.txt"), "--out", str(out), "--no-video"],
                       env=env, check=True, capture_output=True, timeout=120)
        r = json.loads((out / "results" / "pkg.json").read_text())
        assert r["crash"]["returncode"] == 3
        for t in ("ground_drop", "slope_drop", "grasp_and_lift"):
            assert r[t]["verdict"] == "crash", (t, r[t])
        assert r["parse"]["ok"] is False


# ------------------------------------------------------------------ report: derived drop sentence, crash labels
def _report(rows: dict, d: Path) -> str:
    names = list(rows)
    (d / "manifest.txt").write_text("".join(f"{n}/usd/{n}.usd\n" for n in names))
    (d / "main.json").write_text(json.dumps({"assets": rows}))
    (d / "physx.json").write_text("{}")
    (d / "fet.json").write_text("{}")
    subprocess.run([sys.executable, str(HERE / "newton15_cert_report.py"), "--main", str(d / "main.json"), "--physx", str(d / "physx.json"),
                    "--fet003", str(d / "fet.json"), "--assets-root", str(d), "--manifest", str(d / "manifest.txt"), "--out-dir", str(d / "rep")],
                   check=True, capture_output=True, timeout=120)
    return (d / "rep" / "summary.md").read_text()


def _row(name, gd=None, crash=None, earlier=None):
    r = {"name": name, "parse": {"ok": True}, "ground_drop": gd or {"verdict": "pass", "message": ""}, "slope_drop": {"verdict": "pass", "message": ""},
         "grasp_and_lift": {"verdict": "pass"}}
    if crash:
        r["crash"] = crash
    if earlier:
        r["crash_earlier_attempts"] = [earlier]
    return r


def test_report_drop_sentence_and_crash_labels():
    with tempfile.TemporaryDirectory() as d:
        md = _report({"a": _row("a"), "b": _row("b")}, Path(d))
        assert "No NaN, fly-away or tunnelling on any package that ran the drops." in md
    with tempfile.TemporaryDirectory() as d:
        rows = {"a": _row("a", gd={"verdict": "fail", "message": "NaN in body state", "nan": True}),
                "b": _row("b", gd={"verdict": "fail", "message": "tunnelled into the support (12.0 mm)"}),
                "c": _row("c", crash={"returncode": -11, "timed_out": False, "log_tail": "Segmentation fault"}),
                "e": _row("e", earlier={"returncode": -6, "timed_out": False, "log_tail": ""})}
        md = _report(rows, Path(d))
        assert "No NaN" not in md
        assert "NaN on a" in md and "tunnelling on b" in md
        assert "2 worker crashes" not in md and "1 worker crashes (1 after every verdict was written), 1 recovered on a rerun" in md
        assert "c: worker crash rc=-11" in md and "e: worker aborted on an earlier attempt" in md
        assert "polybag_3" not in md.split("## Parse failures and crashes")[1].split("##")[0]


if __name__ == "__main__":
    failed = 0
    for k, fn in sorted(globals().items()):
        if k.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {k}")
            except Exception as e:  # noqa: BLE001
                import traceback
                failed += 1
                print(f"FAIL {k}: {e.__class__.__name__}: {e}")
                traceback.print_exc()
    print(f"{'all passed' if not failed else f'{failed} failed'}")
    sys.exit(1 if failed else 0)
