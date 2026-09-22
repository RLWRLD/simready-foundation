# SPDX-License-Identifier: Apache-2.0
"""What is this USD, and can it be evaluated?

    <bench>/.venv-isaac610/bin/python asset_checks/native/triage.py <file-or-directory>...

Read before running anything: what the asset declares itself to be, what it says it is made of and
where each value came from, and which of the environments could take it. Nothing is assumed and
nothing is written -- an asset that cannot be run says why here rather than after a Kit launch.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import asset_properties  # noqa: E402
from asset_checks import envs, experiments  # noqa: E402
from asset_checks.kit.scene import SOLVER_SIMULATES, asset_kinds  # noqa: E402


def elements(stage):
    """{prim path: (what it is, how many points, how many elements)} for the simulated geometry."""
    import usd_deformable

    out = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not usd_deformable.is_simulated(prim):
            continue
        points = UsdGeom.PointBased(prim).GetPointsAttr().Get() if prim.IsA(UsdGeom.PointBased) else None
        tets = UsdGeom.TetMesh(prim).GetTetVertexIndicesAttr().Get() if prim.IsA(UsdGeom.TetMesh) else None
        counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() if prim.IsA(UsdGeom.Mesh) else None
        kind = "tetrahedra" if tets else ("faces" if counts else "points")
        out[str(prim.GetPath())] = (kind, len(points or []), len(tets or counts or []))
    return out


# How small a tetrahedron has to be before PhysX will not cook a deformable body out of the mesh
# it belongs to. Measured, not quoted: the plum and the strawberry are geometrically perfect --
# no inverted tetrahedron, no zero volume, watertight, manifold, every point used -- and PhysX
# built no body from either, while the apple and the banana ran. The one property that orders with
# that is how many very small tetrahedra the mesh holds:
#
#     banana       0 below 1e-11 m^3   PhysX runs it
#     apple       12                   PhysX runs it
#     plum        43                   PhysX builds no body
#     strawberry  96                   PhysX builds no body
#
# and doubling every coordinate of the plum -- same topology, same material, nothing else touched
# -- makes PhysX run it, with its own "tetrahedron is degenerate or inverted" complaint falling
# from 1065 to 104. So the limit is absolute size, not shape and not stiffness. This is where the
# count is reported; four assets is an ordering, not a threshold, so the number is printed and the
# reader is told what it meant for these four.
TINY_TET = 1.0e-11      # m^3, a cube 0.22 mm on a side


def tiny_tetrahedra(stage):
    """{prim path: (how many tets are below TINY_TET, the smallest one, how many there are)}."""
    import usd_deformable

    out = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if usd_deformable.simulated_kind(prim) != "volume" or not prim.IsA(UsdGeom.TetMesh):
            continue
        tets = UsdGeom.TetMesh(prim).GetTetVertexIndicesAttr().Get()
        points = UsdGeom.PointBased(prim).GetPointsAttr().Get()
        if not tets or not points:
            continue
        p = np.asarray(points, dtype=np.float64)
        t = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
        a, b, c, d = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]], p[t[:, 3]]
        volume = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0)
        out[str(prim.GetPath())] = (int((volume < TINY_TET).sum()), float(volume.min()), len(t))
    return out


def textures(stage):
    """Which referenced files an asset needs and has not got.

    Ask USD, not the filesystem. A `.usdz` carries its textures inside itself and resolves them to
    `package.usdz[inside/the/package.png]`, which is not a path any directory holds -- taking the
    resolver's answer as a filename reported every texture of every packaged asset as missing.
    An empty `resolvedPath` is the resolver saying it could not find it, and that is the finding.
    """
    missing = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        for attr in prim.GetAttributes():
            value = attr.Get() if attr.GetTypeName().type.typeName == "SdfAssetPath" else None
            if value is None or not value.path:
                continue
            if not value.resolvedPath:
                missing.append(f"{prim.GetPath()}.{attr.GetName()} -> {value.path}")
    return missing


def report(path):
    stage = Usd.Stage.Open(str(path))
    default = stage.GetDefaultPrim()
    if not default:
        print(f"  no default prim: a USD this package can evaluate names one")
        return
    kinds = asset_kinds(stage, default.GetPath())
    print(f"  declares: {', '.join(sorted(kinds)) or 'neither a rigid body nor a deformable'}")
    if not kinds:
        print("  -> nothing here to simulate; the asset needs UsdPhysics.RigidBodyAPI or a "
              "deformable sim schema")
        return
    kind = "deformable" if "deformable" in kinds else "rigid"
    for prim_path, (what, points, count) in elements(stage).items():
        print(f"  simulated: {prim_path}  {points} points, {count} {what}")
    for prim_path, (tiny, smallest, total) in tiny_tetrahedra(stage).items():
        note = (f"  tetrahedra: {tiny} of {total} below {TINY_TET:g} m^3, smallest {smallest:.2e}")
        if tiny:
            note += (f"  -- PhysX built no body from the two assets measured here with the most of "
                     f"these (43 and 96); scaling one up, and nothing else, made it run")
        print(note)
    declared = asset_properties.read(str(path))
    for key in ("particle_radius", "thickness", "density", "youngs_modulus", "poissons_ratio",
                "stretch_stiffness", "bend_stiffness", "shear_stiffness",
                "friction", "restitution", "self_collision"):
        value, source = declared.get(key), declared.get(key + "_source")
        if value is not None:
            print(f"  says: {key} = {value}  ({source})")
    # A solid states a modulus and a Poisson ratio; a sheet states stretch and bend stiffnesses and
    # a thickness. Asking a sheet for a modulus would be asking the wrong question, so what is
    # wanted follows from what the geometry is.
    volume = any(what == "tetrahedra" for what, _, _ in elements(stage).values())
    wanted = (("density", "youngs_modulus", "poissons_ratio") if volume
              else ("density", "thickness", "stretch_stiffness"))
    silent = [k for k in wanted if declared.get(k) is None]
    if kind == "deformable" and silent:
        print(f"  !! does not say: {', '.join(silent)} -- an engine will use its own default and "
              f"the result will not be about this asset")
    lost = textures(stage)
    if lost:
        print(f"  !! {len(lost)} referenced file(s) missing, e.g. {lost[0]}")
    runnable = [f"{name} ({', '.join(sorted(s for s, e in solvers.items() if kind in SOLVER_SIMULATES.get(e.solver, set())))})"
                for name, solvers in _engines().items()
                if any(kind in SOLVER_SIMULATES.get(e.solver, set()) for e in solvers.values())]
    print(f"  runs on: {'; '.join(runnable)}")
    print(f"  experiments: {', '.join(sorted(experiments.for_kind(kind)))}")


def _engines():
    out = {}
    for env in envs.ENVIRONMENTS.values():
        name = "physx" if env.engine == "physx" else "newton" + env.newton.rstrip(".")
        if env.engine == "physx" and env.isaac != "6.1.0":
            name = f"physx{env.isaac}"
        out.setdefault(name, {})[env.solver] = env
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    files = []
    for given in args.paths:
        p = pathlib.Path(given)
        files += sorted(q for q in (p.rglob("*.usd*") if p.is_dir() else [p])
                        if q.is_file() and "_physx" not in q.stem)
    if not files:
        raise SystemExit("no USD files found")
    for path in files:
        print(f"\n{path}")
        try:
            report(path)
        except Exception as exc:   # noqa: BLE001 -- a bad asset is a finding, not a crash
            print(f"  !! could not be read: {type(exc).__name__}: {exc}")
    print(f"\n{len(files)} file(s)")


if __name__ == "__main__":
    main()
