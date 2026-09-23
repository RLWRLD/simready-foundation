"""Check that the PhysX copy of an asset is still the same asset.

    python asset_checks/native/physx_parity.py <original.usda> <converted.usda>

PhysX cannot read the AOUSD deformable schemas, so its column of the results is produced from a
rewritten copy of the asset. That rewrite is ours (`to_physx.py`), which means every error in it
arrives disguised as an engine difference: a wrong modulus, a dropped tetrahedron or an invented
default would read as "PhysX behaves differently", and nothing downstream could tell.

So this reads the two files and compares them, and it deliberately shares no code with the thing
it is checking. It is the one duplication the rules allow -- measurement against statement --
because it re-derives from the USDs what `to_physx` claims to have carried, including a mass
computed from the geometry rather than taken from either file's word for it.

A complaint blocks the run. A note does not, but it is printed, because the conversion helpers
fill in defaults the asset never asked for and a number nobody chose should still be visible.
"""
import argparse
import sys

import numpy as np
from pxr import Usd, UsdGeom

import setups
import usd_deformable

# What the two schema families call the same physics. Volume deformables are described by a
# modulus and a Poisson ratio; surface ones by three stiffnesses and a thickness, which PhysX
# takes one for one under a `surface` prefix. Nothing here is per-asset.
SHARED = {"density": "density", "staticFriction": "staticFriction", "dynamicFriction": "dynamicFriction"}
VOLUME = {"youngsModulus": "youngsModulus", "poissonsRatio": "poissonsRatio"}
SURFACE = {"stretchStiffness": "surfaceStretchStiffness", "shearStiffness": "surfaceShearStiffness",
           "bendStiffness": "surfaceBendStiffness", "thickness": "surfaceThickness"}
# (AOUSD name, PhysX name) per kind, from the one place the names and the rule live.
SIM_SCHEMA = {k: (v, usd_deformable.physx_name(v)) for k, v in usd_deformable.SIM.items()}
MATERIAL_SCHEMA = {k: (v, usd_deformable.physx_name(v)) for k, v in usd_deformable.MATERIAL.items()}
RELATIVE_TOLERANCE = 1e-4


def schemas(prim):
    """The raw `apiSchemas` list. These schemas are not registered outside Kit, so
    `GetAppliedSchemas` returns nothing for exactly the prims this file is about."""
    listop = prim.GetMetadata("apiSchemas")
    return list(listop.GetAddedOrExplicitItems()) if listop else list(prim.GetAppliedSchemas())


def find(stage, wanted):
    for prim in stage.Traverse():
        if prim.IsActive() and wanted in schemas(prim):
            return prim
    return None


def bound_material(prim):
    """The physics material bound to `prim` or the nearest ancestor that binds one, as a path."""
    while prim and prim.IsValid():
        rel = prim.GetRelationship("material:binding:physics")
        if rel and rel.GetTargets():
            return rel.GetTargets()[0]
        prim = prim.GetParent()
    return None


def numbers(prim, prefix):
    out = {}
    if prim is None:
        return out
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith(prefix + ":") or not attr.HasAuthoredValue():
            continue
        value = attr.Get()
        if isinstance(value, float) or isinstance(value, int):
            out[name.split(":", 1)[1]] = float(value)
    return out


def triangles(mesh):
    """Fan-triangulate whatever polygons a mesh authors, so a quad grid and its triangulation can
    be compared as the same surface."""
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
    out, at = [], 0
    for count in counts:
        face = indices[at:at + count]
        out.extend([face[0], face[i], face[i + 1]] for i in range(1, count - 1))
        at += count
    return np.asarray(out, dtype=np.int64).reshape(-1, 3) if out else np.zeros((0, 3), dtype=np.int64)


def volume_of(points, tets):
    """Total volume, from the geometry alone."""
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return float(np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6.0)


def area_of(points, tris):
    a, b, c = (points[tris[:, i]] for i in range(3))
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)


def close(x, y):
    return abs(x - y) <= RELATIVE_TOLERANCE * max(abs(x), abs(y), 1e-12)


def compare(original, converted):
    """Returns (complaints, notes). A complaint means the two files are not the same asset."""
    complaints, notes = [], []
    first, second = Usd.Stage.Open(str(original)), Usd.Stage.Open(str(converted))

    kind = next((k for k in SIM_SCHEMA if find(first, SIM_SCHEMA[k][0])), None)
    if kind is None:
        # A bare TetMesh with no sim schema is what Newton's legacy path accepts, so allow it.
        kind = "volume" if any(p.IsA(UsdGeom.TetMesh) for p in first.Traverse()) else None
    if kind is None:
        return [f"{original} declares no deformable geometry to compare"], notes

    source = find(first, SIM_SCHEMA[kind][0]) or next(p for p in first.Traverse() if p.IsA(UsdGeom.TetMesh))
    target = find(second, SIM_SCHEMA[kind][1])
    if target is None:
        other = next((k for k in SIM_SCHEMA if find(second, SIM_SCHEMA[k][1])), None)
        return [f"{converted} has no {kind} deformable"
                + (f" -- it has a {other} one, so the conversion changed what the asset is" if other else "")], notes

    points_a = np.asarray(UsdGeom.PointBased(source).GetPointsAttr().Get(), dtype=np.float64)
    points_b = np.asarray(UsdGeom.PointBased(target).GetPointsAttr().Get(), dtype=np.float64)
    if len(points_a) != len(points_b):
        complaints.append(f"geometry: {len(points_a)} points became {len(points_b)}")
    else:
        moved = float(np.abs(points_a - points_b).max())
        span = float(np.ptp(points_a, axis=0).max())
        if moved > RELATIVE_TOLERANCE * span:
            complaints.append(f"geometry: a point moved {moved * 1000:.3f} mm on a "
                              f"{span * 1000:.1f} mm asset")

    if kind == "volume":
        tets_a = np.asarray(UsdGeom.TetMesh(source).GetTetVertexIndicesAttr().Get() or [], dtype=np.int64)
        tets_b = np.asarray(UsdGeom.TetMesh(target).GetTetVertexIndicesAttr().Get() or [], dtype=np.int64)
        if tets_a.shape != tets_b.shape or not np.array_equal(tets_a, tets_b):
            complaints.append(f"elements: {len(tets_a)} tetrahedra became {len(tets_b)}"
                              + ("" if tets_a.shape != tets_b.shape else " with different vertices"))
        elif len(points_a) == len(points_b):
            va, vb = volume_of(points_a, tets_a), volume_of(points_b, tets_b)
            if not close(va, vb):
                complaints.append(f"volume: {va * 1e6:.3f} cm3 became {vb * 1e6:.3f} cm3")
    else:
        tris_a, tris_b = triangles(UsdGeom.Mesh(source)), triangles(UsdGeom.Mesh(target))
        if len(tris_a) != len(tris_b):
            complaints.append(f"elements: {len(tris_a)} triangles became {len(tris_b)}")
        elif len(points_a) == len(points_b):
            aa, ab = area_of(points_a, tris_a), area_of(points_b, tris_b)
            if not close(aa, ab):
                complaints.append(f"area: {aa * 1e4:.3f} cm2 became {ab * 1e4:.3f} cm2")

    wanted = dict(SHARED, **(VOLUME if kind == "volume" else SURFACE))
    said = numbers(find(first, MATERIAL_SCHEMA[kind][0]), "physics")
    if kind == "surface":
        # The copy carries the default setup's reading of the surface stiffnesses, not the raw
        # authored numbers; the audit applies the same arithmetic, from the same owner.
        said = dict(said, **usd_deformable.surface_stiffnesses(said, setups.parse(setups.DEFAULT)["surface"]))
    carried = find(second, MATERIAL_SCHEMA[kind][1]) or find(second, MATERIAL_SCHEMA["volume"][1])
    got = numbers(carried, "omniphysics")
    # The numbers being right is half of it: the body has to be *bound* to the prim that carries
    # them, or PhysX runs its own default and the copy only looks like the asset.
    bound = bound_material(target)
    if bound is None:
        complaints.append("material: the copy binds no physics material to its body")
    elif carried is not None and bound != carried.GetPath():
        complaints.append(f"material: the body is bound to {bound}, not to {carried.GetPath()} "
                          f"where the carried numbers are")
    for ours, theirs in wanted.items():
        if ours not in said:
            if theirs in got:
                notes.append(f"material: the asset never declares {ours}, and the copy carries "
                             f"{theirs}={got[theirs]:g} -- a value nobody chose")
            continue
        if theirs not in got:
            complaints.append(f"material: the asset declares {ours}={said[ours]:g} and the copy "
                              f"carries no {theirs}")
        elif not close(said[ours], got[theirs]):
            complaints.append(f"material: {ours}={said[ours]:g} became {theirs}={got[theirs]:g}")
    for theirs, value in sorted(got.items()):
        if theirs not in wanted.values():
            notes.append(f"material: the copy carries {theirs}={value:g}, which has no counterpart "
                         f"in what the asset declares")

    density = said.get("density")
    if density is not None and not complaints:
        if kind == "volume":
            mass = density * volume_of(points_a, np.asarray(
                UsdGeom.TetMesh(source).GetTetVertexIndicesAttr().Get(), dtype=np.int64))
        else:
            mass = density * area_of(points_a, triangles(UsdGeom.Mesh(source))) * said.get("thickness", 0.0)
        notes.append(f"mass implied by the geometry and the declared density: {mass * 1000:.2f} g")
    return complaints, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("original")
    ap.add_argument("converted")
    args = ap.parse_args()
    complaints, notes = compare(args.original, args.converted)
    for note in notes:
        print(f"[parity] note: {note}")
    for complaint in complaints:
        print(f"[parity] MISMATCH: {complaint}")
    print(f"[parity] {'the PhysX copy is the same asset' if not complaints else 'the PhysX copy is NOT the same asset'}")
    return 1 if complaints else 0


if __name__ == "__main__":
    sys.exit(main())
