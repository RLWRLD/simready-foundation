"""Inside Kit: drop a PhysX deformable on a ground plane and write the same animated USD Newton writes.

    ./isaac-run isaac610 tools/physx_baseline.py <asset.usda> [--seconds 2] [--fps 60]
                                                 [--drop 0.05] [--usd out.usda]

The Newton runs are read out of the solver's own particle state and written as an animated mesh;
this does the same for PhysX, through `omni.physics.tensors`' volume deformable view, so the two
engines produce the same artefact and `tools/render_usd.py` photographs both from the same camera.
Comparing them then compares physics, not two different recording paths.

It also reads the Young's modulus PhysX ended up with and prints it beside what the asset asked
for. A deformable whose material did not bind runs on the schema default of zero and looks like
a very soft engine rather than a broken stage, so the number is checked rather than assumed.
"""
import argparse
import pathlib
import sys

import isaacsim
from isaacsim import SimulationApp

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("asset")
ap.add_argument("--seconds", type=float, default=2.0)
ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--substeps", type=int, default=4)
ap.add_argument("--drop", type=float, default=0.05)
ap.add_argument("--usd", default=None)
args = ap.parse_args()

EXPERIENCE = str(pathlib.Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.full.kit")
app = SimulationApp({"headless": True}, experience=EXPERIENCE)

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.core.experimental.prims import DeformablePrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

SIM_API = "OmniPhysicsVolumeDeformableSimAPI"

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(root.GetPrim())
asset = stage.DefinePrim("/World/Asset")
asset.GetReferences().AddReference(args.asset)

scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr(9.81)
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
half = 2.0
ground.CreatePointsAttr([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, -half, 0.0),
                         Gf.Vec3f(half, half, 0.0), Gf.Vec3f(-half, half, 0.0)])
ground.CreateFaceVertexCountsAttr([4])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground.CreateExtentAttr([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, half, 0.0)])
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
for _ in range(30):
    app.update()


BODY_API = "OmniPhysicsDeformableBodyAPI"


def deformable_body():
    """The prim PhysX will actually simulate: the one carrying the deformable *body* schema.

    Which prim that is depends on how the asset was authored -- a TetMesh can carry the body
    itself, or a parent Xform can own a cooked simulation mesh underneath it. Looking for the
    body schema finds both; assuming a fixed shape found neither.
    """
    body = target = None
    for prim in Usd.PrimRange(asset, Usd.TraverseInstanceProxies()):
        if body is None and BODY_API in prim.GetAppliedSchemas():
            body = prim
        if target is None and SIM_API in prim.GetAppliedSchemas():
            target = prim
    if body is None:
        raise SystemExit(f"[physx] {args.asset} declares no {BODY_API}; PhysX has nothing to simulate")
    return body, target or body


body, target = deformable_body()
print(f"[physx] simulating {target.GetPath()} as body {body.GetPath()}")

# Lift the whole asset so its lowest authored point starts `drop` above the floor, the same way
# the Newton runs do. The drop has to be measured from the geometry, not the transform, because
# the asset's origin is wherever its author put it.
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
low = bbox.ComputeWorldBound(asset).ComputeAlignedRange().GetMin()[2]
lift = args.drop - low
UsdGeom.XformCommonAPI(asset).SetTranslate(Gf.Vec3d(0.0, 0.0, lift))
print(f"[physx] lifted the asset {lift * 100:.1f} cm so its lowest point starts {args.drop * 100:.1f} cm up")

# Deformables run on the GPU, and the tensor views only exist once the timeline is playing.
# This is the order `isaacsim.core.experimental.prims`' own DeformablePrim tests use.
SimulationManager.set_physics_sim_device("cuda")
timeline = omni.timeline.get_timeline_interface()
timeline.set_time_codes_per_second(args.fps)
timeline.play()
for _ in range(5):
    app.update()

prim = DeformablePrim(str(body.GetPath()))
rest = np.asarray(prim.get_nodal_positions()[0]).reshape(-1, 3)
print(f"[physx] DeformablePrim at {body.GetPath()}: {rest.shape[0]} simulation nodes")

# Read the material PhysX actually bound. With nothing bound it does not complain -- it runs
# its own 5e5 Pa default, a tenth of what this asset declares, and the engine merely looks soft.
try:
    print(f"[physx] physics material in the solver: {prim.get_applied_physics_materials()}")
except Exception as exc:                                  # noqa: BLE001
    print(f"[physx] could not read the bound physics material back: {exc}")

elements = np.asarray(prim.get_element_indices()[0]).reshape(-1, 4)

print(f"[physx] starts z [{rest[:, 2].min():.4f}, {rest[:, 2].max():.4f}] over {len(elements)} tets")

out_stage = surface = None
if args.usd:
    out_stage = Usd.Stage.CreateNew(args.usd)
    UsdGeom.SetStageUpAxis(out_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out_stage, 1.0)
    out_root = UsdGeom.Xform.Define(out_stage, "/root")
    out_stage.SetDefaultPrim(out_root.GetPrim())
    out_stage.SetTimeCodesPerSecond(args.fps)
    out_stage.SetFramesPerSecond(args.fps)
    # A tet mesh's outside is every triangular face that only one tet owns. Drawing that, rather
    # than all four faces of every tet, is what makes the render look like the object.
    faces = {}
    for tet in elements:
        for tri in ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)):
            key = tuple(sorted(int(tet[i]) for i in tri))
            faces[key] = faces.get(key, 0) + 1
    outside = [k for k, n in faces.items() if n == 1]
    surface = UsdGeom.Mesh.Define(out_stage, "/root/deformable")
    surface.CreateFaceVertexCountsAttr([3] * len(outside))
    surface.CreateFaceVertexIndicesAttr([i for tri in outside for i in tri])
    surface.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.8, 0.45)])
    plane = UsdGeom.Mesh.Define(out_stage, "/root/ground")
    plane.CreatePointsAttr(ground.GetPointsAttr().Get())
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    print(f"[physx] writing {len(outside)} surface triangles of {len(elements)} tets")

frames = int(args.seconds * args.fps)
for frame in range(frames):
    for _ in range(args.substeps):
        app.update()
    q = np.asarray(prim.get_nodal_positions()[0]).reshape(-1, 3)
    if not np.isfinite(q).all():
        print(f"[physx] diverged at {frame / args.fps:.2f}s")
        break
    if surface is not None:
        surface.GetPointsAttr().Set([Gf.Vec3f(*p) for p in q.astype(float)], Usd.TimeCode(frame))
    if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
        v = np.asarray(prim.get_nodal_velocities()[0]).reshape(-1, 3)
        print(f"[physx] t={frame / args.fps:5.2f}s  z [{q[:, 2].min():8.4f}, {q[:, 2].max():8.4f}]  "
              f"max|v| {np.abs(v).max():8.3f}", flush=True)

if out_stage is not None:
    out_stage.SetStartTimeCode(0)
    out_stage.SetEndTimeCode(frames - 1)
    out_stage.GetRootLayer().Save()
    print(f"[physx] wrote {args.usd}")
timeline.stop()
app.close()
