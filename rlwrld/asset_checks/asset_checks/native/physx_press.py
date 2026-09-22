"""Inside Kit: press a PhysX deformable with the same plate, on the same schedule, as Newton.

    ./isaac-run isaac610 asset_checks/native/physx_press.py <asset.usda>
        [--seconds 4] [--fps 60] [--usd out.usda]

The schedule, the plate's size and depth, and what counts as a pass all come from
`press_shape.py`, which is the experiment and knows about no engine. What is here is only how
PhysX is asked: the plate is a kinematic rigid body whose transform is written each frame, and
the asset is read through DeformablePrim's tensor view, exactly as the drop is.
"""
import argparse
import pathlib
import sys

import isaacsim
from isaacsim import SimulationApp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import physx_scene  # noqa: E402  (imports no USD at its top: safe before SimulationApp)

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("asset")
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--substeps", type=int, default=physx_scene.SUBSTEPS,
                help="physics steps inside one recorded frame (physx_scene.SUBSTEPS)")
ap.add_argument("--margin", type=float, default=0.0, help="0 = the engine's own contact offset")
ap.add_argument("--usd", default=None)
ap.add_argument("--visual-asset", default=None,
                help="the original (Newton-flavour) asset to take the textured render mesh from; "
                     "the PhysX copy this runs has its source subtree deactivated")
args = ap.parse_args()

EXPERIENCE = str(pathlib.Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.full.kit")
app = SimulationApp({"headless": True}, experience=EXPERIENCE)

# Everything below is imported only now, on purpose. Anything that pulls in `pxr` before the app
# exists initialises USD outside Kit, and Kit can then no longer register its own schema wrappers:
# the run dies during startup with "extension class wrapper ... has not been created yet".
import asset_properties  # noqa: E402
import newton_drop  # noqa: E402  (imports no engine at module level: the contact rule lives there)
import press_shape  # noqa: E402

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import DeformablePrim, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

# PhysX spells a deformable's simulated geometry one of two ways depending on what it is made
# of. Asking for both is what lets a cloth and a soft body go through the same runner.
SIM_APIS = ("OmniPhysicsVolumeDeformableSimAPI", "OmniPhysicsSurfaceDeformableSimAPI")
BODY_API = "OmniPhysicsDeformableBodyAPI"
PLATE = "/World/PressPlate"


def simulation_mesh(result):
    array = result[0] if isinstance(result, tuple) else result
    return np.asarray(array.numpy() if hasattr(array, "numpy") else array)


def world_nodes(prim):
    local = simulation_mesh(prim.get_nodal_positions()).reshape(-1, 3)
    positions, orientations = prim.get_world_poses()
    p = np.asarray(positions.numpy() if hasattr(positions, "numpy") else positions).reshape(-1, 3)[0]
    q = np.asarray(orientations.numpy() if hasattr(orientations, "numpy") else orientations).reshape(-1, 4)[0]
    w, x, y, z = q[0], q[1], q[2], q[3]
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return local @ rotation.T + p


omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
asset = stage.DefinePrim("/World/Asset")
asset.GetReferences().AddReference(args.asset)

physx_scene.world(stage, args.fps, args.substeps)
for _ in range(30):
    app.update()

body = target = None
for prim_ in Usd.PrimRange(asset, Usd.TraverseInstanceProxies()):
    if body is None and BODY_API in prim_.GetAppliedSchemas():
        body = prim_
    if target is None and any(api in prim_.GetAppliedSchemas() for api in SIM_APIS):
        target = prim_
if body is None:
    raise SystemExit(f"[physx] {args.asset} declares no {BODY_API}; PhysX has nothing to simulate")

# The geometry is the prim carrying the *sim* schema and the body is the one carrying the *body*
# schema, and they are only the same prim when the asset authors a TetMesh that is itself the
# body. A cooked surface hierarchy puts an Xform on top with the mesh underneath, and reading
# points off the Xform gets nothing at all.
geometry = target
points = np.asarray(UsdGeom.PointBased(geometry).GetPointsAttr().Get(), dtype=np.float64)
if points.ndim != 2:
    raise SystemExit(f"[physx] {geometry.GetPath()} carries no points to place")
height = float(points[:, 2].max() - points[:, 2].min())
footprint = (float(points[:, 0].max() - points[:, 0].min()), float(points[:, 1].max() - points[:, 1].min()))
centre = (float(points[:, 0].mean()), float(points[:, 1].mean()))

# Same collision sizing as the drop: PhysX's own defaults are scale-free and rest this asset
# 17 mm above the floor. Half the median distance between neighbouring nodes is the asset's own
# resolution, and it is the same quantity Newton's particle radius is set from.
# The same contact size Newton is given, and from the same place: the asset. PhysX's own
# defaults are scale-free -- left alone they rest this banana 17 mm above the floor -- and
# picking a different number here than Newton gets would mean the two engines were never asked
# to touch the ground in the same way.
declared, chosen = asset_properties.read(args.visual_asset or args.asset), {}
offset, offset_source = asset_properties.contact_size(declared)
if offset is None:
    picked = np.random.default_rng(0).choice(len(points), size=min(512, len(points)), replace=False)
    spacing = np.sqrt(((points[picked][:, None, :] - points[None, :, :]) ** 2).sum(-1))
    spacing[spacing < 1e-9] = np.inf
    offset = float(np.median(spacing.min(axis=1)) * 0.5)
    chosen["rest_offset"] = (offset, "the asset declares neither a particle radius nor a shell "
                                     "thickness; half the median distance between nodes")
chosen["contact_offset"] = (offset * 2.0, "twice the rest offset, as Newton's contact margin is")
# PhysX derives its contact response from the bound material, so unlike Newton there is no
# separate penalty stiffness to raise: the asset's own Young's modulus is already what the plate
# pushes against. Recorded here so the two engines' logs can be read against each other.
if declared.get("youngs_modulus") is not None:
    chosen["contact_stiffness_source"] = (declared["youngs_modulus"],
                                          "PhysX contacts run off the bound material's modulus, "
                                          "not a separate penalty constant")
asset_properties.report("physx", declared, chosen)
# Of the five environments only Newton's VBD has a self-collision switch. The OmniPhysics
# deformable schemas this conversion applies declare none -- their only self-collision attribute is
# a `selfCollisionFilterPose` purpose -- so PhysX runs whatever its own default is and there is
# nothing to set from the USD. Said out loud because the alternative is a reader assuming all five
# ran the same model.
_declared, _why = asset_properties.self_collision(declared)
print(f"[physx] self-collision: the asset says {'on' if _declared else 'nothing (' + _why + ')'}, "
      f"and this schema family has no switch to honour it with")
collision = PhysxSchema.PhysxCollisionAPI.Apply(geometry)
collision.CreateRestOffsetAttr(offset)
# Rest on the floor rather than fall onto it: this test is about the plate.
lift = offset - float(points[:, 2].min())
points[:, 2] += lift
raised = [Gf.Vec3f(*p) for p in points]
UsdGeom.PointBased(geometry).GetPointsAttr().Set(raised)
rest_shape = geometry.GetAttribute("omniphysics:restShapePoints")
if not rest_shape or not rest_shape.HasAuthoredValue():
    raise SystemExit(f"[physx] {geometry.GetPath()} authors no omniphysics:restShapePoints")
rest_shape.Set(raised)

# The experiment says how deep the plate goes; PhysX's contact band has to cover that, or the
# nodes past it feel nothing and the plate sweeps through. Same number Newton is given.
indent = press_shape.press_depth(height)
# The band, by the rule both engines share -- the asset's own size, the indentation it has to
# cover, and how far a node moves in one substep -- and then capped, because a band wider than
# this presses the asset with both faces of the plate at once. The cap is the press experiment's;
# the rest is `contact_margin`, the same function Newton is given.
wanted = newton_drop.contact_margin(offset, args.substeps, args.fps, 0.0, depth=indent)
margin = args.margin or min(wanted, press_shape.widest_usable_margin(height))
collision.CreateContactOffsetAttr(margin)
# Reported here rather than added to `chosen`: the band is not known until the asset's height is,
# and `asset_properties.report` has already run by then. A value that reaches the solver but not
# the log is the thing this whole file exists to avoid.
print(f"[physx] ours:  contact_offset = {margin:g}  -- the band Newton is given too, capped at "
      f"{press_shape.widest_usable_margin(height) * 1000:.2f} mm so the plate does not press with both faces")
thickness = press_shape.plate_thickness(height)
start_z = float(points[:, 2].max()) + thickness / 2.0 + offset * 2.0
plate = UsdGeom.Cube.Define(stage, PLATE)
plate.CreateSizeAttr(2.0)
UsdGeom.XformCommonAPI(plate).SetScale(Gf.Vec3f(footprint[0] * press_shape.PLATE_FOOTPRINT,
                                                footprint[1] * press_shape.PLATE_FOOTPRINT,
                                                thickness / 2.0))
UsdGeom.XformCommonAPI(plate).SetTranslate(Gf.Vec3d(centre[0], centre[1], start_z))
plate.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.45, 0.85)])
UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
rigid = UsdPhysics.RigidBodyAPI.Apply(plate.GetPrim())
rigid.CreateKinematicEnabledAttr(True)
for _ in range(10):
    app.update()
print(f"[physx] {height * 1000:.1f} mm tall, rest offset {offset * 1000:.2f} mm, "
      f"plate {thickness * 1000:.1f} mm thick parked at {start_z:.4f}")

SimulationManager.set_physics_sim_device("cuda")
timeline = omni.timeline.get_timeline_interface()
timeline.set_time_codes_per_second(args.fps)
timeline.play()
for _ in range(5):
    app.update()

prim = DeformablePrim(str(body.GetPath()))
raw = simulation_mesh(prim.get_element_indices())
# Three indices per element for a cloth, four for a soft body: the view says which.
per_element = int(prim.num_nodes_per_element) if hasattr(prim, "num_nodes_per_element") else (
    4 if raw.size % 4 == 0 and raw.size % 3 else 3)
elements = raw.reshape(-1, per_element)
import warp as wp  # noqa: E402
prim.set_nodal_positions(wp.array(points.reshape(1, -1, 3).astype(np.float32), dtype=wp.float32))
prim.set_nodal_velocities(wp.zeros((1, points.shape[0], 3), dtype=wp.float32))
# The plate is driven through the physics view, not by writing its USD transform. A kinematic
# body takes its target from the solver's own state; the transform on the prim is where it was
# authored, and editing that mid-run moves the picture and nothing else -- measured, the plate
# descended 12 mm past the banana's top without touching it.
plate_prim = RigidPrim(PLATE)

# The same recording every runner writes. The visual mesh comes from the original asset, not the
# PhysX copy being simulated -- the conversion deactivates the source subtree the textures are on.
frames = int(args.seconds * args.fps)
tape = None
if args.usd:
    import recording
    tape = recording.Recording(args.usd, int(args.fps), frames, points, elements,
                               asset=args.visual_asset, sim_prim_path=str(body.GetPath()),
                               plate_centre=centre,
                               plate=(footprint[0] * press_shape.PLATE_FOOTPRINT,
                                      footprint[1] * press_shape.PLATE_FOOTPRINT, thickness / 2.0))

settle_at, recover_at = press_shape.settle_frame(frames), press_shape.recovery_frame(frames)
start_top = lowest_top = bottom_z = recovered = settled_height = depth = None
deepest = 0.0
for frame in range(frames):
    plate_z = press_shape.plate_height(frame, frames, start_z, bottom_z)
    plate_prim.set_world_poses(positions=np.array([[centre[0], centre[1], plate_z]], dtype=np.float32))
    # One update per recorded frame, holding `args.substeps` steps of physics: the plate's law is
    # written in frames, so a frame that ran for `substeps` times its own length also drove the
    # plate at a `substeps`-th of the speed the experiment asks for. See `physx_scene`.
    app.update()
    q = world_nodes(prim)
    if not np.isfinite(q).all():
        print(f"[physx] diverged at {frame / args.fps:.2f}s")
        break
    if tape is not None:
        tape.frame(frame, q, plate_z=plate_z)
    top_now = float(q[:, 2].max())
    if frame == settle_at:
        start_top = lowest_top = top_now
        floor_now = float(q[:, 2].min())
        settled_height = top_now - floor_now
        centre = press_shape.plate_over(q)      # press where the asset now lies, not where it was
        if tape is not None:
            tape.plate_centre = (float(centre[0]), float(centre[1]))
        depth = press_shape.press_depth(settled_height)
        bottom_z = top_now - depth + thickness / 2.0
        print(f"[physx] settled to {top_now:.4f} ({settled_height * 1000:.1f} mm tall); the plate "
              f"will indent it {depth * 1000:.1f} mm")
    if start_top is None:
        continue
    lowest_top = min(lowest_top, top_now)
    deepest = max(deepest, -float(q[:, 2].min()))
    if frame >= recover_at:
        recovered = top_now
    if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
        print(f"[physx] t={frame / args.fps:5.2f}s  plate {plate_z:7.4f}  top {top_now:7.4f}  "
              f"floor {float(q[:, 2].min()):7.4f}", flush=True)

compressed = (start_top - lowest_top) if start_top is not None else 0.0
recovery = press_shape.recovery_fraction(recovered, lowest_top, compressed, offset)
# PhysX exposes no soft-contact count, so this is not one: it records only whether the plate's
# underside ever got below the asset's top. Reported as a plain yes/no rather than as a count,
# which a "1" in a column of thousands would be read as.
touched = 1 if (bottom_z is not None and bottom_z - thickness / 2.0 < start_top) else 0
print(press_shape.result_line("physx", start_top or 0.0, lowest_top or 0.0, compressed, height,
                              recovery, max(0.0, deepest - offset), "yes" if touched else "no",
                              press_shape.verdict(touched, compressed, height,
                                                  max(0.0, deepest - offset), recovery,
                                                  settled_height=settled_height),
                              indent=depth))
if tape is not None:
    tape.close()
    print(f"[physx] wrote {args.usd}")
timeline.stop()
app.close()
