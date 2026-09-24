"""Inside Kit: drop a PhysX deformable on a ground plane and write the animated USD Newton writes.

    ./isaac-run isaac610 asset_checks/native/physx_drop.py <asset.usda>
        [--seconds 2] [--fps 60] [--drop 0.05] [--usd out.usda]

The Newton runs are read out of the solver's own particle state and written as an animated mesh;
this does the same for PhysX, through `DeformablePrim`'s tensor view, so the two engines produce
the same artefact and `render_usd.py` photographs both from the same camera. Comparing them then
compares physics, not two different recording paths.

It also checks what it is comparing. PhysX does not complain about a deformable with no material
bound -- it runs its own 5e5 Pa default, a tenth of what this banana declares, and looks like a
softer engine rather than a broken stage. And the starting height is set through the view rather
than assumed, because creating that view costs frames the body spends falling.
"""
import argparse
import pathlib
import sys

import isaacsim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import drop_shape  # noqa: E402  (imports nothing; safe before SimulationApp)
import stepping  # noqa: E402  (a plain integer; safe before SimulationApp)
import physx_scene  # noqa: E402  (same: USD is imported inside `world`, not at its top)
from isaacsim import SimulationApp

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("asset")
ap.add_argument("--seconds", type=float, required=True,
                help="simulated seconds: the experiment's own SECONDS, which run.py passes")
ap.add_argument("--fps", type=float, default=stepping.FPS)
ap.add_argument("--substeps", type=int, default=stepping.SUBSTEPS,
                help="physics steps inside one recorded frame (stepping.SUBSTEPS)")
ap.add_argument("--drop", type=float, default=drop_shape.DROP_HEIGHT)
ap.add_argument("--usd", default=None)
ap.add_argument("--visual-asset", default=None,
                help="the original (Newton-flavour) asset to take the textured render mesh from; "
                     "the PhysX copy this runs has its source subtree deactivated")
args = ap.parse_args()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

EXPERIENCE = str(pathlib.Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.full.kit")
app = SimulationApp({"headless": True}, experience=EXPERIENCE)

# Everything below is imported only now, on purpose. Anything that pulls in `pxr` before the app
# exists initialises USD outside Kit, and Kit can then no longer register its own schema wrappers:
# the run dies during startup with "extension class wrapper ... has not been created yet".
import asset_properties  # noqa: E402
import usd_deformable  # noqa: E402  (which mesh the asset asks to be simulated)
import newton_drop  # noqa: E402  (imports no engine at module level: the contact rule lives there)
import recording  # noqa: E402  (the floor's size, and the recording)

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.core.experimental.prims import DeformablePrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

# The body schema, as PhysX spells the AOUSD name; the rule lives with the names.
BODY_API = usd_deformable.physx_name(usd_deformable.BODY)
# What counts as a pass, as fractions of the asset's own size and its own drop.
# "Settled" is judged on the 99th percentile of node speed, not the maximum. A maximum over a
# few thousand nodes is decided by whichever single node is jittering, so an asset that has not
# moved a tenth of a millimetre in a second still reads as moving; the percentile asks whether
# the body is at rest, which is the question.


def simulation_mesh(result):
    """The first of the three arrays DeformablePrim returns.

    `get_nodal_positions` and `get_element_indices` each hand back (simulation, collision, rest).
    The simulation mesh is the one the solver moves; the rest mesh never moves at all, so a video
    made from it would show a deformable that does not deform.
    """
    array = result[0] if isinstance(result, tuple) else result
    return np.asarray(array.numpy() if hasattr(array, "numpy") else array)


def world_nodes(prim):
    """The simulation mesh in world space: the body's pose applied to its nodes."""
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
root = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(root.GetPrim())
asset = stage.DefinePrim("/World/Asset")
asset.GetReferences().AddReference(args.asset)

physx_scene.world(stage, args.fps, args.substeps)
FIXTURES = ["/World/Ground"]
for _ in range(30):
    app.update()

# Which mesh this asset asks to be simulated -- one, or a refusal naming the several. Asked of
# the asset being evaluated (the original, with the AOUSD names and the material the parity check
# measured), not of the PhysX copy, which spells the same physics under other names.
_original = Usd.Stage.Open(args.visual_asset or args.asset)
_refusal = usd_deformable.why_not_runnable(_original, pathlib.Path(args.visual_asset or args.asset).name,
                                          most=physx_scene.BODIES)
if _refusal:
    raise SystemExit(_refusal)
_kind, target = usd_deformable.find(stage)[0]
body = None
for prim_ in Usd.PrimRange(asset, Usd.TraverseInstanceProxies()):
    if BODY_API in prim_.GetAppliedSchemas():
        body = prim_
        break
if body is None:
    raise SystemExit(f"[physx] {args.asset} declares no {BODY_API}; PhysX has nothing to simulate")
target = target or body
print(f"[physx] simulating {target.GetPath()} as body {body.GetPath()}")

# Raise the asset by moving its points, not its transform: a deformable's rest shape is authored
# on the prim and PhysX builds the body from that, so an ancestor Xform does not reach it. It is
# also exactly what the Newton runner does to its particles, so both engines start from the same
# geometry rather than from the same intention.
# The geometry is the prim carrying the *sim* schema and the body is the one carrying the *body*
# schema; they are the same prim only when the asset authors a TetMesh that is itself the body. A
# cooked surface hierarchy puts an Xform on top with the mesh underneath, and reading points off
# the Xform gets nothing at all.
geometry = target
points = np.asarray(UsdGeom.PointBased(geometry).GetPointsAttr().Get(), dtype=np.float64)
if points.ndim != 2:
    raise SystemExit(f"[physx] {geometry.GetPath()} carries no points to place")
height = float(points[:, 2].max() - points[:, 2].min())
lift = args.drop - float(points[:, 2].min())
points[:, 2] += lift
raised = [Gf.Vec3f(*p) for p in points]
UsdGeom.PointBased(geometry).GetPointsAttr().Set(raised)
rest_shape = geometry.GetAttribute("omniphysics:restShapePoints")
if not rest_shape or not rest_shape.HasAuthoredValue():
    raise SystemExit(f"[physx] {geometry.GetPath()} authors no omniphysics:restShapePoints; PhysX would "
                     f"build the body from a shape this script cannot place")
rest_shape.Set(raised)
# Size the collision offsets to the asset, the way the Newton runs size their particle radius.
# Left alone these default to PhysX's own scale-free numbers -- measured, the banana came to rest
# 17 mm above the floor, which is not the asset's physics, it is a 2 cm contact offset meant for
# a scene built in metres. Half the median distance between neighbouring nodes is the asset's own
# resolution, and it is the same quantity Newton is given.
# The same contact size Newton is given, and from the same place: the asset. PhysX's own
# defaults are scale-free -- left alone they rest this banana 17 mm above the floor -- and
# picking a different number here than Newton gets would mean the two engines were never asked
# to touch the ground in the same way.
declared, chosen = asset_properties.read(args.visual_asset or args.asset), {}
offset, offset_source = asset_properties.contact_size(declared)
if offset is None:
    offset = newton_drop.auto_radius(points)   # the same rule Newton's radius comes from
    chosen["rest_offset"] = (offset, "the asset declares neither a particle radius nor a shell "
                                     "thickness; half the median distance between nodes")
# The same band Newton is given, by the same rule and from the same function: wide enough for the
# asset's own size *and* for how far a node moves in one substep of this fall. It was twice the
# rest offset here and `contact_margin` there, and the asymmetry surfaced the first time an asset
# fell faster than the banana -- a 4035-tetrahedron apple sank 4.8 mm into a floor that Newton's
# VBD held it out of completely.
contact_offset = newton_drop.contact_margin(offset, args.substeps, args.fps, args.drop)
chosen["contact_offset"] = (contact_offset,
                            f"the band Newton is given too: {newton_drop.CONTACT_MARGIN_OF_RADIUS:g}x the rest "
                            f"offset, or one substep of this fall, whichever is wider")
import setups  # noqa: E402  (PhysX runs the default setup only; run.py refuses others)
asset_properties.report("physx", declared, chosen, setups.parse(setups.DEFAULT))
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
collision.CreateContactOffsetAttr(contact_offset)
for _ in range(10):
    app.update()
print(f"[physx] raised the asset {lift * 100:.1f} cm; it is {height * 1000:.1f} mm tall, "
      f"rest offset {offset * 1000:.2f} mm, contact offset {contact_offset * 1000:.2f} mm")

physx_scene.floor(stage, recording.ground_half(points))
_mu, _mu_why = asset_properties.friction(declared)
physx_scene.floor_material(stage, _mu, declared.get("restitution") or 0.0, FIXTURES)
SimulationManager.set_physics_sim_device("cuda")
timeline = omni.timeline.get_timeline_interface()
timeline.set_time_codes_per_second(args.fps)
timeline.play()
for _ in range(5):
    app.update()

prim = DeformablePrim(str(body.GetPath()))
raw = simulation_mesh(prim.get_element_indices())
# Three indices per element for a cloth, four for a soft body: the view says which.
if not hasattr(prim, "num_nodes_per_element"):
    raise SystemExit("[physx] this DeformablePrim reports no num_nodes_per_element; the element "
                     "shape is not guessed from the index count")
per_element = int(prim.num_nodes_per_element)
elements = raw.reshape(-1, per_element)
bound_materials = prim.get_applied_physics_materials()
print(f"[physx] physics material: {bound_materials}")
if not bound_materials or not all(bound_materials):
    raise SystemExit("[physx] the body has no physics material bound: PhysX would run its own "
                     "default (youngs_modulus 5e5) and look like a softer engine")

# Put the asset where the experiment says it starts. Creating the tensor view costs a handful of
# rendered frames and the body falls through them -- measured, a banana raised to 15 cm reported
# 7.4 cm by the time the view existed, which is 0.125 s of free fall exactly. Reading that as the
# starting height would let every engine's drop begin wherever its setup happened to end.
prim.set_nodal_positions(wp.array(points.reshape(1, -1, 3).astype(np.float32), dtype=wp.float32))
prim.set_nodal_velocities(wp.zeros((1, points.shape[0], 3), dtype=wp.float32))
start = world_nodes(prim)
print(f"[physx] placed at z [{start[:, 2].min():.4f}, {start[:, 2].max():.4f}] over {len(elements)} tets, "
      f"asked for {args.drop:.4f}")
if abs(start[:, 2].min() - args.drop) > offset:
    raise SystemExit(f"[physx] the solver would not take the starting pose: asked {args.drop:.4f}, "
                     f"it holds {start[:, 2].min():.4f}")

# The same recording every runner writes: the tetrahedral surface the solver moved, and the
# asset's textured render mesh carried along by it. The visual mesh comes from the original
# asset, not the PhysX copy this is simulating -- the conversion deactivates the source subtree
# the textures live on.
frames = int(args.seconds * args.fps)
tape = None
if args.usd:
    import recording
    tape = recording.Recording(args.usd, int(args.fps), frames, start, elements,
                               asset=args.visual_asset, sim_prim_path=str(body.GetPath()),
                               ground_half=recording.ground_half(points))

q, first_frame = start, None
for frame in range(frames):
    # One update per recorded frame: the scene's step rate is what puts `args.substeps` steps of
    # physics inside it. Calling update once per substep instead ran the frame for `substeps`
    # times its own length -- see `physx_scene` for the measurement that found it.
    app.update()
    q = world_nodes(prim)
    if not np.isfinite(q).all():
        print(f"[physx] diverged at {frame / args.fps:.2f}s")
        break
    if tape is not None:
        tape.frame(frame, q)
    if frame == 0:
        first_frame, said = drop_shape.check_free_fall(
            float(start[:, 2].min() - q[:, 2].min()), args.fps, float(start[:, 2].min()))
        print(f"[physx] {said}", flush=True)
    if frame % max(1, int(args.fps / 4)) == 0 or frame == frames - 1:
        v = simulation_mesh(prim.get_nodal_velocities()).reshape(-1, 3)
        print(f"[physx] t={frame / args.fps:5.2f}s  z [{q[:, 2].min():8.4f}, {q[:, 2].max():8.4f}]  "
              f"max|v| {np.abs(v).max():8.3f}", flush=True)

v = simulation_mesh(prim.get_nodal_velocities()).reshape(-1, 3)
fell = float(start[:, 2].min() - q[:, 2].min())
below = drop_shape.below_floor(float(q[:, 2].min()), offset)
speed = float(np.percentile(np.abs(v), 99))
peak = float(np.abs(v).max())
# The verdict and the line it is printed on belong to the experiment, which is why
# they are asked for rather than written out here: the same words were spelled out
# in both drop runners, and a pair of copies is a pair waiting to drift.
decision = drop_shape.verdict(bool(np.isfinite(q).all()), fell,
                              float(start[:, 2].min()), below, height,
                              speed, offset, contact_offset,
                              extent=float(q[:, 2].max() - q[:, 2].min()))
kept = drop_shape.height_kept(float(q[:, 2].max() - q[:, 2].min()), height, offset)
print(drop_shape.result_line("physx", fell, float(q[:, 2].min()), float(q[:, 2].max()),
                             below, speed, peak, decision, kept, first_frame))

if tape is not None:
    tape.close()
    print(f"[physx] wrote {args.usd}")
timeline.stop()
app.close()
