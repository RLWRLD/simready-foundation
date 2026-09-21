"""Inside Kit: turn an animated USD into frames, with one fixed camera that frames the whole run.

    ./isaac-run isaac610 native/render_usd.py <recording.usda> <out_dir> [--fps 60] [--size 640] [--show visual|collision|both]

Newton writes what it simulated with `newton.viewer.ViewerUSD`: the deformable's surface as an
animated mesh and its particles as an animated point instancer. That file is the honest record of
the physics -- but it is geometry, not pictures. This opens it in Kit and photographs it.

Why this exists at all: Isaac's own Fabric sync writes rigid body transforms and nothing else, so
a deformable simulated inside Isaac never reaches the renderer and every video of one came out
still. Rendering the USD afterwards sidesteps that entirely -- the frames can only show what the
solver actually produced, because that is all the file contains.

The camera is placed once, from the bounds of the whole animation, so every frame of every engine
is shot from the same place and the videos can be put side by side.
"""
import argparse
import math
import pathlib
import sys

import isaacsim
from isaacsim import SimulationApp

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("stage")
ap.add_argument("out_dir")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--show", default="both", choices=("both", "collision", "visual"),
                help="which of the two meshes a recording holds to photograph: the tetrahedral "
                     "surface the solver moved, the asset's textured render mesh carried along "
                     "by it, or whatever the file has")
args = ap.parse_args()

EXPERIENCE = str(pathlib.Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.full.kit")
app = SimulationApp({"headless": True, "width": args.size, "height": args.size}, experience=EXPERIENCE)

import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.kit.viewport.utility  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/app/asyncRendering", False)          # so a grab cannot race the render thread
settings.set("/app/asyncRenderingLowLatency", False)
# What the rigid captures turn off, from the code that turns them off
# (`simready_benchmark_engine_kit/kit_engine_proxy.py`): without these the stage's red and green
# axis lines are drawn across the floor and no rigid video has them.
settings.set("/app/viewport/grid/enabled", False)      # omni.kit.viewport.legacy_gizmos reads this
settings.set("/persistent/app/viewport/displayOptions", 0)   # bitmask; 0 is nothing visible
settings.set("/rtx/wireframe/enabled", False)
for key in ("visualizationCollisionMesh", "visualizationDisplayJoints", "visualizationSimulationOutput"):
    settings.set(f"/persistent/physics/{key}", False)
settings.set("/rtx/pathtracing/spp", 1)

# A recording holds both meshes so the two videos come from one run and line up frame for
# frame. Hiding one is how a camera is pointed at the other; the framing is computed afterwards,
# so it follows whichever is left visible.
HIDDEN = {"collision": "/root/visual", "visual": "/root/collision"}   # fixtures show in both

GROUND_SPAN = 50.0   # a prim wider than this is scenery, not the subject
ASSET_GROUND = "/root/ground"     # the recording's own floor; the room replaces it
FIXTURES = "/root/fixtures"       # what the experiment placed: a plate, a slope, a gripper's pads

out = pathlib.Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
omni.usd.get_context().open_stage(args.stage)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

if args.show != "both":
    hide = stage.GetPrimAtPath(HIDDEN[args.show])
    keep = stage.GetPrimAtPath(HIDDEN["collision" if args.show == "visual" else "visual"])
    if not keep or not keep.IsValid():
        print(f"[render] FAIL: this recording has no {args.show} mesh to photograph")
        app.close()
        sys.exit(4)
    if hide and hide.IsValid():
        # The root and every gprim beneath it, so the hiding does not rest on how a scene
        # delegate treats an ancestor's opinion; the count is printed so a run proves it.
        hidden = 0
        for prim in Usd.PrimRange(hide):
            if prim == hide or prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
        print(f"[render] showing the {args.show} mesh; {HIDDEN[args.show]} hidden ({hidden} prims)")

# The room is the floor now, and two coincident floors z-fight.
asset_ground = stage.GetPrimAtPath(ASSET_GROUND)
if asset_ground and asset_ground.IsValid():
    UsdGeom.Imageable(asset_ground).MakeInvisible()

start, end = stage.GetStartTimeCode(), stage.GetEndTimeCode()
stage_fps = stage.GetTimeCodesPerSecond() or 60.0
print(f"[render] {args.stage}: time {start}..{end} at {stage_fps} tcps")




def is_scenery(prim, times):
    """A prim wider than GROUND_SPAN is the world around the subject, not the subject: Newton writes
    its ground as a 1000 m quad, and the slope drop's slope is a 100 m plane spanning the room.
    NVIDIA does not frame that either -- its fixed camera frames the asset and the gripper, and
    hands the slope test back to a follow camera."""
    box = UsdGeom.BBoxCache(Usd.TimeCode(times[0]), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = box.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        return False
    size = r.GetSize()
    return max(size[0], size[1], size[2]) > GROUND_SPAN


def moving_prims(times):
    """The prims whose bounds change over the run -- that is, the simulation.

    Framing on everything puts the camera a thousand metres back, because Newton writes its
    ground plane as a 1000 m quad and its point instancer keeps a unit-sphere prototype at the
    origin. Neither is the subject, and neither moves. Asking "what moves?" needs no list of
    names to keep up to date: it is true of a banana, a cloth, a gripper's fingers and anything
    added later, and false of scenery, prototypes and the static copy of the asset that the USD
    import leaves at the origin.
    """
    caches = [UsdGeom.BBoxCache(Usd.TimeCode(t), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
              for t in times]
    moving, still = [], []   # `still`: visible, not scenery, and not moving -- what a fallback may frame
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer)):
            continue
        ranges = [c.ComputeWorldBound(prim).ComputeAlignedRange() for c in caches]
        if any(r.IsEmpty() for r in ranges):
            continue
        if is_scenery(prim, times):
            continue               # never the subject, moving or not
        if not UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited:
            continue               # a hidden mesh must not pull the camera towards itself
        travel = max(Gf.Vec3d(a.GetMidpoint() - b.GetMidpoint()).GetLength()
                     for a in ranges for b in ranges)
        stretch = max(abs(max(a.GetSize()) - max(b.GetSize())) for a in ranges for b in ranges)
        (moving if max(travel, stretch) > 1e-5 else still).append(prim)
    print(f"[render] framing on {[str(p.GetPath()) for p in moving]}")
    if not moving:
        print("[render] nothing in this stage moves; framing on everything that is not scenery")
    return moving or still


def world_bounds(prims, times):
    """The box that holds the whole animation, so the camera is placed once and left alone."""
    cache_purpose = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    total = Gf.Range3d()
    for t in times:
        bbox = UsdGeom.BBoxCache(Usd.TimeCode(t), cache_purpose, useExtentsHint=False)
        for prim in prims:
            r = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            if not r.IsEmpty():
                total.UnionWith(r)
    return total


samples = [start + (end - start) * f for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
moving = moving_prims(samples)
# What the experiment involves, which is what NVIDIA's `_place_fixed_camera` frames: the asset and
# the gripper, down to the floor. Here that is what moves plus the fixtures the experiment placed
# -- a press plate, a slope, a gripper's pads -- and the floor is z = 0, where the recording puts
# it. Framing only what moves put the camera so close to a sliding orange that the slope it slid
# down filled the frame as a wall.
fixtures = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(FIXTURES))
            if prim.IsA(UsdGeom.Gprim) and not is_scenery(prim, samples)]
subject = moving + fixtures
bounds = world_bounds(subject, samples)
if not bounds.IsEmpty():
    bounds.UnionWith(Gf.Vec3d(bounds.GetMin()[0], bounds.GetMin()[1], min(0.0, bounds.GetMin()[2])))
print(f"[render] framing on {len(moving)} moving prim(s) and {len(fixtures)} fixture(s), down to the floor")
if bounds.IsEmpty():
    print("[render] FAIL: nothing with bounds in this stage")
    app.close()
    sys.exit(2)
centre = bounds.GetMidpoint()
size = bounds.GetSize()
span = max(size[0], size[1], size[2])
print(f"[render] bounds centre {tuple(round(c, 4) for c in centre)} size {tuple(round(s, 4) for s in size)}")

# The framing, the room and the floor of the rigid runs, from the code that made them rather than
# from a reading of it. `simready_benchmark_engine_kit` owns the camera maths and the room;
# NVIDIA's own ground_drop test states the colours; `asset_checks.kit.scene.add_visual_cues` draws
# the checkerboard that makes the floor readable. A copy of any of this is a copy that drifts --
# an earlier hand-rolled version of these forty lines put the camera high over a plain grey room.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from asset_checks.kit import scene as rigid_scene                   # noqa: E402
from simready_benchmark_engine_kit import camera_follow             # noqa: E402
from simready_benchmark_engine_kit.scene_handle import KitSceneHandle  # noqa: E402

# `simready_benchmark_kit_suite/fet003_physics/ground_drop.py`, the rigid drop these sit beside.
WALL_COLOUR = (0.3, 0.4, 0.7)     # "saturated blue walls"
GROUND_COLOUR = (0.25, 0.35, 0.6)  # "darker blue ground"
DOME_INTENSITY = 1000.0
ROOM_OF_SUBJECT = 10.0

handle = KitSceneHandle(stage)
room = handle.add_room()
# `room.py::auto_size`, literally: each axis ten times the subject, snapped up to a multiple of 100
# stage units. On a metre stage that is a 100 m room around a 10 cm orange -- which is the room the
# live rigid captures show, far blue walls and all -- so it is kept, not "corrected".
snap = 100.0
room.set_size(*(max(math.ceil(max(float(s), 0.01) * ROOM_OF_SUBJECT / snap) * snap, snap) for s in size))
room.set_color(*WALL_COLOUR)
room.show_ground(color=GROUND_COLOUR)
handle.lighting.add_dome(intensity=DOME_INTENSITY)
cues = rigid_scene.add_visual_cues(stage, (centre[0], centre[1]))   # its defaults are the rigid runs' cues
print(f"[render] room {tuple(round(max(math.ceil(max(float(s), 0.01) * ROOM_OF_SUBJECT / snap) * snap, snap), 3) for s in size)}, "
      f"floor cues {cues}")

params = camera_follow.compute_target_from_bbox(center=tuple(float(c) for c in centre),
                                                size=tuple(float(s) for s in size))
camera = UsdGeom.Camera.Define(stage, "/RenderCamera")
camera_follow.apply_camera_params(stage, "/RenderCamera", params)
print(f"[render] camera from camera_follow: {params}")


# Anything that moves and was not given a colour gets one. Newton writes the simulated surface
# as a plain mesh with no displayColor, so the asset came out the same grey as the floor and the
# same grey as the plate's shadow -- a correct simulation that reads as nothing happening.
for prim in moving:
    gprim = UsdGeom.Gprim(prim)
    if gprim and not (gprim.GetDisplayColorAttr().HasAuthoredValue() or prim.GetChildren()):
        gprim.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.78, 0.25)])

viewport = omni.kit.viewport.utility.get_active_viewport()
viewport.set_active_camera("/RenderCamera")
viewport.resolution = (args.size, args.size)

timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(start / stage_fps)
timeline.set_end_time(end / stage_fps)
timeline.set_time_codes_per_second(stage_fps)
timeline.stop()

capture_iface = omni.kit.capture.viewport.acquire_capture_interface() if False else None
for _ in range(30):                                    # let materials and the RTX scene resolve
    app.update()

step = max(1, int(round(stage_fps / args.fps)))
frames = list(range(int(start), int(end) + 1, step))
print(f"[render] {len(frames)} frames, every {step} time codes")
for i, t in enumerate(frames):
    timeline.set_current_time(t / stage_fps)
    for _ in range(2):
        app.update()
    path = str(out / f"frame_{i:05d}.png")
    cap = omni.kit.viewport.utility.capture_viewport_to_file(viewport, file_path=path)
    app.update()
    try:
        cap.wait_for_result(0)
    except Exception:
        pass
    app.update()
    if i % 20 == 0:
        print(f"[render] frame {i}/{len(frames)}", flush=True)

# capture_viewport_to_file writes asynchronously, so the last grabs can still be in flight.
# Counting before they land reported a short render that was in fact complete.
for _ in range(20):
    app.update()
written = sorted(out.glob("frame_*.png"))
print(f"[render] wrote {len(written)} of {len(frames)} frames to {out}")
if len(written) != len(frames):
    print("[render] FAIL: frame count does not match")
    app.close()
    sys.exit(3)
print("[render] OK")
app.close()
