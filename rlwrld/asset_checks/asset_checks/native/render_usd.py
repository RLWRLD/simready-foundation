"""Inside Kit: turn an animated USD into frames, with one fixed camera that frames the whole run.

    ./isaac-run isaac610 tools/render_usd.py <animated.usda> <out_dir> [--fps 30] [--size 1024]

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
import pathlib
import sys

import isaacsim
from isaacsim import SimulationApp

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("stage")
ap.add_argument("out_dir")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--show", default="both", choices=("both", "sim", "visual"),
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/app/asyncRendering", False)          # so a grab cannot race the render thread
settings.set("/app/asyncRenderingLowLatency", False)
settings.set("/rtx/pathtracing/spp", 1)

# A recording holds both meshes so the two videos come from one run and line up frame for
# frame. Hiding one is how a camera is pointed at the other; the framing is computed afterwards,
# so it follows whichever is left visible.
HIDDEN = {"sim": "/root/visual", "visual": "/root/sim"}

out = pathlib.Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
omni.usd.get_context().open_stage(args.stage)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

if args.show != "both":
    hide = stage.GetPrimAtPath(HIDDEN[args.show])
    keep = stage.GetPrimAtPath(HIDDEN["sim" if args.show == "visual" else "visual"])
    if not keep or not keep.IsValid():
        print(f"[render] FAIL: this recording has no {args.show} mesh to photograph")
        app.close()
        sys.exit(4)
    if hide and hide.IsValid():
        UsdGeom.Imageable(hide).MakeInvisible()
        print(f"[render] showing the {args.show} mesh; {HIDDEN[args.show]} hidden")

start, end = stage.GetStartTimeCode(), stage.GetEndTimeCode()
stage_fps = stage.GetTimeCodesPerSecond() or 60.0
print(f"[render] {args.stage}: time {start}..{end} at {stage_fps} tcps")


GROUND_SPAN = 50.0   # a prim wider than this is scenery, not the subject


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
    moving, still = [], []
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer)):
            continue
        ranges = [c.ComputeWorldBound(prim).ComputeAlignedRange() for c in caches]
        if any(r.IsEmpty() for r in ranges):
            continue
        size = ranges[0].GetSize()
        if max(size[0], size[1], size[2]) > GROUND_SPAN:
            still.append(prim)
            continue
        if not UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited:
            still.append(prim)     # a hidden mesh must not pull the camera towards itself
            continue
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
bounds = world_bounds(moving, samples)
if bounds.IsEmpty():
    print("[render] FAIL: nothing with bounds in this stage")
    app.close()
    sys.exit(2)
centre = bounds.GetMidpoint()
size = bounds.GetSize()
span = max(size[0], size[1], size[2])
print(f"[render] bounds centre {tuple(round(c, 4) for c in centre)} size {tuple(round(s, 4) for s in size)}")

# A three-quarter view from just above the floor. The obvious camera -- up high, looking down --
# is the wrong one here: a press puts an opaque plate directly between it and the asset, and the
# whole video is a picture of the plate. Low and to the side, you see the gap the plate is
# closing and the asset squeezing out into it, and a drop still reads as plain vertical motion.
distance = max(span * 2.0, 0.30)
eye = Gf.Vec3d(centre[0] + distance * 0.75, centre[1] - distance * 0.95, centre[2] + distance * 0.22)
camera = UsdGeom.Camera.Define(stage, "/RenderCamera")
camera.CreateFocalLengthAttr(28.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(max(1e-3, distance * 0.01), distance * 20.0))
forward = (centre - eye).GetNormalized()
right = Gf.Cross(forward, Gf.Vec3d(0, 0, 1)).GetNormalized()
up = Gf.Cross(right, forward).GetNormalized()
m = Gf.Matrix4d(1.0)
m.SetRow3(0, right)
m.SetRow3(1, up)
m.SetRow3(2, -forward)
m.SetTranslateOnly(eye)
UsdGeom.Xformable(camera).AddTransformOp().Set(m)

if not any(prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase)
           for prim in stage.Traverse()):
    dome = UsdLux.DomeLight.Define(stage, "/RenderDome")
    dome.CreateIntensityAttr(800.0)
    key = UsdLux.DistantLight.Define(stage, "/RenderKey")
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 0.0, 35.0))
    print("[render] the stage had no lights; added a dome and a key")

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
