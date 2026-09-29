# SPDX-License-Identifier: Apache-2.0
"""How a recorded run becomes its two videos. One owner, for rigid and deformable runs alike.

A run -- a rigid body's poses, a deformable's points -- is written to an animated USD by
`native/recording.py`. This renders that file twice through `native/render_usd.py`, in the layout
NVIDIA's rigid tests use: once showing the asset's own textured mesh (`visual`) and once the
geometry the engine collides with (`collision`), then encodes each into an mp4 at every speed in
SPEEDS. Both pipelines come through here, so two videos can only differ in physics.
"""
import pathlib
import shutil
import subprocess

CAPTURE_FPS = 60           # frames rendered per simulated second
# Every speed a view is encoded at, from the one set of frames -- encoding again costs an ffmpeg
# pass where rendering again would cost a Kit launch. Real time is what the asset actually did;
# adding `"4xslower": 4.0` here is the whole of what it takes to have a slow one beside it.
SPEEDS = {"realtime": 1.0}
SIZE = 640                 # px, square: the rigid comparison panels
VIEWS = ("visual", "collision")
NATIVE = pathlib.Path(__file__).resolve().parent / "native"
FFMPEG = shutil.which("ffmpeg")


def describe():
    """The sentence a summary uses for a cell's videos, from VIEWS and SPEEDS rather than retyped."""
    return (f"{len(VIEWS) * len(SPEEDS)} video(s) per cell, from one run: "
            + ", ".join(f"`__{v}`" for v in VIEWS) + " at " + ", ".join(f"`__{s}`" for s in SPEEDS)
            + " (`__collision` is the geometry the solver moved, `__visual` the asset's own mesh carried by it).")


def record_rigid(bench, result_json, asset, usda, log):
    """Write the animated USD of a rigid cell from its result.json. Runs in the Isaac 6.1.0 venv,
    which has pxr; returns the exit code."""
    cmd = [str(bench / ".venv-isaac610" / "bin" / "python"), str(NATIVE / "recording.py"),
           "--rigid", str(result_json), str(asset), str(usda), "--fps", str(CAPTURE_FPS)]
    with open(log, "w") as out:
        return subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)


def render(bench, usda, frames_dir, view, log, timeout, size=SIZE):
    """Render one view of a recording in Kit; -> (frame files, None) or ([], why it failed).

    `render_usd.py` exits 2 when the recording has no bounds and 3 when it rendered fewer frames
    than the recording holds; a timeout leaves a partial set. Any of them is a failed render, and
    a video made from what it left would pass for a whole one."""
    cmd = [str(bench / "isaac-run"), "isaac610", str(NATIVE / "render_usd.py"), str(usda), str(frames_dir),
           "--fps", str(CAPTURE_FPS), "--size", str(size), "--show", view]
    with open(log, "w") as out:
        try:
            code = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [], f"render timed out after {timeout} s"
    if code != 0:
        return [], f"render exited {code}"
    return sorted(pathlib.Path(frames_dir).glob("frame_*.png")), None


def encode(frames_dir, video, slowdown):
    """Frames -> mp4, played `slowdown` times slower than the capture rate."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg is not on PATH")
    subprocess.call([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(CAPTURE_FPS / slowdown),
                     "-i", str(pathlib.Path(frames_dir) / "frame_%05d.png"),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video)])
    return pathlib.Path(video).exists()


def draw(bench, cell, usda, stem, timeout, log_prefix, keep_frames=False):
    """Every view of one cell's recording at every speed.

    -> ([(view, speed, mp4 name, frame count)], [what failed]). A cell's videos are a view and a
    speed: `visual` and `collision` are two pictures of one run; each speed is one reading of it.
    A view whose render failed makes no video, and the failure is returned for the cell's record.
    """
    made, failed = [], []
    for view in VIEWS:
        frames_dir = cell / f"frames_{view}"
        frames, why = render(bench, usda, frames_dir, view, cell / f"{log_prefix}render_{view}.log", timeout)
        if why or not frames:
            failed.append(f"{view}: {why or 'no frames'} (see {log_prefix}render_{view}.log)")
            print(f"[video] {cell.name}: {failed[-1]}", flush=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
            continue
        for speed, slowdown in SPEEDS.items():
            mp4 = cell / f"{stem}__{view}__{speed}.mp4"
            if encode(frames_dir, mp4, slowdown):
                made.append((view, speed, mp4.name, len(frames)))
            else:
                failed.append(f"{view} {speed}: ffmpeg made no {mp4.name}")
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
    return made, failed
