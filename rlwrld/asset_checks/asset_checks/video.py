# SPDX-License-Identifier: Apache-2.0
"""How a recorded run becomes its two videos. One owner, for rigid and deformable runs alike.

A run -- a rigid body's poses, a deformable's points -- is written to an animated USD by
`native/recording.py`. This renders that file twice through `native/render_usd.py`, in the layout
NVIDIA's rigid tests use: once showing the asset's own textured mesh (`visual`) and once the
geometry the engine collides with (`collision`), then encodes each into an mp4 that plays
PLAYBACK_SLOWDOWN times slower than real time. Both pipelines come through here, so two videos
can only differ in physics.
"""
import pathlib
import shutil
import subprocess

CAPTURE_FPS = 60           # frames rendered per simulated second
PLAYBACK_SLOWDOWN = 4.0    # how many times slower than real time the videos play
SIZE = 640                 # px, square: the rigid comparison panels
VIEWS = ("visual", "collision")
NATIVE = pathlib.Path(__file__).resolve().parent / "native"
FFMPEG = shutil.which("ffmpeg")


def slow_tag():
    return f"{PLAYBACK_SLOWDOWN:g}xslower" if PLAYBACK_SLOWDOWN != 1.0 else "realtime"


def record_rigid(bench, result_json, asset, usda, log):
    """Write the animated USD of a rigid cell from its result.json. Runs in the Isaac 6.1.0 venv,
    which has pxr; returns the exit code."""
    cmd = [str(bench / ".venv-isaac610" / "bin" / "python"), str(NATIVE / "recording.py"),
           "--rigid", str(result_json), str(asset), str(usda), "--fps", str(CAPTURE_FPS)]
    with open(log, "w") as out:
        return subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)


def render(bench, usda, frames_dir, view, log, timeout, size=SIZE):
    """Render one view of a recording in Kit; -> the frame files written (none on failure)."""
    cmd = [str(bench / "isaac-run"), "isaac610", str(NATIVE / "render_usd.py"), str(usda), str(frames_dir),
           "--fps", str(CAPTURE_FPS), "--size", str(size), "--show", view]
    with open(log, "w") as out:
        try:
            subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    return sorted(pathlib.Path(frames_dir).glob("frame_*.png"))


def encode(frames_dir, video, keep_frames=False):
    """Frames -> mp4 at the slowed-down rate; the frames go once the video exists, unless kept."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg is not on PATH")
    subprocess.call([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(CAPTURE_FPS / PLAYBACK_SLOWDOWN),
                     "-i", str(pathlib.Path(frames_dir) / "frame_%05d.png"),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video)])
    made = pathlib.Path(video).exists()
    if made and not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return made


def draw(bench, cell, usda, stem, timeout, log_prefix, keep_frames=False):
    """Both views of one cell's recording -> [(view, mp4 name, frame count)], one entry per video made."""
    made = []
    for view in VIEWS:
        frames_dir = cell / f"frames_{view}"
        frames = render(bench, usda, frames_dir, view, cell / f"{log_prefix}render_{view}.log", timeout)
        if not frames:
            print(f"[video] {cell.name}: no {view} frames (see {log_prefix}render_{view}.log)", flush=True)
            continue
        mp4 = cell / f"{stem}__{view}__{slow_tag()}.mp4"
        if encode(frames_dir, mp4, keep_frames):
            made.append((view, mp4.name, len(frames)))
    return made
