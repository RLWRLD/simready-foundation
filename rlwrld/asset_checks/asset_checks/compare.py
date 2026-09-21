"""Side-by-side videos of one run directory: for each asset and test, one mp4 with a panel per
environment (PhysX | Newton 1.2 | Newton 1.5), each labelled with its environment and NVIDIA's
verdict, plus the reason when the test failed or was skipped.

    <bench>/.venv-isaac610/bin/python -m asset_checks.compare <run dir> [--panel-px 640]

Reads each run's result.json (verdict, message, media) and writes <run dir>/compare/<asset>__<test>.mp4.
The panels start together (every environment captures on the same test frames); a shorter video
holds its last frame, and an environment without any footage is a grey panel with its label.

A run whose test aborted mid-loop keeps its captured frames but never reached NVIDIA's video step;
for those this encodes the frames into the run's own <test>_frames.mp4 first, so every cell that
simulated at all has a video. A skipped run never simulated and has no frames.
"""
import argparse
import json
import pathlib
import re
import subprocess
import tempfile

from asset_checks import envs

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FPS = 15


def reason(result):
    lines = (result.get("message") or "").strip().split("\n")
    text = lines[1] if len(lines) > 1 else lines[0]
    return text.split(": ", 1)[-1].strip()


def label(path, width, env, result):
    from PIL import Image, ImageDraw, ImageFont

    verdict = (result.get("verdict") if result else None) or "no verdict"
    colour = {"pass": (90, 200, 110), "fail": (235, 90, 80)}.get(verdict, (235, 180, 60))
    size = max(14, width // 32)
    font = ImageFont.truetype(FONT, size) if pathlib.Path(FONT).exists() else ImageFont.load_default(size)
    img = Image.new("RGB", (width, 3 * size + 12), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.text((8, 4), envs.ENVIRONMENTS[env].note, fill=(235, 235, 235), font=font)
    why = reason(result) if result and verdict != "pass" else ""
    draw.text((8, 6 + size), (verdict.upper() + (": " + why if why else ""))[: width // (size // 2 + 1)], fill=colour, font=font)
    img.save(path)
    return img.size[1]


def video_of(ffmpeg, run_dir, result):
    """The run's video, or one encoded here from its captured frames when the test never wrote one."""
    videos = [m["filename"] for m in (result or {}).get("media") or [] if m.get("kind") == "video"]
    if videos:
        return run_dir / videos[0]
    frames = sorted(run_dir / m["filename"] for m in (result or {}).get("media") or [] if m.get("kind") == "image")
    if not frames:
        return None
    target = run_dir / f"{frames[0].stem.rsplit('_', 1)[0]}_frames.mp4"
    if not target.exists():
        listing = run_dir / "_frames.txt"
        listing.write_text("".join(f"file '{f}'\nduration {1 / FPS:.4f}\n" for f in frames) + f"file '{frames[-1]}'\n")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-vsync", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(target)], check=True)
        listing.unlink()
    return target


def duration(ffmpeg, video):
    out = subprocess.run([ffmpeg, "-i", str(video)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    import imageio_ffmpeg

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--panel-px", type=int, default=640)
    args = ap.parse_args()
    ffmpeg, root, px = imageio_ffmpeg.get_ffmpeg_exe(), pathlib.Path(args.run_dir), args.panel_px
    cells = {}
    for res in root.glob("*/*/*/result.json"):
        asset, env, test = res.parts[-4:-1]
        cells.setdefault((asset, test), {})[env] = res
    out_dir = root / "compare"
    out_dir.mkdir(exist_ok=True)
    written = 0
    for (asset, test), by_env in sorted(cells.items()):
        panels = []
        for env in envs.ENVIRONMENTS:
            result = json.loads(by_env[env].read_text()) if env in by_env else None
            panels.append((env, result, video_of(ffmpeg, by_env[env].parent, result) if env in by_env else None))
        length = max([duration(ffmpeg, v) for _, _, v in panels if v] or [2.0])
        with tempfile.TemporaryDirectory() as tmp:
            inputs, chains = [], []
            for i, (env, result, video) in enumerate(panels):
                band = pathlib.Path(tmp) / f"label_{i}.png"
                label(band, px, env, result)
                if video:
                    inputs += ["-i", str(video)]
                    source = f"[{2 * i}:v]fps={FPS},scale={px}:{px},tpad=stop_mode=clone:stop_duration={length:.2f}"
                else:
                    inputs += ["-f", "lavfi", "-i", f"color=c=0x505050:s={px}x{px}:r={FPS}:d={length:.2f}"]
                    source = f"[{2 * i}:v]null"
                inputs += ["-loop", "1", "-i", str(band)]
                chains.append(f"{source}[v{i}];[{2 * i + 1}:v]fps={FPS}[l{i}];[l{i}][v{i}]vstack=shortest=1[p{i}]")
            graph = ";".join(chains) + ";" + "".join(f"[p{i}]" for i in range(len(panels))) + f"hstack={len(panels)}[out]"
            target = out_dir / f"{asset}__{test}.mp4"
            subprocess.run([ffmpeg, "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                            "-t", f"{length:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(target)], check=True)
            written += 1
    print(f"[compare] {written} videos for {len(cells)} asset x test cells -> {out_dir}")


if __name__ == "__main__":
    main()
