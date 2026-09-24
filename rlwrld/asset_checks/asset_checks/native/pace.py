"""How fast the engine steps, against the clock: a measurement on every run.

An asset that is right but steps 400x slower than real time cannot be teleoperated or used for
policy inference, and nothing else in a result says so. `realtime_x` is simulated seconds per
wall second of stepping only -- the substep loops, synchronised on the device, the first frame
left out (it compiles the kernels). Above 1 is faster than real time. The GPU's utilisation
before the run is printed with it, because another job on the same device makes the number
someone else's.
"""
import subprocess
import time

import warp as wp


def gpu_busy():
    """The visible GPU's utilisation in percent before stepping, or None if it cannot be read."""
    try:
        import os
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Pace:
    def __init__(self, tag, fps):
        self.tag, self.fps, self.wall, self.frames, self._t = tag, fps, 0.0, 0, None
        busy = gpu_busy()
        self.busy = busy
        print(f"[{tag}] GPU utilisation before stepping: "
              + (f"{busy}%" if busy is not None else "unreadable"))

    def start(self, frame):
        if frame == 0:
            self._t = None
            return
        wp.synchronize_device()
        self._t = time.perf_counter()

    def stop(self):
        if self._t is None:
            return
        wp.synchronize_device()
        self.wall += time.perf_counter() - self._t
        self.frames += 1

    def tokens(self):
        if not self.frames or self.wall <= 0.0:
            return ""
        x = self.frames / self.fps / self.wall
        return (f"realtime_x={x:.4g} step_ms_per_frame={1e3 * self.wall / self.frames:.3g}"
                + (f" gpu_busy_before={self.busy}" if self.busy is not None else ""))
