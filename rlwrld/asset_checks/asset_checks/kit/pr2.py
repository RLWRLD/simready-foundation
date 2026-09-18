# SPDX-License-Identifier: Apache-2.0
"""PR #2's settle / tunnel / tilt criteria (newton15_cert.py, branch rlwrld/newton-conformance),
applied to a trajectory recorded during an NVIDIA test. PR #2 runs its own scenes (a 1 cm drop, a
15 degree walled slope); these numbers describe NVIDIA's scene with PR #2's yardstick."""

# SETTLE_LIN, SETTLE_ANG, SETTLE_HOLD, TUNNEL_DEPTH, EXPLODE_SPEED of newton15_cert.py; its
# "more than 5 frames at 100 Hz" below the support is 0.05 s here.
CRITERIA = {
    "settle_lin_m_s": 2e-2,
    "settle_ang_rad_s": 5e-1,
    "settle_hold_s": 0.5,
    "tunnel_depth_m": 0.01,
    "tunnel_hold_s": 0.05,
    "explode_speed_m_s": 10.0,
    "soft_penetration_m": 0.003,
    "tilt_deg": 20.0,
}


def evaluate(traj, floor=0.0, expect_rest=True):
    """traj: {"t", "lowest_vertex_z", "lin", "ang", "tilt_deg"} lists sampled once per physics step.
    expect_rest=False drops the settle requirement (a scene where the asset should keep moving)."""
    c = CRITERIA
    if not traj["t"]:
        return {"passed": False, "message": "no physics steps recorded", "criteria": c}
    dt = traj["t"][1] - traj["t"][0] if len(traj["t"]) > 1 else 1.0 / 240.0
    depth = [floor - z for z in traj["lowest_vertex_z"]]
    hold = max(1, int(round(c["tunnel_hold_s"] / dt)))
    run_below, tunnel = 0, False
    for d in depth:
        run_below = run_below + 1 if d > c["tunnel_depth_m"] else 0
        tunnel = tunnel or run_below > hold
    settle_n = max(1, int(round(c["settle_hold_s"] / dt)))
    quiet, moved, settle_time = 0, False, None
    for i, (v, w) in enumerate(zip(traj["lin"], traj["ang"])):
        moved = moved or v > c["settle_lin_m_s"]  # the rest before the release is not a settle
        quiet = quiet + 1 if (moved and v < c["settle_lin_m_s"] and w < c["settle_ang_rad_s"]) else 0
        if quiet >= settle_n:
            settle_time = round(traj["t"][i - settle_n + 1], 3)
            break
    exploded = max(traj["lin"]) > c["explode_speed_m_s"]
    final_depth = depth[-1]
    failure = ("flew away / exploded" if exploded else
               "tunnelled into the support" if tunnel or final_depth > c["tunnel_depth_m"] else
               "did not come to rest" if expect_rest and settle_time is None else None)
    notes = []
    max_pen = max(depth)
    if max_pen > c["soft_penetration_m"]:
        notes.append(f"soft contact: dips {max_pen * 1000:.1f} mm below the floor")
    if traj["tilt_deg"][-1] > c["tilt_deg"]:
        notes.append(f"tipped over ({traj['tilt_deg'][-1]:.0f} deg from the start pose)")
    return {"passed": failure is None, "rest_required": expect_rest, "message": failure or "; ".join(notes), "settle_time_s": settle_time,
            "max_penetration_m": round(max_pen, 4), "final_depth_m": round(final_depth, 4),
            "tilt_deg": round(traj["tilt_deg"][-1], 1), "max_speed_m_s": round(max(traj["lin"]), 3), "criteria": c}
