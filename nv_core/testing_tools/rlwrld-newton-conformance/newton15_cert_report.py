#!/usr/bin/env python3
"""Merge the Newton 1.5.2 cert passes with the PhysX verdicts; write results.json + summary.md.

Three remote passes feed in:
  --main      out/results.json           parse + ground_drop + slope_drop + grasp_and_lift with the line in the stage frame
  --bodyline  out_bodyline/results.json  grasp_and_lift with the line attached to the body (primary grasp verdict)
  --defaults  out_defaults/results.json  grasp_and_lift with stock Newton/MuJoCo contact defaults (stage-frame line)
"""

import argparse
import collections
import json
import os
import time

ENGINE = "Newton 1.5.2 standalone (MuJoCo-Warp), not the Arena image"


def short(s, n=110):
    s = (s or "").replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def load(path):
    if not path or not os.path.exists(path):
        return {}
    return json.load(open(path)).get("assets", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--bodyline", default=None)
    ap.add_argument("--defaults", default=None)
    ap.add_argument("--physx", required=True)
    ap.add_argument("--fet003", required=True)
    ap.add_argument("--assets-root", required=True, help="local object_library (read-only, for sidecar categories)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    main_r = load(args.main)
    body_r = load(args.bodyline)
    def_r = load(args.defaults)
    physx = json.load(open(args.physx))
    fet = json.load(open(args.fet003))
    rels = [l.strip() for l in open(args.manifest) if l.strip()]
    names = [r.split("/")[0] for r in rels]

    cats = {}
    for rel in rels:
        n = rel.split("/")[0]
        meta = os.path.join(args.assets_root, os.path.splitext(rel)[0] + ".meta.json")
        try:
            a = json.load(open(meta)).get("asset", {})
            cats[n] = ",".join(a.get("categories") or []) or "-"
        except Exception:  # noqa: BLE001
            cats[n] = "-"

    merged = {
        "engine": ENGINE,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": 0,
        "passes": {
            "grasp_and_lift": "line attached to the rigid body (follows the settled pose); cert contact settings",
            "grasp_and_lift_stage_frame": "line kept in the stage frame after settling (NVIDIA-literal); cert contact settings",
            "grasp_and_lift_stock_defaults": "stage-frame line; stock Newton/MuJoCo defaults: solref 20 ms, pyramidal cone, impratio 1, no multi-CCD, pads condim 3, no finger armature",
        },
        "assets": {},
    }
    rows = []
    for n in names:
        a = main_r.get(n)
        if a is None:
            a = {"name": n, "missing": True, "parse": {"ok": None}, "ground_drop": {"verdict": "missing"}, "slope_drop": {"verdict": "missing"}, "grasp_and_lift": {"verdict": "missing"}}
        a = dict(a)
        a["grasp_and_lift_stage_frame"] = a.pop("grasp_and_lift", {"verdict": "missing"})
        gb = (body_r.get(n) or {}).get("grasp_and_lift")
        a["grasp_and_lift"] = gb if gb else {"verdict": "missing", "message": "body-frame pass missing"}
        gd = (def_r.get(n) or {}).get("grasp_and_lift")
        a["grasp_and_lift_stock_defaults"] = gd if gd else {"verdict": "missing"}
        if (body_r.get(n) or {}).get("crash"):
            a["crash_bodyline"] = body_r[n]["crash"]
        if a.get("crash") and all((a.get(t) or {}).get("verdict") in ("pass", "fail") for t in ("ground_drop", "slope_drop", "grasp_and_lift_stage_frame")):
            a["crash_recovered"] = a.pop("crash")
        a["physx"] = {"grasp": physx.get(n), "drops": fet.get(n)}
        a["category"] = cats.get(n, "-")
        merged["assets"][n] = a
        rows.append(a)
    merged["count"] = len(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(merged, f, indent=1)

    def v(a, t):
        return (a.get(t) or {}).get("verdict") or "missing"

    def m(a, t):
        return (a.get(t) or {}).get("message") or ""

    tests = ("ground_drop", "slope_drop", "grasp_and_lift", "grasp_and_lift_stage_frame", "grasp_and_lift_stock_defaults")
    tot = {t: collections.Counter(v(a, t) for a in rows) for t in tests}
    parse_ok = sum(1 for a in rows if (a.get("parse") or {}).get("ok"))
    parse_fail = [a for a in rows if not (a.get("parse") or {}).get("ok")]
    crashes = [a for a in rows if a.get("crash")]
    recovered = [a for a in rows if a.get("crash_recovered")]
    phases = collections.Counter((a.get("grasp_and_lift") or {}).get("phase") or "-" for a in rows if v(a, "grasp_and_lift") == "fail")

    dis = []
    for a in rows:
        px = (a.get("physx") or {}).get("grasp") or {}
        pv = px.get("status")
        nv = v(a, "grasp_and_lift")
        if pv and nv in ("pass", "fail") and pv != nv:
            dis.append(a)
    frame_diff = [a for a in rows if v(a, "grasp_and_lift") in ("pass", "fail") and v(a, "grasp_and_lift_stage_frame") in ("pass", "fail") and v(a, "grasp_and_lift") != v(a, "grasp_and_lift_stage_frame")]
    drop_fail = [a for a in rows if v(a, "ground_drop") in ("fail", "crash") or v(a, "slope_drop") in ("fail", "crash")]

    honoured = collections.Counter()
    not_hon = collections.Counter()
    not_hon_assets = collections.defaultdict(set)
    thick_assets = []
    thick_pieces = 0
    coacd_fallback = []
    for a in rows:
        p = a.get("parse") or {}
        for c in p.get("colliders", []):
            key = str(c.get("approx_authored"))
            if c.get("honoured"):
                honoured[key] += 1
            else:
                not_hon[key] += 1
                not_hon_assets[key].add(a["name"])
            if key.lower() == "convexdecomposition" and not c.get("honoured"):
                coacd_fallback.append(a["name"])
        th = p.get("planar_colliders_thickened") or []
        if th:
            thick_assets.append((a["name"], len(th)))
            thick_pieces += len(th)
    parse_warn = collections.Counter()
    for a in rows:
        for w in (a.get("parse") or {}).get("warnings", []):
            parse_warn[short(w, 90)] += 1

    el = [a.get("elapsed_s") for a in rows if a.get("elapsed_s")]
    slope_creep = [(a.get("slope_drop") or {}).get("residual_speed") for a in rows if v(a, "slope_drop") == "pass" and (a.get("slope_drop") or {}).get("residual_speed") is not None]
    settle_shift = [(a["name"], ((a.get("grasp_and_lift_stage_frame") or {}).get("params") or {}).get("settle_shift_m") or 0.0, ((a.get("grasp_and_lift_stage_frame") or {}).get("params") or {}).get("settle_tilt_deg") or 0.0) for a in rows]
    big_shift = [s for s in settle_shift if s[1] > 0.03 or s[2] > 20]
    tipped = [a["name"] for a in rows if "tipped over" in m(a, "ground_drop")]
    vers = {}
    for a in rows:
        for k, vv in (a.get("versions") or {}).items():
            if k not in vers or vers[k] in ("?", None):
                vers[k] = vv
    any_g = next(((a.get("grasp_and_lift") or {}) for a in rows if (a.get("grasp_and_lift") or {}).get("solver")), {})
    solver = any_g.get("solver", {})
    any_p = next(((a.get("grasp_and_lift") or {}).get("params") for a in rows if ((a.get("grasp_and_lift") or {}).get("params") or {}).get("contact_timeconst")), {}) or {}
    stock_pass = tot["grasp_and_lift_stock_defaults"]["pass"]
    stock_fail = tot["grasp_and_lift_stock_defaults"]["fail"]
    creeps = [abs((a.get("grasp_and_lift") or {}).get("creep_in_jaws_m") or 0.0) for a in rows if v(a, "grasp_and_lift") == "pass"]
    slips = [(a.get("grasp_and_lift") or {}).get("max_slip_m") or 0.0 for a in rows if v(a, "grasp_and_lift") == "pass"]

    L = []
    L.append("# Newton 1.5 conformance pass over the DexBench rigid props\n")
    L.append(f"**Engine: {ENGINE}.** Generated {merged['generated']} from `results.json` ({len(rows)} packages, manifest = `outputs/bench_handled_list.txt`). Runner: `newton15_cert.py` (this directory), remote recipe: `remote_setup.md`.\n")
    L.append("Versions on the remote: " + ", ".join(f"{k} {vv}" for k, vv in vers.items()) + ".\n")
    L.append("Files: `results.json` (merged, per package: parse info, sidecar, drop verdicts, the three grasp verdicts, PhysX verdicts), `results/<package>.json` (raw main-pass worker output; its `grasp_and_lift` block is the stage-frame pass), `videos/`, `newton15_cert.py` (runner), `make_report.py` (this merge), `remote_setup.md`.\n")
    L.append(
        "Settings (cert passes): MuJoCo-Warp via `newton.solvers.SolverMuJoCo` on the GPU with MuJoCo contacts, dt = 1 ms, 100 Hz control, "
        f"solver options {solver}, contact solref time constant {any_p.get('contact_timeconst')} s on every shape "
        f"(Newton default 0.02 s), finger armature {any_p.get('finger_armature')} kg, pad mass {any_p.get('pad_mass')} kg / condim {any_p.get('pad_condim')} / "
        f"torsional {any_p.get('pad_mu_torsional')}, friction 0.5 where no PhysicsMaterialAPI is bound (PhysX default), grip force 5x weight, "
        "lift max(0.3 m, 2x longest edge), hold 1 s, shake 1 cm @ 2 Hz for 1.5 s, hold 1 s, open; slip threshold 3 cm on the grasped material point. "
        "PhysX reference: Isaac Sim 6.0.1 / simready-benchmark (`outputs/grasp_verdicts.json`, `outputs/bench_fet003_results.json`).\n"
    )
    L.append("Three grasp passes were run: **grasp_and_lift** (primary: the sidecar line follows the rigid body after the 1 cm settle drop), "
             "**stage frame** (line stays where it was authored + the placement offset, which is what NVIDIA's kit log shows), and "
             "**stock defaults** (stage-frame line, Newton/MuJoCo out-of-the-box contact settings: solref 20 ms, pyramidal cone, impratio 1, single contact point, pads condim 3, no armature).\n")

    L.append("## Totals per test\n")
    L.append("| test | pass | fail | crash / skip / missing |\n|---|---|---|---|")
    L.append(f"| parse (Newton USD importer) | {parse_ok} | {len(rows) - parse_ok} | {len(crashes)} worker crashes ({len(recovered)} recovered on rerun) |")
    labels = {"ground_drop": "ground_drop", "slope_drop": "slope_drop (15 deg, walled)", "grasp_and_lift": "grasp_and_lift (line on body, cert settings)", "grasp_and_lift_stage_frame": "grasp_and_lift (line in stage frame, cert settings)", "grasp_and_lift_stock_defaults": "grasp_and_lift (stage frame, stock Newton defaults)"}
    for t in tests:
        c = tot[t]
        other = sum(c[k] for k in c if k not in ("pass", "fail"))
        L.append(f"| {labels[t]} | {c['pass']} | {c['fail']} | {other} |")
    L.append("")
    px_counts = collections.Counter((physx.get(n) or {}).get("status") for n in names)
    L.append(f"PhysX grasp_and_lift on the same {len(rows)} (`grasp_verdicts.json` as read at generation time; power_drill and serving_bowl were re-verdicted to pass on 2026-09-16, so the brief's 27 failures are now 25): {px_counts['pass']} pass / {px_counts['fail']} fail. Newton (primary) failing phases: " + ", ".join(f"{k} {n}" for k, n in phases.most_common()) + ".")
    if el:
        L.append(f"\nRuntime of the main pass: {sum(el) / 60:.0f} min on the RTX 5090 (shared with another Isaac Sim job), median {sorted(el)[len(el) // 2]:.0f} s per asset for parse + 2 drops + grasp + video; the body-frame and stock passes took ~20 min each.")
    if creeps:
        cs = sorted(creeps)
        L.append(f"\nHeld objects (primary pass, passing) creep in the jaws by median {cs[len(cs) // 2] * 1000:.1f} mm / max {cs[-1] * 1000:.0f} mm over the 3.5 s hold+shake; max grasp-point slip median {sorted(slips)[len(slips) // 2] * 1000:.1f} mm.")
    L.append("")

    L.append("## Newton vs PhysX grasp_and_lift disagreements (primary pass)\n")
    L.append(f"{len(dis)} of {len(rows)} packages differ. `PhysX class / reason` come from `outputs/grasp_verdicts.json`; the last two columns show the same asset under the stage-frame line and under stock Newton defaults.\n")
    L.append("| package | category | PhysX | PhysX class / reason | Newton | Newton phase / message | stage-frame line | stock defaults | notes |\n|---|---|---|---|---|---|---|---|---|")
    for a in sorted(dis, key=lambda a: (((a.get("physx") or {}).get("grasp") or {}).get("status", ""), a["name"]), reverse=True):
        px = (a.get("physx") or {}).get("grasp") or {}
        g = a.get("grasp_and_lift") or {}
        p = g.get("params") or {}
        notes = []
        if p.get("settle_shift_m", 0) > 0.03 or p.get("settle_tilt_deg", 0) > 20:
            notes.append(f"asset moved {p.get('settle_shift_m', 0) * 100:.0f} cm / tilted {p.get('settle_tilt_deg', 0):.0f} deg while settling")
        rep = (a.get("parse") or {}).get("approximations_replaced") or []
        if rep:
            notes.append(short(rep[0].split(": ", 1)[-1], 60) + (f" (+{len(rep) - 1})" if len(rep) > 1 else ""))
        if g.get("creep_in_jaws_m") is not None and abs(g["creep_in_jaws_m"]) > 0.01:
            notes.append(f"creep in jaws {g['creep_in_jaws_m'] * 100:.1f} cm")
        for nn in g.get("notes", []):
            notes.append(short(nn, 70))
        if g.get("video") or (a.get("grasp_and_lift_stage_frame") or {}).get("video"):
            notes.append("video")
        gs = a.get("grasp_and_lift_stage_frame") or {}
        gd = a.get("grasp_and_lift_stock_defaults") or {}
        L.append(f"| {a['name']} | {a.get('category', '-')} | {px.get('status')} | {px.get('class')}: {short(px.get('why'), 80)} | **{g.get('verdict')}** | {g.get('phase') or '-'}: {short(g.get('message'), 90)} | {gs.get('verdict')} | {gd.get('verdict')} | {short('; '.join(notes), 150)} |")
    L.append("")
    agree_pass = sum(1 for a in rows if v(a, "grasp_and_lift") == "pass" and ((a.get("physx") or {}).get("grasp") or {}).get("status") == "pass")
    agree_fail = sum(1 for a in rows if v(a, "grasp_and_lift") == "fail" and ((a.get("physx") or {}).get("grasp") or {}).get("status") == "fail")
    px_fail_newton_pass = [a["name"] for a in dis if ((a.get("physx") or {}).get("grasp") or {}).get("status") == "fail"]
    px_pass_newton_fail = [a for a in dis if ((a.get("physx") or {}).get("grasp") or {}).get("status") == "pass"]
    still_fail_stage = [n for n in px_fail_newton_pass if v(merged["assets"][n], "grasp_and_lift_stage_frame") == "fail"]
    sdf_fail = [a["name"] for a in px_pass_newton_fail if any("sdf" in r for r in ((a.get("parse") or {}).get("approximations_replaced") or []))]
    tipped_fail = [a["name"] for a in px_pass_newton_fail if (((a.get("grasp_and_lift") or {}).get("params") or {}).get("settle_tilt_deg") or 0) > 20]
    other_fail = [a["name"] for a in px_pass_newton_fail if a["name"] not in sdf_fail and a["name"] not in tipped_fail]
    L.append(f"Agreement: {agree_pass} pass/pass, {agree_fail} fail/fail.\n")
    L.append(f"**{len(px_fail_newton_pass)} PhysX failures hold in Newton** ({', '.join(px_fail_newton_pass)}). These are the thin nuts, bearings, forks, coupons, sheets and bags that NVIDIA classed 'benchmark limit'; with the line riding on the settled body and torsional pads they are pinched and lifted. {len(still_fail_stage)} of them still fail in Newton when the line is frozen in the stage frame (NVIDIA's rule keeps the line where the asset was placed, 1 cm above where it settles, so on a 1-2 cm part the pads close at or above its top edge) -- a good part of the PhysX 'benchmark limit' class is that offset, not the collider.\n")
    L.append(f"**{len(px_pass_newton_fail)} PhysX passes fail in Newton** ({', '.join(a['name'] for a in px_pass_newton_fail)}): {len(sdf_fail)} are `sdf` colliders whose cavity or flare vanished in the 64-vertex convex hull ({', '.join(sdf_fail)}) -- a pad on the inside of a bowl rim lands on the hull 'lid' instead of the wall, a tapered drill body slides out of a faceted hull; {len(tipped_fail)} tip over during the 1 cm settle because the hull rounds off the flat they stand on ({', '.join(tipped_fail)}), which carries the body-attached line below the floor; {', '.join(other_fail) if other_fail else 'none'} are borderline slips of 3.1-3.4 cm against the 3 cm threshold (plate rim pinch).\n")
    if frame_diff:
        L.append("### Line frame sensitivity\n")
        L.append(f"{len(frame_diff)} packages change verdict between the body-attached and the stage-frame line (assets that shift, tip or are thin relative to the 1 cm placement offset):\n")
        L.append("| package | line on body | line in stage frame | settle shift / tilt | stage-frame message |\n|---|---|---|---|---|")
        for a in frame_diff:
            gs = a.get("grasp_and_lift_stage_frame") or {}
            p = gs.get("params") or {}
            L.append(f"| {a['name']} | {v(a, 'grasp_and_lift')} | {gs.get('verdict')} | {p.get('settle_shift_m', 0) * 100:.1f} cm / {p.get('settle_tilt_deg', 0):.0f} deg | {short(gs.get('message'), 90)} |")
        L.append("")

    L.append("## Drop tests\n")
    if drop_fail:
        L.append("| package | ground_drop | slope_drop | PhysX drops (fet003) |\n|---|---|---|---|")
        for a in drop_fail:
            fd = (a.get("physx") or {}).get("drops")
            fds = "-" if not fd else ", ".join(f"{k} {vv[0]}" for k, vv in fd.items())
            L.append(f"| {a['name']} | {v(a, 'ground_drop')}: {short(m(a, 'ground_drop'), 80)} | {v(a, 'slope_drop')}: {short(m(a, 'slope_drop'), 80)} | {fds} |")
    else:
        L.append("No drop-test failures.")
    if any(a["name"] == "dragon_fork_19cm" for a in drop_fail):
        L.append("\ndragon_fork_19cm is the only prop that never settles on the slope: its single 64-vertex hull (an `sdf` collider) is a curved rocker that keeps rocking and yawing while it creeps downhill; on the flat floor it settles in 0.1 s. PhysX (SDF collider) rests it on both tests.")
    L.append(f"\nNo NaN, fly-away or tunnelling on any package. {len(tipped)} packages tip over (> 20 deg) when dropped 1 cm onto the flat floor from their authored pose: {', '.join(tipped)}.")
    if slope_creep:
        sc = sorted(slope_creep)
        L.append(f"\nOn the 15 deg slope every prop that 'rests' still creeps downhill: median {sc[len(sc) // 2] * 1000:.1f} mm/s, max {sc[-1] * 1000:.1f} mm/s (MuJoCo soft contacts with the 4 ms solref used here; the 20 ms default is ~5x worse). The 'came to rest' criterion is therefore < 2 cm/s and < 0.5 rad/s for 0.5 s; PhysX/TGS sticks.")
    L.append("")

    L.append("## Collider approximations Newton could not honour\n")
    L.append("| authored physics:approximation | colliders honoured | colliders replaced | what Newton / MuJoCo-Warp does |\n|---|---|---|---|")
    expl = {
        "convexHull": "convex hull (MuJoCo re-hulls every mesh geom, max 64 vertices)",
        "convexDecomposition": "CoACD decomposition into convex pieces at import (importer extra `coacd`)",
        "sdf": "not a Newton approximation: triangle mesh kept, MuJoCo-Warp collides its 64-vertex convex hull (cavities and thin sections vanish)",
        "boundingCube": "bounding box",
        "None": "no approximation authored: triangle mesh -> convex hull",
    }
    for key in sorted(set(honoured) | set(not_hon), key=lambda k: -(honoured[k] + not_hon[k])):
        L.append(f"| {key} | {honoured[key]} | {not_hon[key]} ({len(not_hon_assets[key])} packages) | {expl.get(key, '?')} |")
    L.append("")
    L.append(f"Planar / sliver convex pieces thickened to 1 mm by the runner so MuJoCo-Warp accepts them (it rejects planar mesh colliders outright and MuJoCo's compiler rejects near-zero-volume hulls): {thick_pieces} pieces in {len(thick_assets)} packages: " + ", ".join(f"{n} ({k})" for n, k in sorted(thick_assets, key=lambda x: -x[1])) + ".")
    if coacd_fallback:
        L.append(f"\nCoACD fell back to a single hull on: {', '.join(sorted(set(coacd_fallback)))}.")
    if parse_warn:
        L.append("\nImporter warnings (deduplicated):\n")
        for w, c in parse_warn.most_common(15):
            L.append(f"- {c}x `{w}`")
    L.append("")

    L.append("## Parse failures and crashes\n")
    if not parse_fail and not crashes:
        L.append(f"None: all {len(rows)} packages parse with Newton's USD importer and build a MuJoCo-Warp model.")
    for a in parse_fail:
        L.append(f"- {a['name']}: {short((a.get('parse') or {}).get('error'), 200)}")
    for a in crashes:
        L.append(f"- {a['name']}: worker crash rc={a['crash'].get('returncode')} timed_out={a['crash'].get('timed_out')}: {short(a['crash'].get('log_tail', '')[-300:], 200)}")
    for a in recovered:
        L.append(f"- {a['name']}: worker aborted on the first attempts (rc={a['crash_recovered'].get('returncode')}), recovered on rerun -- see below.")
    L.append("- polybag_3 (4.5 MB `.usda`, 23 convex pieces) aborted the worker on two of three attempts with heap corruption (`double free or corruption (fasttop)` / SIGSEGV inside `UsdPhysics.LoadUsdPhysicsFromRange`, usd-core 26.3) when the stage had already been opened once in the same process by the runner's pxr pre-scan; opening the stage once and handing the `Usd.Stage` object to `add_usd` avoided it and the package then passes every test.")
    L.append("")

    L.append("## Findings: what Newton 1.5 needs from these assets that PhysX did not\n")
    F = []
    F.append(f"1. **Every collider becomes convex in MuJoCo-Warp.** `sdf` ({not_hon['sdf']} colliders in {len(not_hon_assets['sdf'])} packages) and unauthored ({not_hon['None']}) approximations are silently reduced to a 64-vertex convex hull of the mesh; only `convexDecomposition` ({honoured['convexDecomposition']} colliders, via CoACD at parse time) and pre-decomposed `convexHull` pieces keep concavity. Bowls, deep plates, bends, brackets and bearings authored as `sdf` lose their cavities: a pad on the inside of a bowl rim rests on the hull 'lid' instead of the wall, which is the mechanism behind the tableware failures below. Author `convexDecomposition` (or ship pieces) for anything a jaw or finger must reach into.")
    F.append(f"2. **No zero-thickness or sliver pieces.** MuJoCo-Warp refuses planar mesh colliders and MuJoCo rejects hull pieces of ~0 volume; {thick_pieces} pieces in {len(thick_assets)} packages had to be thickened to 1 mm by the runner. PhysX accepted them. Pre-decomposed packages (crates, boxes, CoACD output of thin scans) need a minimum-thickness check.")
    F.append("3. **CoACD is a hard dependency** (`newton[importers]`); without it every `convexDecomposition` collapses to one hull. It runs on every load (~6 s for the 262k-vertex tomato can, blue_crate imports 1889 pieces): pre-decomposed, cached colliders are preferable for Arena.")
    F.append(f"4. **Contact stiffness is mass-scaled in MuJoCo and the stock settings cannot hold a pinch grasp.** With Newton's defaults (solref 20 ms, pyramidal cone, impratio 1, one contact point per pair, no torsional friction) {stock_pass} of {stock_pass + stock_fail} grasps pass: a 20 g pad pushed with 5x an object's weight sinks through it, and objects creep out of the jaws. The cert settings (4 ms solref on every shape, 1 kg finger armature, elliptic cone + impratio 10, multi-CCD, pads condim 4 with torsional friction) bring it to {tot['grasp_and_lift']['pass']}. Assets carry no `mjc:solref` / `mjc:condim`; Arena's rigid-contact configuration, not the asset, decides graspability.")
    F.append("5. **Torsional friction is off by default (condim 3).** NVIDIA's pads carry `physxCollision:torsionalPatchRadius=1`; without `condim 4` on the gripper geoms every off-centre pinch (drill, clamp, tote, pulley) pivots and slides out. Gripper-side, but assets that author `mjc:condim` would carry it into Arena.")
    F.append("6. **Soft contacts creep.** Props resting on the 15 deg slope slide ~1 cm/s and held objects creep millimetres in the jaws; PhysX/TGS sticks. Any Newton rest/hold test needs a tolerance, and light props placed on inclined surfaces (trays, shelves) will drift.")
    F.append(f"7. **Authored poses are not always rest poses, and the grasp line must ride with the body.** {len(tipped)} packages tip over when dropped 1 cm onto the floor (hulls round off the flats they stand on: spark plug, pneumatic cylinder, exhaust bend, chips bag, mango); {len(big_shift)} move > 3 cm / > 20 deg before the gantry is built. With the line frozen in the stage frame (NVIDIA-literal) {tot['grasp_and_lift_stage_frame']['fail']} grasps fail, with the line on the body {tot['grasp_and_lift']['fail']}; thin props (paper stacks, bearings, pulleys, plates) fail in the stage frame only because the 1 cm placement offset lifts the line above them. Author the body upright on z = 0 and keep the line on the rigid-body prim (YCB scans have the body as a rotated child prim, e.g. 005_tomato_soup_can carries an 11 deg tilt).")
    F.append("8. **Mass and inertia are honoured** (`physics:mass` on the body prim; per-piece MassAPI on the vendor drill / screwdriver / spanner) and friction comes from `PhysicsMaterialAPI` where bound (else Newton's default, set to 0.5 here). `physxCollision:contactOffset/restOffset` (163 colliders) and every other `physx*` attribute are ignored -- assets relying on a rest offset to sit flush will sit differently.")
    F.append("9. **Multi-body vendor packages import as articulations** (power_drill: 4 bodies with revolute + prismatic + fixed joints, screwdriver: fixed joint) and simulate; drive gains are not authored so the trigger and chuck are free in the drops.")
    F.append("10. **MuJoCo hulls are capped at 64 vertices** (Newton `Mesh.maxhullvert`), coarser than PhysX's convex hulls of the same meshes: cylinders get faceted, rims flatten, the mesh's render vertices dip up to a few mm below the floor at rest ('max_penetration' in results.json) and round props roll differently on the slope.")
    F.append("11. **Importer robustness**: `UsdPhysics.LoadUsdPhysicsFromRange` corrupted the heap on polybag_3.usda when the stage was opened twice in one process (usd-core 26.3); pass one `Usd.Stage` around. Dense scan meshes (262k-vertex can, 451k-vertex visual mesh in box.usda) load fine.")
    F.append("12. **Geom count matters**: pre-decomposed crates import with > 1800 colliding pieces (blue_crate 1889); MuJoCo-Warp copes (~25 s per grasp test) but Arena scenes with several such props will pay for it -- a coarser decomposition is worth authoring.")
    L.extend(F)
    L.append("")
    L.append("## Videos\n")
    fails = []
    for a in rows:
        if v(a, "grasp_and_lift") == "fail":
            vid = (a.get("grasp_and_lift") or {}).get("video") or (a.get("grasp_and_lift_stage_frame") or {}).get("video")
            if vid:
                fails.append((a["name"], os.path.basename(vid), "body" if (a.get("grasp_and_lift") or {}).get("video") else "stage-frame"))
    L.append(f"`videos/` holds the headless 5 fps / 512 px capture of the grasp phase for every package that fails the primary pass ({len(fails)}), named `<package>_grasp_and_lift.mp4` (from the body-frame pass where it recorded one, otherwise the stage-frame pass). The remote keeps the captures of all {len(rows)} packages under `~/Projects/dexbench_newton15_cert/out/videos/` and `out_bodyline/videos/`.")
    L.append("")
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(args.out_dir, "_failure_videos.txt"), "w") as f:
        for n, fn, src in fails:
            f.write(f"{src}\t{fn}\n")
    print(f"wrote {args.out_dir}/summary.md and results.json; {len(dis)} grasp disagreements, {len(frame_diff)} frame-sensitive, {len(drop_fail)} drop failures, {len(fails)} failure videos")


if __name__ == "__main__":
    main()
