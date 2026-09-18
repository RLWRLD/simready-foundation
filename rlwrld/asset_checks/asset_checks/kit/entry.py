# SPDX-License-Identifier: Apache-2.0
"""Kit-side entry: `isaacsim <experience> --exec "entry.py <request.json>" ...` (built by run.py).

Writes <out_dir>/result.json with status "done" or "error", the Kit settings and package versions
the run actually had, and the NVIDIA test's result. The process then exits hard, as engine-kit's
kit_runner does: an --exec script cannot end Kit's own loop with post_quit().
"""
import json
import os
import sys
import traceback

def _versions():
    from importlib import metadata

    out = {}
    for name in ("isaacsim", "newton", "mujoco-warp", "torch", "simready-benchmark-engine-kit", "simready-foundation-tier-core"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def main():
    request_path = sys.argv[1]
    req = json.load(open(request_path))
    result = {"status": "error", "request": req}

    async def run():
        import carb.settings
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(60):  # let startup finish before touching the stage, as kit_runner waits for readiness
            await app.next_update_async()
        settings = carb.settings.get_settings()
        result["kit_settings"] = {key: settings.get(key) for key in req["expected_settings"]}
        result["versions"] = _versions()
        from asset_checks.kit import nvidia_test

        if req["experiment"] not in nvidia_test.TESTS:
            raise ValueError(f"unknown experiment {req['experiment']!r}; known: {sorted(nvidia_test.TESTS)}")
        result.update(await nvidia_test.run(req))
        result["status"] = "done"

    try:
        import omni.kit.app
        import omni.kit.async_engine

        task = omni.kit.async_engine.run_coroutine(run())
        app = omni.kit.app.get_app()
        while not task.done():
            app.update()
        task.result()
    except Exception:  # noqa: BLE001 - recorded in result.json, which run.py checks
        result["error"] = traceback.format_exc()
    with open(os.path.join(req["out_dir"], "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


main()
