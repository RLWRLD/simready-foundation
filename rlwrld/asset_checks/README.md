# asset_checks

Physics checks of one SimReady asset in three Isaac Sim environments, side by side:

| env | Isaac Sim | engine |
|---|---|---|
| `physx` | 6.1.0 | PhysX |
| `newton12` | 6.0.1 | Newton 1.2.1 |
| `newton15` | 6.1.0 | Newton 1.5.0 |

```bash
PYTHONPATH=rlwrld/asset_checks python3 -m asset_checks.run \
    --bench ~/Workspace/Research/Robotics/simready-bench --out <new dir> <asset.usd> [...]
```

`--bench` is the directory with the two Isaac Sim venvs, `isaac-run` and the `GPU` file. Every
asset x environment x experiment runs in its own Kit process; `<out>/summary.md` is the table and
each run keeps `request.json`, `result.json` (verdicts, trajectory, what Kit actually ran with),
`kit.log` and its captured frames.

## What is NVIDIA's and what is ours

Taken as is: engine-kit's scene building, frame stepping and capture; the Foundation 7.1 test's
procedure, parameters (read from the registered test) and stability checks (`stability.py`).
Nothing of NVIDIA's is modified.

Added here, because NVIDIA's pieces assume PhysX:
- **Pose reads** (`kit/reading.py`): PhysX writes simulated poses to USD, Newton only to Fabric.
- **Runtime physics variant** (`kit/scene.py`): the asset's variant for the engine is selected, and
  the result says whether the variant's payload schemas actually composed.
- **Launch** (`envs.py`, `run.py`): one GPU (`/physics/cudaDevice`, `/renderer/multiGpu/enabled`,
  `/renderer/activeGpu`), a torch CUDA check before any Kit starts, and a run counts only if Kit
  reports the engine, settings and Newton version that were asked for.

## Experiments

- `drop` -- NVIDIA's FET003 `ground_drop`: dropped from twice its bounding-box height, verdict by
  NVIDIA's touch / penetration / rest rules. The settle / tunnel / tilt criteria of PR #2
  (`newton15_cert.py`, branch `rlwrld/newton-conformance`) are computed on the same trajectory,
  from collider vertices as PR #2 measures them. PR #2 itself drops from 1 cm.

## Known limits

- Under Newton the camera does not follow the asset: engine-kit's camera follow reads USD. Verdicts
  do not use the camera.
- In NVIDIA's sample props an explicit `apiSchemas` list on the collider discards the runtime
  payload's schemas; `payload schemas lost` in the summary counts them.
