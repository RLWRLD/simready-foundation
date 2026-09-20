# asset_checks

Physics checks of one SimReady asset in three Isaac Sim environments, side by side:

| env | Isaac Sim | engine |
|---|---|---|
| `physx` | 6.1.0 | PhysX |
| `newton12` | 6.0.1 | Newton 1.2.1 |
| `newton15` | 6.1.0 | Newton 1.5.0 |

```bash
PYTHONPATH=rlwrld/asset_checks python3 -m asset_checks.run \
    --bench ~/Workspace/Research/Robotics/simready-bench --out <new dir> \
    --experiments drop,slope,grasp [--newton-contact pr2] <asset.usd> [...]
```

Side-by-side videos (PhysX | Newton 1.2 | Newton 1.5 per asset and test, labelled with NVIDIA's
verdict) from a finished run directory, with the bench venv's ffmpeg and PIL:

```bash
PYTHONPATH=rlwrld/asset_checks <bench>/.venv-isaac610/bin/python -m asset_checks.compare <out dir>
```

`--bench` is the directory with the two Isaac Sim venvs, `isaac-run` and the `GPU` file. Every
asset x environment x experiment runs in its own Kit process; `<out>/summary.md` is the table and
each run keeps `request.json`, `result.json` (verdict, NVIDIA's metrics and logs, trajectory, what
Kit and the solver actually ran with), `kit.log`, the video and its captured frames.

## What is NVIDIA's and what is ours

Taken as is: the Foundation 7.1 tests registered in `simready_benchmark_kit_suite` (FET003
`ground_drop` and `slope_drop`, FET005 `grasp_and_lift`), run through their registry with their
own config defaults, prechecks, verdicts, metrics and video; engine-kit's scene building, stepping,
camera follow and capture; NVIDIA's static validator, whose passed features the tests see.
Nothing of NVIDIA's is modified.

Added here, because those pieces assume PhysX:
- **Pose reads** (`kit/reading.py`): PhysX writes simulated poses to USD, Newton only to Fabric.
  A pose that is not finite raises "physics diverged" instead of dropping out of a bound.
- **Engine hooks** (`kit/nvidia_api.py`, `kit/nvidia_test.py`): engine-kit 2026.6.5 lacks
  `physics_utils.active_physics_engine` and `fabric_utils`; they are supplied only when missing, and
  `result.json` lists what was supplied. Under Newton, asset bounds and the camera follow read the
  live Fabric poses, and PhysX-only scene preparation is skipped.
- **Camera** (`--camera`, `nvidia_test.Scene`): `fixed` (default) places one camera at the first
  camera update and keeps it: it frames the asset and the gripper at that moment, down to the floor
  and up by the test's `lift_max_height`, with engine-kit's `compute_target_from_bbox`. The slope
  test does not bound how far the asset slides and keeps engine-kit's `follow`. `result.json`
  records which. Captures are engine-kit's size (1024 px) unless `--capture-px` is given. The frames
  are Isaac Sim RTX renders of NVIDIA's test scene (grey room, one dome light, no viewport light rig).
- **Floor cues** (`scene.add_visual_cues`): in that room floor, far walls and horizon blend and
  nothing casts a shadow, so a 10 cm matte checkerboard (visual only: no collider) and a near-overhead
  key light are added under `/World/AssetChecksLook` when a test sets up its camera; each tile is a
  10 cm size reference. `--plain-scene` renders the room as NVIDIA builds it.
- **Runtime physics variant** (`kit/scene.py`): the asset's variant for the engine is selected, and
  the result says whether the variant's payload schemas actually composed.
- **Launch** (`envs.py`, `run.py`): one GPU (`/physics/cudaDevice`, `/renderer/multiGpu/enabled`,
  `/renderer/activeGpu`), a torch CUDA check before any Kit starts, `SIMREADY_PHYSICS_RUNTIME` for
  the grasp test's feature gate, and a run counts only if Kit reports the engine, settings, Newton
  version and pose source that were asked for.
- **Stepping** (`nvidia_test.Proxy.physics_step`): engine-kit 2026.6.5 pauses the timeline after
  the first update of a play so each step is one frame, but never clears its `_physics_paused` flag;
  in a test that plays twice (the grasp) the second play ran at 2 frames per step, more on capture
  steps. The flag is cleared on each play's first step. `run.py` measures the outcome from the
  timeline: every step after a play's first must advance exactly one frame, or the run is INVALID.
- **Trajectory** (`nvidia_test.Recorder`): the asset's rigid bodies every physics step, with the
  timeline's time (`timeline_t`) and Newton's simulated time (`sim_t`); PR #2's settle / tunnel /
  tilt criteria (`kit/pr2.py`, from `newton15_cert.py` on branch `rlwrld/newton-conformance`) are
  computed on it for drop and slope.
- **Newton contact profile** (`--newton-contact`, `kit/contact.py`): `stock` leaves Isaac's
  defaults; `pr2` authors PR #2's MuJoCo contact settings (4 ms solref on every collider, condim 4
  with 0.05 torsional friction on the gripper pads, elliptic cone, impratio 10) before every play.
  Either way `solver_seen` records what MuJoCo actually compiled, joint armature and damping included.
- **Diagnostics, off by default**: `--dump-physics` writes `physics_play<n>.npz` per play (every
  array of the compiled MuJoCo model and its GPU copy, the Newton model, Isaac's Newton config);
  `--trace-contacts N` records the solver's pad/asset contacts (count, normal force, depth) every N
  steps.

## Adding an experiment

An experiment is one of NVIDIA's registered tests. `kit/nvidia_test.py` names the ones this runs:

```python
TESTS = {
    "drop":  ("simready_benchmark_kit_suite.fet003_physics.ground_drop", "ground_drop"),
    "slope": ("simready_benchmark_kit_suite.fet003_physics.slope_drop",  "slope_drop"),
    "grasp": ("simready_benchmark_kit_suite.fet005_grasp.grasp_and_lift", "grasp_and_lift"),
}
```

To add one, put `"<your name>": ("<module that registers it>", "<its registered name>")` in that
table and pass `--experiments <your name>`. `registered()` imports the module and takes the test the
registry holds under that name, failing if it is not registered exactly once; the test's own
`config_defaults` are its parameters, and anything passed in the request must be a key it declares.
The installed suite registers more than these three -- `fet001_visual`, `fet004_multibody`,
`fet011_semantics`, `fet022_driven_joints`, `fet028_gripper` -- and a test of your own registered
with NVIDIA's `@test` decorator works the same way.

What the runner gives every experiment, whatever it is: the asset with its runtime physics variant
selected, the solver asked for, one frame per physics step, live poses under Newton, the fixed
camera and floor cues, the per-step trajectory, and a result that is INVALID unless the environment,
solver, stepping, pose source and media all came out as asked. What an experiment has to bring is
its own scene, its own phases and its own verdict -- none of which this runner touches.

## Known limits

- Newton 1.2.1 reads a collider's authored mass only when its rigid body also has MassAPI; NVIDIA's
  sample props author mass on the collider alone, so Newton 1.2 recomputes it from density 1000
  (the apple 0.57x, the coffee cup 0.14x its authored mass). Newton 1.5 and PhysX read it.
- Isaac's Newton stage gives every joint that authors no armature `cfg.armature` = 0.1 kg m^2
  (PhysX: 0). On a light articulated prop this outweighs the links' own inertia and the joints
  barely move; `solver_seen.dof_armature` shows it.
- NVIDIA's grasp precheck (`check_physics_ready`) applies PhysX's cooking rules under every engine,
  so a dynamic collider with approximation `none` skips the grasp test under Newton too.
- In NVIDIA's sample props an explicit `apiSchemas` list on the collider discards the runtime
  payload's schemas; `payload schemas lost` in the summary counts them.
