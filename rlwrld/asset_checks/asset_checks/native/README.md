# Deformables: run the engine, not the host

```
python asset_checks/native/run.py <asset.usda> --out <dir> \
    --envs newton12_vbd,newton12_xpbd,newton15_vbd,newton15_xpbd,physx \
    --experiments drop,press
```

You name four things -- a USD, an experiment, an engine and a solver -- and this measures the
asset and makes the videos. The principle behind every file here: **the asset's physics and
textures live in the USD; the experiment never changes with the engine; only the engine changes.**
Anything a run has to decide that the USD does not say is printed as ours, with the reason.

## One picture of every run

A rigid run (NVIDIA's tests, through `asset_checks/run.py`) and a deformable run (here) end the
same way. Each writes a recording -- an animated USD of what moved: body poses for a rigid asset,
the solver's nodes for a deformable -- and `video.py` photographs it twice through `render_usd.py`,
in the layout NVIDIA's rigid tests use (their room and colours, our floor cues, their camera):

* `visual`: the asset's own textured mesh, carried along by what the engine moved;
* `collision`: the geometry the engine collides with -- the CollisionAPI meshes the USD declares,
  for a rigid body; the surface the solver moves, for a deformable.

Fixtures the experiment placed -- the press plate, a slope, a gripper's pads -- show in both.
`asset_checks.compare` then puts the environments side by side, one strip per asset, experiment
and view, each panel under the environment's name and its verdict.

Output, per asset, shared with the rigid runner: `<out>/<asset>/<env>/<experiment>/` holding
`recording.usda`, `run.log`, the two videos, a `result.json` the compositor reads; `<out>/compare/`
holding the strips; `<out>/<asset>/summary.md` with the numbers and the engines side by side.

## Why this exists beside the Kit runner

NVIDIA's three tests read rigid-body transforms and refuse an asset without `RigidBodyAPI`, and
Isaac's Fabric sync carries rigid-body transforms only. A deformable driven through that path is
both unmeasurable and invisible: every video of one came out still, because the mesh on the stage
never changes. Here each engine is driven directly, the solver's own state is what gets measured,
and the recording each run writes is what gets photographed.

Every cell is a separate process in its own virtual environment, because the two Newton versions
cannot share one: `isaacsim-core` pins `newton[sim]==1.2.1` against Isaac 6.0.1 and `==1.5.0`
against 6.1.0.

## The files

| file | what it owns |
|---|---|
| `run.py` | the grid: which cells, in which venv; the cell directories; the strips |
| `drop_shape.py` / `press_shape.py` | **the experiments themselves** -- schedule, depth, thresholds, verdicts. No engine imports, no engine parameters. |
| `asset_properties.py` | what the asset declares (material, thickness, radius, self-collision), each with its source; what we chose where it is silent, said out loud |
| `newton_drop.py` / `newton_press.py` | Newton: import, size, colour, step, measure. The shared Newton constants. |
| `physx_drop.py` / `physx_press.py` | the same two, inside Kit, through `DeformablePrim` |
| `to_physx.py` | the same asset re-authored for PhysX: same elements, same declared material |
| `physx_parity.py` | the independent oracle that the conversion carried everything, run before any PhysX cell |
| `usd_deformable.py` | building a cloth from its declaration where an importer produces nothing; finding the simulated prim by what it is |
| `recording.py` | the recording, for a deformable (`Recording`) and a rigid run (`RigidRecording`): one layout |
| `skinning.py` | binding the render mesh to the simulated elements, in the authored pose |
| `render_usd.py` | any recording -> frames: the rigid runs' room, cues and camera, one view at a time |
| `../video.py` | how a recording becomes its two videos, for both pipelines |
| `../compare.py` | the environments side by side, one strip per view |
| `agreement.py` | the engines' answers to one question in one table; deliberately no threshold |
| `check_sources.py` | the static check for the failures that only show at run time: undefined names, use before definition, pxr imported before Kit |

## The things that were not tuning

Each of these was a silent wrong answer, not a crash, and each is now a rule rather than a number.

- **Particle radius** must be set on the `ModelBuilder` *before* `add_usd`. Newton defaults it to
  0.1 m; a centimetre-scale asset imports as particles far larger than itself and every solver
  throws it out of the scene.
- **The CollisionPipeline is built before the solver.** Newton's own error says so. The other
  order leaves a rigid body with no contacts while a static ground still works -- so a drop test
  passes and hides it.
- **VBD's colouring must include the bending edges** (`builder.color(include_bending=True)`, as
  every cloth example Newton ships does). Without them a sheet is stable until it lands, then
  diverges for every stiffness, damping and substep count.
- **`rigid_body_particle_contact_buffer_size` is fixed at 256** and documented as never growing.
  Sized from the asset.
- **XPBD's `soft_body_relaxation` is the only Jacobi averaging there is** -- `apply_particle_deltas`
  never divides by how many constraints touched a particle. It must carry `1/n`, and n is the
  mesh's own connectivity. Newton 1.2.1 spends the same parameter as a compliance instead, and
  never reads the material at all, so the rule is applied only where the kernel means it.
- **Self-collision is the asset's to declare.** No deformable schema in either family has a
  switch for it; where the asset is silent the schema default (off) applies, to every engine
  that has the switch, and the ones that do not say so.
- **The asset is measured after it settles.** This banana loses 16 mm of height just lying down.
- **The press happens where the asset is**, not where it was authored; it rolls while settling.
- **A speed is judged against a speed**: the fall the experiment gives, not the asset's height,
  which is zero for a sheet. And the drop is the starting clearance, not how far a runner lifted.
- **A mesh with animated points carries an animated extent**, or every consumer of bounds --
  the camera included -- reads the authored box at every time.
- **PhysX's starting pose is set through the view**, its collision offsets are sized from the
  asset, its plate is driven through `RigidPrim`, and its floor has thickness.

`docs/newton_official_reference.md` in `simready-bench` has the measurements behind each.
