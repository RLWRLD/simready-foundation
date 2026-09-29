# Deformables: run the engine, not the host

```
python asset_checks/native/run.py <asset.usda> --bench <simready-bench> --out <dir> \
    [--experiments drop,press] [--setup auto-auto-auto-auto]
```

Usually reached through `asset_checks.check ... --engine newton1.5 --solver vbd`. Deformables are
checked in one environment: **Newton 1.5 VBD** (`envs.DEFORMABLE`; the scope decision of
2026-09-29 removed Newton 1.2, XPBD and PhysX deformables). This measures the asset and makes the
videos. The principle behind every file here: **the asset's physics and
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
never changes. Here Newton is driven directly, the solver's own state is what gets measured,
and the recording each run writes is what gets photographed.

Every cell is a separate process in the Isaac 6.1.0 venv, which carries Newton 1.5.0.

## A run refuses rather than guesses

- **Every vendor attribute is accounted for** (`asset_properties.account`): an authored attribute
  the run neither reads nor sets aside by its setup, and that is not classified in
  `asset_properties.ACCOUNTED`, refuses the run; so does a quantity stated twice whose two
  statements disagree (e.g. `newton:kMu` against `physics:youngsModulus`/`poissonsRatio`).
- **The setup** (`setups.py`, default `setups.CANON = auto-auto-auto-auto`) says where each number
  the USD does not settle comes from; half a recipe is refused, not completed.
- **The kernel's unit is read off its source**: damping is handed to SolverVBD in the unit Newton
  1.5.0's kernels read, and a Newton whose kernels read it otherwise is refused.
- **Every RESULT line carries `realtime_x`** (`pace.py`): simulated seconds per second of stepping.

## The files

| file | what it owns |
|---|---|
| `run.py` | the cells of one asset, their directories, videos, result.json, summary and strips |
| `drop_shape.py` / `press_shape.py` | **the experiments themselves** -- schedule, depth, thresholds, verdicts. No engine imports, no engine parameters. |
| `newton_drop.py` / `newton_press.py` | the two experiments' scenes: the asset lifted over a floor / set down under a kinematic plate, and what is measured |
| `loader.py` | **the one loading path**: a deformable USD into a Newton builder with everything it states (radii, damping, surface stiffness, seal springs, contact exclusions, self-contact, contact numbers) and the audit that refuses what nobody read |
| `newton_scene.py` | what both experiments share outside the asset: contact band, collision pipeline, stepping, the substep loop (`Stepper`), element arrays |
| `asset_properties.py` | what the asset declares, each with its source; the vendor recipe; `account`, which refuses unread or contradicting attributes |
| `setups.py` | where every number the USD does not settle comes from; `CANON` |
| `usd_deformable.py` | reading the declared bodies, materials, structure (seals, exclusions) and render meshes |
| `integrity.py` | whether a bag stayed a bag: seal gaps and contents outside the film, in every RESULT |
| `pace.py` | `realtime_x`: the engine's pace against the clock |
| `stepping.py` | the canon stepping: substeps, iterations, fps |
| `recording.py` | the recording, for a deformable (`Recording`) and a rigid run (`RigidRecording`): one layout |
| `skinning.py` | binding the render mesh to the simulated elements, in the authored pose |
| `render_usd.py` | any recording -> frames: the rigid runs' room, cues and camera, one view at a time |
| `../video.py` | how a recording becomes its videos, for both pipelines; a failed render makes none |
| `../compare.py` | the environments side by side, one strip per view |
| `check_sources.py` | static check: undefined names, use before definition, a USD/engine module imported before Kit (computed from the imports) |
| `check_rules.py` | static check: no asset named in code, experiments know no engine, experiments declared once |

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
- **Self-collision is the asset's to declare.** No deformable schema in either family has a
  switch for it; where the asset is silent the schema default (off) applies, to every engine
  that has the switch, and the ones that do not say so.
- **The asset is measured after it settles.** This banana loses 16 mm of height just lying down.
- **The press happens where the asset is**, not where it was authored; it rolls while settling.
- **A speed is judged against a speed**: the fall the experiment gives, not the asset's height,
  which is zero for a sheet. And the drop is the starting clearance, not how far a runner lifted.
- **A mesh with animated points carries an animated extent**, or every consumer of bounds --
  the camera included -- reads the authored box at every time.
- **A vendor's recipe is read, or the run refuses.** The polybags' fold edge exclusions sat unread
  for two days (2026-09-22..24) while the bag they hold together flew apart at its own stepping; they
  are Newton's `add_cloth_mesh` edge ids, and now applied.

`rlwrld/asset_checks/docs/newton_official_reference.md` has the measurements behind each.
