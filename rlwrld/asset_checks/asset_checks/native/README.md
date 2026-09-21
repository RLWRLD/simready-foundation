# Deformables: run the engine, not the host

```
python asset_checks/native/run.py <asset.usda> --out <dir> \
    --envs newton12_vbd,newton12_xpbd,newton15_vbd,newton15_xpbd,physx \
    --experiments drop,press
```

You name four things -- a USD, an experiment, an engine and a solver -- and this measures the
asset and makes the video.

## Why this exists beside the Kit runner

NVIDIA's three tests read rigid-body transforms and refuse an asset without `RigidBodyAPI`, and
Isaac's Fabric sync carries rigid-body transforms only. A deformable driven through that path is
both unmeasurable and invisible: every video of one came out still, because the mesh on the stage
never changes. Here each engine is driven directly, the solver's own state is what gets measured,
and the animated USD each run writes is what gets photographed.

Every cell is a separate process in its own virtual environment, because the two Newton versions
cannot share one: `isaacsim-core` pins `newton[sim]==1.2.1` against Isaac 6.0.1 and `==1.5.0`
against 6.1.0.

## The files

| file | what it owns |
|---|---|
| `run.py` | the grid: which cells, in which venv, and the videos |
| `press_shape.py` | **the press experiment itself** -- schedule, depth, thresholds. No engine imports. |
| `newton_drop.py` | Newton: import, size, step, verdict. Also the shared Newton constants. |
| `newton_press.py` | Newton: the plate |
| `physx_drop.py` / `physx_press.py` | the same two, inside Kit, through `DeformablePrim` |
| `to_physx.py` | the same asset re-authored for PhysX: same tetrahedra, same declared material |
| `render_usd.py` | any animated USD -> frames, one camera placed on what moves |

## The things that were not tuning

Each of these was a silent wrong answer, not a crash, and each is now a rule rather than a number.

- **Particle radius** must be set on the `ModelBuilder` *before* `add_usd`. Newton defaults it to
  0.1 m; a centimetre-scale asset imports as particles far larger than itself and every solver
  throws it out of the scene.
- **The CollisionPipeline is built before the solver.** Newton's own error says so. The other
  order leaves a rigid body with no contacts while a static ground still works -- so a drop test
  passes and hides it.
- **`rigid_body_particle_contact_buffer_size` is fixed at 256** and documented as never growing.
  Sized from the asset.
- **XPBD's `soft_body_relaxation` is the only Jacobi averaging there is** -- `apply_particle_deltas`
  never divides by how many constraints touched a particle. It must carry `1/n`, and n is the
  mesh's own connectivity. Newton 1.2.1 spends the same parameter as a compliance instead, and
  never reads the material at all, so the rule is applied only where the kernel means it.
- **The asset is measured after it settles.** This banana loses 16 mm of height just lying down.
- **The plate is thicker than the contact margin**, or both of its faces push the asset at once.
- **PhysX's starting pose is set through the view**, its collision offsets are sized from the
  asset, its plate is driven through `RigidPrim`, and its floor has thickness.

`docs/newton_official_reference.md` in `simready-bench` has the measurements behind each.
