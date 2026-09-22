# What Newton's own examples do, and where ours differs

Read 2026-09-21 out of the installed packages, not the web: the shipped
`newton/examples/` tree is the primary source and it matches the version we run.

| example | venv | what it is |
|---|---|---|
| `softbody/example_softbody_franka.py` | 601 + 610 | Franka **grasps** a tet-mesh rubber duck (VBD) |
| `softbody/example_softbody_hanging.py` | 601 + 610 | soft beams hanging, damping sweep |
| `multiphysics/example_softbody_dropping_to_cloth.py` | 601 + 610 | soft body **dropped** onto cloth |
| `multiphysics/example_rigid_soft_contact.py` | 610 | sphere **presses** a soft FEM beam (xpbd / vbd / semi-implicit / coupled) |
| `vbd/example_vbd_gripper_soft_grid.py` | 610 | parallel-jaw gripper closes, **grips**, lifts, holds |

## Every one of them drives Newton directly

`ModelBuilder` -> `builder.color()` (VBD) -> `finalize()` -> `CollisionPipeline` ->
per substep `clear_forces()`, `pipeline.collide()`, `solver.step()`. None of them goes
through Isaac Sim. Isaac is a host application, not the engine under test.

## Numbers, quoted

| knob | official | ours (before today) |
|---|---|---|
| `sim_substeps` | 10 (softbody), 20 (gripper), 32 (rigid-soft) | **5** |
| `iterations` | 5-10 | 15 |
| `soft_contact_kd` | 0 / 1e-5 / 1e-3 / 1e1 / 2e-1 | **1e2** |
| `soft_contact_ke` | 1e2 .. 2e6, paired with the above | 1e5 floor |
| particle radius | `builder.default_particle_radius = 0.01` **before** `add_usd` | per-particle after import |
| `soft_contact_margin` | 0.01 (meter scale) | 4 x radius |

`soft_contact_kd` is the one that is not a matter of taste: every official example is
between 0 and 2e-1, and we ran 1e2 -- two to seven orders of magnitude of damping the
asset never asked for.

## The flag that explains the press failures

`CollisionPipeline(..., enable_rigid_soft_full_surface_contact=True)` -- Newton 1.5.0 only.
The gripper example's own docstring says it plainly: with the flag off, only
per-particle contact exists, "the grid slips out and falls". A plate pressing a
deformable, or a floor holding one, is the same geometry problem. Newton 1.2.1's
`CollisionPipeline` has no such parameter, so 1.2 is expected to be worse at contact
here -- that is a property of the engine, which is exactly what we are measuring.

## Rendering does not need Isaac

`newton.viewer.ViewerUSD` exists in both versions and `log_state` writes particles
(`_log_particles`) and cloth triangles (`_log_triangles`) as animated USD. Isaac's
Fabric sync writes rigid body transforms only, which is why every deformable video so
far was still.

## What the two Newton versions actually do with the asset's material

Measured 2026-09-21 on `assets/fruits/banana.usda`, which declares
`physics:youngsModulus = 4800000`, `physics:poissonsRatio = 0.4`, and says so about itself:
*"stiffness variant: E is the literature value x 6, requested for handover experiments"*.
Newton 1.5.0 derives exactly the Lame pair those imply -- `k_mu = 1.714e6`,
`k_lambda = 6.857e6`. So does 1.2.1's importer.

Then the two solvers part ways. `xpbd/kernels.py::solve_tetrahedra`:

| | Newton 1.2.1 | Newton 1.5.0 |
|---|---|---|
| compliance | `relaxation` (a constant 0.9) | `inv_rest_volume / k_mu` |
| the material lines | commented out | read |
| `relaxation` | not applied to the correction | multiplies the correction |

**Newton 1.2.1's XPBD does not read the asset's stiffness at all.** Negative test, the same
drop with the tet material multiplied by 100: 1.2.1 answers 0.01423 m and 0.01471 m -- a
hundredfold change in stiffness moves the result by 3%. Any stiffness comparison run on
1.2.1 + XPBD is measuring nothing about the asset.

1.5.0 reads it and, at 4.8 MPa, diverges: z reaches +-94 m in the first frame. It is the tet
solve alone -- with contacts switched off it still explodes, with the tet solve switched off it
falls cleanly. Not a tuning shortfall either: 32 -> 1024 substeps only halves the blow-up per
doubling (94, 21, 11, 5.5, 2.7 m), and 10 -> 40 iterations does not fix it.

## The knob that does fix it, and the rule behind it

`apply_particle_deltas` adds every constraint's correction to a particle **in full**; it never
divides by how many constraints touched it. `soft_body_relaxation` is therefore the only
averaging in the Jacobi sweep, and SolverXPBD passes it to `solve_tetrahedra` and nowhere else.
A particle shared by n tet constraints is moved n times too far unless relaxation carries 1/n:

    relaxation = 1 / (2 constraints x 4 particles x tet_count / particle_count)

The banana: 2 x 4 x 12936 / 3074 = 33.7 -> 0.030. Measured: 0.9 and 0.3 diverge; 0.1 settles at
29.5 mm thick and 0.03 at 28.2 mm, against VBD's 29.9 mm. The rule is the mesh's own
connectivity, so it holds for an asset nobody has seen yet, and it is capped at SolverXPBD's
own default so a coarse or soft asset is left alone.

## Two more things the examples had right and we had wrong

**Build the CollisionPipeline before the solver.** `softbody/example_softbody_franka.py` -- the
grasping one -- constructs the pipeline first, and SolverVBD's own error message says why:
*"Pre-size before capture by constructing CollisionPipeline before SolverVBD"*. It sizes its
per-body contact state from the contacts that exist when it is built. Built the other way round
a static ground still works (a plane hangs off body -1 and needs no per-body list) so a drop
test passes and hides it; a rigid body in the scene gets no contacts at all.

**The per-body particle contact list is fixed at 256 and never grows.** Both versions'
`SolverVBD` docstring says `rigid_body_particle_contact_buffer_size` "will not be dynamically
resized during runtime", and Newton's own release notes describe the symptom -- contacts past
the limit are dropped without a word. A 3074-particle asset needs it sized from the asset.

## Newton 1.2.1's VBD does not drive a jointed body

Measured, by counting soft contacts per shape during a press:

| | ground contacts | plate contacts | plate body_q |
|---|---|---|---|
| Newton 1.2.1 | 1149 | **0** | 0.0589 -> 0.0586 (it never moved) |
| Newton 1.5.0 | 4988 | 3126 | 0.0590 -> 0.0393 (it pressed) |

The prismatic drive the gripper example uses is simply not actuated by 1.2.1's VBD; the
docs' own feature matrix calls it "rigid bodies with limited joint support". So the press
plate is **kinematic** -- its pose prescribed each substep, its velocity handed to the solver
for friction. That also removes a confound worth naming: a PD-driven plate stops at a depth
that depends on how hard the asset pushes back, so every engine would press to a different
depth. Prescribing the motion presses every engine's asset by the same amount and leaves the
asset's response as the only variable. Newton 1.2.1 + VBD then compresses the banana 13.0 mm
and it recovers 100%.

## Measure the asset after it settles, not as authored

The banana's authored pose is not its resting shape: lying down under gravity costs it 16 mm
of height before any plate moves. A press that takes its "start height" from frame zero reports
that settling as compression it never caused -- and worse, it aims the plate at a height the
asset no longer has, so the plate stops in the air and presses nothing. The press now measures
at the end of its settle phase and derives the plate's depth from that.


## A correction: Newton 1.2.1 does simulate cloth

An earlier version of this document, and of the code, treated "Newton 1.2.1 imports no particles
from a `PhysicsSurfaceDeformableSimAPI` mesh" as an engine limit. That was a guess and it was
wrong. Newton 1.2.1 ships **eight** cloth examples and has `ModelBuilder.add_cloth_mesh`; two of
those examples (`example_cloth_bending.py`, `example_cloth_franka.py`) open a USD stage and hand
the mesh's points and indices straight to it. What 1.2.1 lacks is one importer path -- its
`import_usd.py` contains no reference to a surface deformable at all -- and an importer gap is not
a capability boundary.

`asset_checks/native/usd_deformable.py` does what those examples do, with the conversion taken
verbatim from Newton 1.5.0's own `import_usd_deformable_cloth.py`:

    tri_ke  = stretchStiffness * thickness       # membrane stiffness ~ E*h
    edge_ke = bendStiffness * thickness**3       # bending ~ E*h^3
    tri_ka  = 0                                  # the proposal authors no Poisson term
    density = volumetric density * thickness     # Newton's cloth density is areal
    particle_radius = 0.5 * thickness            # the shell's physical half-thickness
    shearStiffness: dropped -- Newton's isotropic membrane shares one modulus with stretch

Verified by running both paths under 1.5.0 on the same asset:

| | 1.5.0's importer | ours |
|---|---|---|
| particles / triangles / edges | 441 / 800 / 1240 | 441 / 800 / 1240 |
| `tri_materials[0]` | [10, 0, 10, 0, 0] | [10, 0, 10, 0, 0] |
| bending stiffness | 1.0000002e-12 | 1.0000002e-12 |
| particle radius / total mass | 0.000500 / 0.008000 | 0.000500 / 0.008000 |

It runs only where the importer produced nothing, so an importer that works always wins.

**PhysX takes the surface quantities one for one** (`omniphysics:surfaceStretchStiffness`,
`surfaceShearStiffness`, `surfaceBendStiffness`, `surfaceThickness`) and therefore honours the
shearStiffness Newton drops. That is a real difference between the engines on this asset, and it
belongs in the results rather than in a footnote.

## What comes from the asset, and what does not

An audit of what actually reached the solver found four places where it was not the asset:

| | was | now |
|---|---|---|
| particle radius | Newton's importer ignores `newton:particleRadius`; we derived 1.39 mm | the asset's 0.76 mm, read back from the builder after the geometry is in |
| friction | per solver, from two examples: VBD 0.3, XPBD 1.0 | the asset's if declared, else one value for every engine and solver |
| shape materials | `fill_()` across every shape | only the fixtures the experiment adds |
| PhysX friction | `deformableUtils`' own default, 0.25, against Newton's 0.5 | the same single fallback both read |

Every run now prints which numbers came from the asset, with the prim and attribute that carried
each, and which are ours with the reason. What is legitimately ours is only what the USD has no
way to express: the experiment's fixtures, the solver's numerics, and a stated fallback where the
asset is silent.

## VBD's colouring must include the bending edges

`builder.color()` leaves them out unless asked, and VBD sweeps one colour at a time holding the
others fixed: two particles joined only by a bending constraint then share a colour and the
constraint is solved against a stale neighbour. Our cloth has 1240 of them.

The failure is not a wobble. The sheet is stable until it lands and then diverges **on the frame
it lands**, for every contact stiffness (1e2 to 2e6), every damping (1e2 to 1e-2) and every
substep count (10 to 200) -- twelve runs, all frame 6. That uniformity is the tell: a resolution
limit cannot produce it.

`example_cloth_poker_cards.py` -- cloth dropped onto a ground plane, the shipped example of
exactly this situation -- calls `builder.color(include_bending=True)`, and `color`'s own docstring
says to set it whenever the model has bending edges. With it, 31.70 mm below the floor becomes
0.00 mm and the sheet lies at z = 0.00049 m: one particle radius, the same rest height PhysX and
both XPBDs give.

**A sweep taken off a broken model argues for the wrong thing.** Contact stiffness 1e2 to 2e6
moved the penetration 31.70 -> 0.22 mm, a clean monotone curve. It was measured on the
mis-coloured model, and with the colouring right the value we already had is the best of the lot.
Worth keeping from it: every shipped `soft_contact_ke` of 1e2 is in an example whose cloth *hangs*
and touches nothing; the ones that touch something use 1e4 to 2e6.

## Self-collision belongs to the asset

`SELF_CONTACT = {"volume": False, "surface": True}` was a physical claim keyed on the element
type: a cloth self-collided and a soft body did not. No deformable schema in either family has a
switch for it -- the AOUSD/OmniPhysics surface deformable has only a `selfCollisionFilterPose`
purpose, and PhysX's `physxParticle:selfCollision` is on a schema these assets never apply -- so
where the asset is silent the deformable schema's own default answers, and
`physxDeformableBody:selfCollision` defaults to **off**, the opposite of what we had.

Of the five environments only Newton's VBD has the switch. SolverXPBD has none, and the
OmniPhysics schemas the PhysX conversion applies declare none, so each runner prints which case it
is in rather than letting a reader assume all five ran the same model. Measured on this cloth the
answer does not move: z 0.00050-0.00051 with it on, 0.00050-0.00055 with it off.

Newton 1.2.1's VBD cannot run this cloth with it off -- it diverges on the frame the sheet lands,
and with it on the sheet survives but never settles (53 mm of crumple at 0.37 m/s after two
seconds). Every cloth example 1.2.1 ships that has a ground plane turns self-contact on. 1.5.0 is
correct either way. That is the engine result, and the asset's declaration still decides.

## Two thresholds that were measuring the wrong thing

**The drop is the starting clearance, not the lift.** The runners passed how far *they* raised the
asset where the experiment asked how far it can fall. An asset authored at the drop height is
lifted by nothing, so every threshold built on it collapsed to zero: a cloth lying perfectly still
at 0.000 m/s read `never-settled`, and `never-fell` could never fire at all.

**A speed is judged against a speed.** `2.0 * height` is zero for a sheet, so it fell back to the
contact size -- a distance used as a speed. The drop states its own scale: `sqrt(2 g clearance)`,
0.99 m/s here, and 2% of that is the threshold. It is looser for a sheet and tighter for a large
asset, so PhysX's banana at 62 mm/s now reads `never-settled` where the height-scaled rule passed
it. After two seconds on the floor it is still jittering, so that is the rule saying something
true.

## A mesh with animated points must carry an animated extent

`UsdGeom.BBoxCache` reads an authored extent instead of the points, so a mesh referenced from the
asset keeps the asset's static box at every time. The cloth's render mesh fell 49.5 mm and
reported never moving; the renderer's "what moves?" found nothing and framed the floor instead.
A freshly created mesh has no extent and escaped it. Every consumer of bounds was wrong, not only
the camera.

## The collision video must show what the engine collides with

None of the eleven SimReady sample assets applies `UsdPhysics.MeshCollisionAPI` or authors an
approximation, so PhysX cannot use their triangle meshes on a dynamic body and cooks a convex hull
instead -- it logs that as an error on every load, and engine-kit's
`physics_utils.report_collision_approximations` warns by name. The dishwand's sponge collides as a
151 cm3 hull, not the 124 cm3 shape it looks like; the toaster collides with its bread slots
filled in.

`reading.collider_approximation` applies PhysX's rule -- a bare `physics:approximation` without
the schema applied is ignored, a static collider keeps what it declares, a dynamic one that
declares nothing usable gets a hull -- and `reading.collider_shape` builds it. A decomposition or
an SDF is more than a copy of geometry can say, so there the mesh is kept and the run prints which
approximation is really in force.

**scipy's `ConvexHull.simplices` are not consistently wound.** Orient each against
`hull.equations` or the mesh renders inside out in patches. The tell was hulls measuring *less*
volume than the meshes they contain, which is impossible.

The deformables need none of this: the banana hands PhysX its own tetrahedra, one TetMesh that is
both simulation and collision mesh, and the cloth is a surface deformable with one mesh. The
separate `collision_tetmesh` only appears where PhysX has to cook tetrahedra itself.

## Testing "does this solver read the material?" -- soften it, do not stiffen it

The evidence quoted here for years was that multiplying the banana's stiffness by 100 moved
Newton 1.2.1's answer by 3%. Re-measured, x100 moves *neither* version: 1.2.1 settles 3.86 -> 3.92
mm and 1.5.0 25.59 -> 25.61. XPBD with a finite iteration count is already at its stiff limit, so
nothing can get stiffer, and the test separates the two cases not at all.

Softening does. At x0.0001 the asset's shear modulus falls from 1.71 MPa to 171 Pa:

| k_mu | x1 | x0.01 | x0.0001 |
|---|---|---|---|
| Newton 1.2.1 XPBD, settled thickness | 4.02 mm | 3.91 | **3.96** |
| Newton 1.5.0 XPBD, settled thickness | 25.59 mm | 24.60 | **10.89** |

1.5.0 reads it and collapses; 1.2.1 does not move, exactly as its kernel says -- the lines that
would read `materials[tid, 0]` and `[tid, 1]` are commented out and `stretching_compliance` is
assigned the relaxation constant instead.

This is why the banana flattens to 3.9 mm under 1.2.1's XPBD and 25.6 mm under 1.5.0's while both
are handed identical inputs: the two runs log the same asset values (radius 0.00076 m, density
1010, E 4.8 MPa, nu 0.4, each with the prim and attribute that carried it) and the same contact
numerics (ke 75, kd 1, kf 1000, mu 0.5). The one thing set per version is `soft_body_relaxation`
-- 0.9 against 0.0297 -- because the name means a compliance in one and a Jacobi factor in the
other, and giving 1.2.1 the other version's value makes its tetrahedra thirty times stiffer and
kills the run. That is a solver numeric, not the asset's material.

## How a frame advances, and how many steps are in it (2026-09-22)

**In Kit, `app.update()` does not mean "one substep".** Isaac's documentation: *"PhysX determines
how many physics steps to calculate based on the TimeStepsPerSecond and the variable elapsed time
since the last call."* One update advances the timeline by `1/timeCodesPerSecond` and PhysX takes
`timeStepsPerSecond x elapsed` steps inside it. With the scene's rate left at its default of 60,
calling update N times per recorded frame runs **N/60 s of simulation per frame at dt = 1/60**, not
one frame at dt = 1/(60N).

Measured, on the apple: 27.245 mm of fall in the first recorded frame, where four steps of 1/60 s
from rest give `g*dt^2*n(n+1)/2` = 27.25 mm and the 1/60 s the frame claimed would give 1.4 to 2.7.
Every PhysX deformable video before this played four times fast and every contact was solved on a
step four times coarser than Newton's. `asset_checks/native/physx_scene.py` sets
`PhysxSchema.PhysxSceneAPI.CreateTimeStepsPerSecondAttr(fps * substeps)`, reads it back, and calls
`app.update()` **once** per recorded frame.

**One step rate for every engine**, `asset_checks/native/stepping.py`: `SUBSTEPS = 32`, the finest
any of these solvers' own examples ask for (the rigid-soft example above). Per-solver counts made
the same experiment three different amounts of work -- measured on the apple press, PhysX walked
towards Newton's answer as its step shrank (below floor 4.0 -> 2.6 -> 1.4 mm at 240 -> 480 -> 960 Hz
against Newton VBD's 0.0), so a failing verdict at 240 Hz measured the number we chose. It costs
almost nothing: 39 s at 1920 Hz against 32 s at 240 Hz, because a run is bound by what surrounds
the step.

Only the step is shared. Colouring, compliance and contact offsets still come from each engine's
own examples.

**The guard.** `drop_shape.free_fall(fps)` is what a released asset falls in one frame: between
`g/(2*fps^2)` (infinitely many substeps) and `g/fps^2` (one), for any solver. Every drop records
`first_frame_x`, the measured fall over `g/fps^2`. Falling *short* of that raises -- nothing an
asset or a solver does makes gravity weaker, so only the pipeline can. Falling *long* is recorded
and not judged: XPBD moves a bag of film 11.2x further than gravity can in its first frame, on both
Newton versions, and then settles it perfectly.
