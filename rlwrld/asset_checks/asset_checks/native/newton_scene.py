"""What both Newton experiments share outside the asset: the contact band, the collision pipeline,
the stepping, the substep loop, and the element arrays a recording binds.

`loader.py` builds the asset; the experiment (newton_drop, newton_press) adds its fixtures and its
motion; this is the rest, once, so the two experiments -- and a collaborator's own scene -- step the
same engine the same way.
"""
import inspect

import warp as wp

import newton

import asset_properties
import setups
import stepping

# How far out a soft contact is generated, in particle radii. Newton's examples say 0.01 m, but
# they are metre-scale scenes; on a 17 cm asset that margin is a centimetre of empty space around
# every particle, and it forces every other length in the scene -- the press plate has to be
# thicker than it -- to a size that has nothing to do with the asset. Two radii is the asset's own
# resolution, which is what everything else here is derived from.
CONTACT_MARGIN_OF_RADIUS = 2.0


def contact_margin(radius, substeps, fps, drop, depth=0.0, gravity=9.81):
    """How far out to look for contact: wide enough for everything the experiment asks of it.

    A penalty contact exists only while the particle is inside the band, so the band is also the
    deepest indentation the model can represent (measured: a plate driven 10 mm into a band of
    1.5 mm touched a shell of 95 of 3074 particles and swept through the rest). `depth` is the
    indentation the experiment intends; `drop` is the fall it intends, which sets how far a
    particle moves in one substep -- sqrt(2 g drop) / (fps substeps) -- and a band narrower than
    that stride lets a particle pass the floor between two substeps with no contact in between.
    """
    stride = (2.0 * gravity * max(drop, 0.0)) ** 0.5 / (fps * substeps)
    return max(CONTACT_MARGIN_OF_RADIUS * radius, depth, stride)


def collision_pipeline(model, margin, tag):
    """Newton's CollisionPipeline with full-surface rigid-soft contact (a shape meets the surface
    between particles, not only the particles; the gripper example's docstring says a mesh slips
    out of the jaws without it). Built before the solver, always: SolverVBD sizes its per-body
    contact state from the contacts that exist, and a rigid body built the other way round
    generates no contact at all."""
    if "enable_rigid_soft_full_surface_contact" not in inspect.signature(newton.CollisionPipeline.__init__).parameters:
        raise SystemExit("this Newton's CollisionPipeline has no full-surface soft contact; this package is for Newton 1.5")
    print(f"[{tag}] soft contact margin {margin * 1000:.2f} mm; full-surface soft contact on", flush=True)
    return newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=margin,
                                    enable_rigid_soft_full_surface_contact=True)


def resolve_stepping(asset, setup_name, fps, substeps=0, iterations=None, tag="run"):
    """-> (setup, substeps, iterations) from the setup, with explicit `substeps`/`iterations`
    allowed only over the canon stepping: a number given twice has two owners."""
    declared = asset_properties.read(asset)
    setup, resolved = setups.resolve(setups.parse(setup_name), declared)
    for line in resolved:
        print(f"[{tag}] setup {setup_name}: {line}")
    fps_, substeps_, iterations_, why = setups.stepping_of(setup, declared["recipe"], fps,
                                                           stepping.SUBSTEPS, stepping.ITERATIONS)
    if fps_ != fps:
        raise SystemExit(f"the {setup['stepping']} stepping is at {fps_:g} fps and the run at {fps:g}")
    if substeps or iterations is not None:
        if setup["stepping"] != "canon":
            raise SystemExit(f"--substeps/--iterations and a '{setup['stepping']}' stepping source: "
                             f"two owners for the step; give one")
        substeps_ = substeps or substeps_
        iterations_ = iterations if iterations is not None else iterations_
        why = "the command line"
    print(f"[{tag}] setup {setup_name}: {substeps_} substeps x {iterations_} iterations at {fps:g} fps -- {why}")
    return setup, substeps_, iterations_


class Stepper:
    """One frame of Newton's substep loop -- the one copy of it.

    Newton's examples record the substep loop as a CUDA graph and replay it (e.g.
    multiphysics/example_softbody_dropping_to_cloth.py); stepped from Python instead, each
    substep's dozens of kernel launches cost more than the kernels on a small asset, and the pace a
    run reports would be this runner's, not the engine's. What is recorded is two substeps -- the
    state buffers are back where they started after an even number of swaps -- replayed
    substeps/2 times a frame. A whole frame recorded as one graph died with CUDA error 700
    (illegal memory access) at 128 substeps x 80 iterations on both polybags, where the eager loop
    and two-substep graphs run; the cause inside Warp/Newton is not isolated, and the two-substep
    graph is as fast (apple, 32x10: 36.1 vs 36.8 ms/frame).

    `before_substep(state)` is for a fixture driven from the host every substep (a press plate);
    a graph would freeze its value, so with one the loop is stepped from Python, and says so.
    SolverVBD's collision BVH is rebuilt every frame, before the substeps, as Newton's grasping
    example does: measured, without it a plate held 5 mm inside an asset while only 553 of 3074
    particles were ever in contact, because the tree still described the start.
    """
    RECORDED = 2

    def __init__(self, model, solver, pipeline, substeps, fps, tag, before_substep=None):
        self.solver, self.pipeline, self.substeps = solver, pipeline, substeps
        self.dt = 1.0 / (fps * substeps)
        self.state_0, self.state_1, self.control = model.state(), model.state(), model.control()
        self.contacts = pipeline.contacts()
        self.before_substep = before_substep
        self.graph = None
        if before_substep is not None:
            print(f"[{tag}] substep loop stepped from Python: a fixture is set from the host every substep")
        elif not wp.get_device().is_cuda:
            print(f"[{tag}] substep loop stepped from Python: not a CUDA device")
        elif substeps % self.RECORDED:
            print(f"[{tag}] substep loop stepped from Python: {substeps} substeps is odd, so the state "
                  f"buffers end a frame swapped and a replay would read the wrong one")
        else:
            with wp.ScopedCapture() as capture:
                self._substeps(self.RECORDED)
            self.graph = capture.graph
            print(f"[{tag}] substeps recorded as a CUDA graph of {self.RECORDED}, "
                  f"replayed {substeps // self.RECORDED}x per frame")

    def _substeps(self, n):
        for _ in range(n):
            self.state_0.clear_forces()
            if self.before_substep is not None:
                self.before_substep(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def frame(self):
        self.solver.rebuild_bvh(self.state_0)
        if self.graph is not None:
            for _ in range(self.substeps // self.RECORDED):
                wp.capture_launch(self.graph)
        else:
            self._substeps(self.substeps)


def solver_elements(model, kinds=None):
    """What each of this asset's bodies is made of, in the order the asset declares them:
    tetrahedra for a volume, triangles for a surface -- the arrays a recording binds each render
    mesh to. A loaded polybag is a film of triangles around a filling of tetrahedra, both indexed
    off the one particle array. `kinds` is what `usd_deformable.bodies` said; without it this
    answers for a single body."""
    tets = model.tet_indices.numpy() if model.tet_count else None
    tris = model.tri_indices.numpy() if model.tri_count else None
    if kinds is None:
        if tets is not None:
            return [tets]
        if tris is not None:
            return [tris]
        raise SystemExit("the solver built particles but no elements; there is no surface to draw")
    have = {"volume": tets, "surface": tris}
    if len(set(kinds)) != len(kinds):
        raise SystemExit(f"this asset declares two bodies of the same kind ({', '.join(kinds)}), "
                         f"and one element array cannot be split between them by kind alone")
    out = []
    for kind in kinds:
        if have.get(kind) is None:
            raise SystemExit(f"the asset declares a {kind} body but the solver built no "
                             f"{'tetrahedra' if kind == 'volume' else 'triangles'} for it")
        out.append(have[kind])
    return out
