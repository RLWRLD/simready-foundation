"""The animated USD every run writes, whichever engine produced the motion.

One shape of artefact for all four runners. Each frame the recording is handed the simulation
mesh's nodes -- particles from Newton, tensor-view nodes from PhysX -- and it writes two things:

  * `/root/sim`, the tetrahedral surface the solver actually moved. This is the physics, with
    nothing between it and the viewer;
  * `/root/visual`, the asset's own textured render mesh, carried along by the tetrahedra it sits
    in. This is what the thing looks like.

They are two views of one run, written from the same numbers in the same file, so `render_usd.py`
can photograph either by hiding the other and the two videos line up frame for frame.

Nothing here knows which engine is calling. That is the point: a difference between two videos
has to be a difference between two solvers, not between two recorders.
"""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import skinning

SIM = "/root/sim"
VISUAL = "/root/visual"
GROUND = "/root/ground"
PLATE = "/root/plate"
SIM_COLOUR = (0.92, 0.78, 0.25)


class Recording:
    """Write one run to `path`, with both meshes animated.

    `asset` is the source USD; referencing it is what brings the render mesh's UVs, material and
    textures along, which a mesh copied point by point would lose.
    `sim_prim_path` is the prim in that asset the solver simulates -- it is hidden in the
    recording, because its points are the rest pose and it would sit inside the moving one.
    """

    def __init__(self, path, fps, frames, node_points, tet_indices, asset=None,
                 sim_prim_path=None, ground_half=2.0, plate=None):
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/root").GetPrim())
        self.stage.SetTimeCodesPerSecond(fps)
        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(max(0, frames - 1))
        self.tet_indices = np.asarray(tet_indices, dtype=np.int64)

        faces = skinning.surface_faces(self.tet_indices)
        self.sim = UsdGeom.Mesh.Define(self.stage, SIM)
        self.sim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        self.sim.CreateFaceVertexIndicesAttr(Vt.IntArray([i for face in faces for i in face]))
        self.sim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*SIM_COLOUR)]))
        self.sim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        self.visual = self.bound = None
        if asset:
            self._bind_visual(asset, sim_prim_path, node_points)

        plane = UsdGeom.Mesh.Define(self.stage, GROUND)
        h = ground_half
        plane.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-h, -h, 0.0), Gf.Vec3f(h, -h, 0.0),
                                              Gf.Vec3f(h, h, 0.0), Gf.Vec3f(-h, h, 0.0)]))
        plane.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        plane.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))

        self.plate = None
        if plate:
            half_x, half_y, half_z = plate
            self.plate = UsdGeom.Cube.Define(self.stage, PLATE)
            self.plate.CreateSizeAttr(2.0)
            UsdGeom.XformCommonAPI(self.plate).SetScale(Gf.Vec3f(half_x, half_y, half_z))
            self.plate.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.25, 0.45, 0.85)]))
            self.plate_api = UsdGeom.XformCommonAPI(self.plate)

    def _bind_visual(self, asset, sim_prim_path, _unused):
        source = self.stage.DefinePrim(VISUAL)
        source.GetReferences().AddReference(str(asset))
        # The asset's own simulated prim comes along with the reference and would sit inside the
        # moving mesh at its rest pose. The recording already draws that geometry, moving, as
        # /root/sim.
        if sim_prim_path:
            simulated = self.stage.GetPrimAtPath(f"{VISUAL}/{Sdf.Path(sim_prim_path).name}")
            if simulated:
                UsdGeom.Imageable(simulated).MakeInvisible()

        # The render mesh is the largest mesh under the reference that is not the simulated one.
        candidates = [p for p in Usd.PrimRange(source) if p.IsA(UsdGeom.Mesh) and not p.IsA(UsdGeom.TetMesh)]
        if not candidates:
            print("[recording] the asset has no separate render mesh; only the simulated surface is drawn")
            self.stage.RemovePrim(source.GetPath())
            return
        render_prim = max(candidates, key=lambda p: len(UsdGeom.PointBased(p).GetPointsAttr().Get() or []))
        rest = np.asarray(UsdGeom.PointBased(render_prim).GetPointsAttr().Get(), dtype=np.float64)

        # Both meshes are bound in the pose the asset was authored in, which is the one frame they
        # are known to agree in. Binding the render mesh against nodes the runner has already
        # moved -- lifted to a drop height, or set down on the floor -- puts every vertex outside
        # the tetrahedral hull, and they all clamp onto its nearest faces: the banana came out as
        # a flat yellow smear that no solver produced.
        tets = [q for q in Usd.PrimRange(source) if q.IsA(UsdGeom.TetMesh)]
        if not tets:
            print("[recording] the asset ships no tetrahedral mesh to bind against; "
                  "only the simulated surface is drawn")
            self.stage.RemovePrim(source.GetPath())
            return
        authored = np.asarray(UsdGeom.PointBased(tets[0]).GetPointsAttr().Get(), dtype=np.float64)
        if len(authored) <= int(self.tet_indices.max()):
            raise SystemExit(f"[recording] {tets[0].GetPath()} has {len(authored)} points but the "
                             f"solver indexes {int(self.tet_indices.max()) + 1}: these are not the "
                             f"same mesh, and binding them would invent a shape")
        chosen, weights, outside = skinning.bind(rest, authored, self.tet_indices)
        self.visual = UsdGeom.PointBased(render_prim)
        self.bound = (chosen, weights)
        print(f"[recording] bound {len(rest)} render vertices to {len(self.tet_indices)} tetrahedra "
              f"({outside} outside the hull, carried by their nearest)")

    def frame(self, index, node_points, plate_z=None):
        nodes = np.asarray(node_points, dtype=np.float64)
        time = Usd.TimeCode(index)
        self.sim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in nodes]), time)
        if self.visual is not None:
            moved = skinning.deform(self.bound[0], self.bound[1], self.tet_indices, nodes)
            self.visual.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in moved]), time)
        if self.plate is not None and plate_z is not None:
            self.plate_api.SetTranslate(Gf.Vec3d(0.0, 0.0, float(plate_z)), time)

    def close(self):
        self.stage.GetRootLayer().Save()
