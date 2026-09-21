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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

import skinning
import usd_deformable

SIM = "/root/sim"
VISUAL = "/root/visual"
GROUND = "/root/ground"
PLATE = "/root/plate"
SIM_COLOUR = (0.92, 0.78, 0.25)
PLATE_OPACITY = 0.18


class Recording:
    """Write one run to `path`, with both meshes animated.

    `asset` is the source USD; referencing it is what brings the render mesh's UVs, material and
    textures along, which a mesh copied point by point would lose.
    `sim_prim_path` is the prim in that asset the solver simulates -- it is hidden in the
    recording, because its points are the rest pose and it would sit inside the moving one.
    """

    def __init__(self, path, fps, frames, node_points, elements, asset=None,
                 sim_prim_path=None, ground_half=2.0, plate=None, plate_centre=(0.0, 0.0)):
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/root").GetPrim())
        self.stage.SetTimeCodesPerSecond(fps)
        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(max(0, frames - 1))
        self.elements = np.asarray(elements, dtype=np.int64)

        faces = skinning.surface_faces(self.elements)
        self.sim = UsdGeom.Mesh.Define(self.stage, SIM)
        self.sim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        self.sim.CreateFaceVertexIndicesAttr(Vt.IntArray([i for face in faces for i in face]))
        self.sim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*SIM_COLOUR)]))
        self.sim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        self.visual = self.binding = None
        # The asset's own rest pose is what a binding is made in, and it is read back from the
        # reference rather than taken from the runner, which has usually moved the nodes already.
        self.nodes_rest = np.asarray(node_points, dtype=np.float64)
        if asset:
            self._bind_visual(asset, sim_prim_path, None)

        plane = UsdGeom.Mesh.Define(self.stage, GROUND)
        h = ground_half
        plane.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-h, -h, 0.0), Gf.Vec3f(h, -h, 0.0),
                                              Gf.Vec3f(h, h, 0.0), Gf.Vec3f(-h, h, 0.0)]))
        plane.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        plane.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))

        self.plate = None
        # Where the simulation puts the plate, not the origin. The runner centres it on the
        # asset -- which for a curved banana is not (0, 0) -- and drawing it at the origin put
        # the plate beside the thing it was pressing in every video.
        self.plate_centre = (float(plate_centre[0]), float(plate_centre[1]))
        if plate:
            half_x, half_y, half_z = plate
            self.plate = UsdGeom.Cube.Define(self.stage, PLATE)
            self.plate.CreateSizeAttr(2.0)
            UsdGeom.XformCommonAPI(self.plate).SetScale(Gf.Vec3f(half_x, half_y, half_z))
            # See-through, because a solid plate hides the thing being measured, with its edges
            # drawn so its position is still readable against the asset.
            self.plate.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.30, 0.50, 0.90)]))
            self.plate.CreateDisplayOpacityAttr(Vt.FloatArray([PLATE_OPACITY]))
            # RTX does not read displayOpacity on its own -- a surface is opaque unless a
            # material says otherwise, which is why the plate stayed solid with the opacity
            # attribute already written. A UsdPreviewSurface carries it.
            glass = UsdShade.Material.Define(self.stage, PLATE + "_material")
            shader = UsdShade.Shader.Define(self.stage, PLATE + "_material/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.35, 0.55, 0.95))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(PLATE_OPACITY)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            glass.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(self.plate.GetPrim()).Bind(glass)
            self.plate_edges = UsdGeom.BasisCurves.Define(self.stage, PLATE + "_edges")
            self.plate_edges.CreateTypeAttr(UsdGeom.Tokens.linear)
            self.plate_edges.CreateCurveVertexCountsAttr(Vt.IntArray([2] * 12))
            corners = [(sx, sy, sz) for sx in (-half_x, half_x) for sy in (-half_y, half_y)
                       for sz in (-half_z, half_z)]
            edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                     if sum(abs(corners[a][i] - corners[b][i]) > 1e-12 for i in range(3)) == 1]
            self.plate_edges.CreatePointsAttr(
                Vt.Vec3fArray([Gf.Vec3f(*corners[i]) for edge in edges for i in edge]))
            self.plate_edges.CreateWidthsAttr(Vt.FloatArray([max(half_z * 0.08, 1e-4)] * 24))
            self.plate_edges.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.10, 0.25, 0.60)]))
            self.plate_edges_api = UsdGeom.XformCommonAPI(self.plate_edges)
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

        # Which prim under the reference is the one the solver moves, and which is the one with
        # the textures on it. They are often different -- a soft body is simulated as tetrahedra
        # and drawn as a finer mesh -- and they are sometimes the same prim, which is what a
        # cloth is. Both cases are the same question asked of the asset, not a special case for
        # any one of them.
        # Find the simulated prim inside the reference by what it *is*, not by a path. A path
        # belongs to the file it came from: the PhysX runner knows its asset's body as
        # /Asset/Body while the reference here is the original, where the same geometry is
        # /World/banana. Looking the name up across files found nothing, the binding fell back to
        # the runner's already-moved nodes, and the render mesh came out 71 mm from the thing it
        # was supposed to follow.
        simulated = next((q for q in Usd.PrimRange(source) if usd_deformable.is_simulated(q)), None)
        if simulated is None and sim_prim_path:
            simulated = self.stage.GetPrimAtPath(f"{VISUAL}/{Sdf.Path(sim_prim_path).name}")
            if not (simulated and simulated.IsValid()):
                simulated = None
        drawable = [q for q in Usd.PrimRange(source)
                    if q.IsA(UsdGeom.PointBased) and UsdGeom.PointBased(q).GetPointsAttr().Get()]
        others = [q for q in drawable if not (simulated and q.GetPath() == simulated.GetPath())]
        if not drawable:
            print("[recording] the asset has no drawable geometry; only the simulated surface is shown")
            self.stage.RemovePrim(source.GetPath())
            return

        if others:
            render_prim = max(others, key=lambda q: len(UsdGeom.PointBased(q).GetPointsAttr().Get()))
            if simulated:
                # Its points are the rest pose; the recording already draws that geometry moving.
                UsdGeom.Imageable(simulated).MakeInvisible()
        else:
            # The asset draws what it simulates -- a cloth, typically. Then the visual version is
            # that same mesh with the asset's own material on it, moving directly.
            render_prim = simulated or drawable[0]

        rest = np.asarray(UsdGeom.PointBased(render_prim).GetPointsAttr().Get(), dtype=np.float64)
        self.visual = UsdGeom.PointBased(render_prim)

        # The simulated mesh's *authored* points, read from the reference rather than taken from
        # the runner. The runner has already moved its nodes -- lifted to a drop height, or set
        # down on the floor -- and binding against those puts every render vertex outside the
        # elements, where they all clamp onto the nearest surface and the asset renders as a flat
        # smear no solver produced. The authored pose is the one frame the two meshes agree in.
        #
        # This was fixed once and then reintroduced by a rewrite that kept the comment and
        # dropped the code, so it is asserted below rather than trusted.
        simulated_rest = (np.asarray(UsdGeom.PointBased(simulated).GetPointsAttr().Get(), dtype=np.float64)
                          if simulated else self.nodes_rest)
        if len(rest) == len(simulated_rest) and np.allclose(rest, simulated_rest, atol=1e-9):
            # Same points: nothing to bind, and binding would only add error.
            self.binding = None
            print(f"[recording] the asset draws what it simulates ({len(rest)} points), moved directly")
            return

        # Both meshes are bound in the pose the asset was authored in, the one frame they are
        # known to agree in. Binding against nodes the runner has already moved -- lifted to a
        # drop height, or set down on the floor -- puts every vertex outside the elements and
        # they all clamp to the nearest surface: the banana came out as a flat smear no solver
        # produced.
        if len(simulated_rest) <= int(self.elements.max()):
            raise SystemExit(f"[recording] the asset's simulated mesh has {len(simulated_rest)} "
                             f"points but the solver indexes {int(self.elements.max()) + 1}: these "
                             f"are not the same mesh, and binding them would invent a shape")
        # A binding made in the right pose leaves most vertices inside their element. If nearly
        # all of them land outside, the two meshes were not in the same pose and the result would
        # be a smear -- so it fails here rather than rendering something nobody simulated.
        # Assert the pose directly rather than inferring it from how many vertices landed
        # outside: that count has its own definition and its own bugs, and this is the thing
        # that actually has to be true. A render mesh bound against nodes the runner has already
        # moved produces a flat smear no solver computed.
        span = max(float(np.ptp(simulated_rest, axis=0).max()), 1e-9)
        drift = float(np.abs(rest.mean(axis=0) - simulated_rest.mean(axis=0)).max())
        if drift > 0.05 * span:
            raise SystemExit(f"[recording] the render mesh and the simulated mesh are not in the "
                             f"same pose -- their centres are {drift * 1000:.1f} mm apart on a "
                             f"{span * 1000:.1f} mm asset. Binding them would invent a shape.")
        self.binding = skinning.bind(rest, simulated_rest, self.elements)
        print(f"[recording] bound {len(rest)} render vertices to {len(self.elements)} "
              f"{'tetrahedra' if self.elements.shape[1] == 4 else 'triangles'} "
              f"({self.binding['outside']} outside, carried by their nearest)")

    @staticmethod
    def _write_points(pointbased, points, time):
        """The points at this time, and the extent with them.

        A mesh with animated points and a static authored extent reports the authored box at every
        time: `UsdGeom.BBoxCache` reads the extent when one is there rather than the points, and a
        mesh referenced from the asset brings the asset's. A cloth that fell 49.5 mm read as never
        moving, and the camera framed the recording's floor instead of it.
        """
        points = np.asarray(points, dtype=np.float64)
        pointbased.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]), time)
        lo, hi = points.min(axis=0), points.max(axis=0)
        pointbased.CreateExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]), time)

    def frame(self, index, node_points, plate_z=None):
        nodes = np.asarray(node_points, dtype=np.float64)
        time = Usd.TimeCode(index)
        self._write_points(self.sim, nodes, time)
        if self.visual is not None:
            moved = nodes if self.binding is None else skinning.deform(self.binding, self.elements, nodes)
            self._write_points(self.visual, moved, time)
        if self.plate is not None and plate_z is not None:
            where = Gf.Vec3d(self.plate_centre[0], self.plate_centre[1], float(plate_z))
            self.plate_api.SetTranslate(where, time)
            self.plate_edges_api.SetTranslate(where, time)

    def close(self):
        self.stage.GetRootLayer().Save()
