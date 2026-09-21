"""The animated USD every run writes, whichever engine produced the motion.

One shape of artefact for all four runners. Each frame the recording is handed the simulation
mesh's nodes -- particles from Newton, tensor-view nodes from PhysX -- and it writes two things:

  * `/root/collision`, the surface the solver actually moved -- the geometry the engine collides
    with. This is the physics, with nothing between it and the viewer;
  * `/root/visual`, the asset's own textured render mesh, carried along by the tetrahedra it sits
    in. This is what the thing looks like.

They are two views of one run, written from the same numbers in the same file, so `render_usd.py`
can photograph either by hiding the other and the two videos line up frame for frame.

Nothing here knows which engine is calling. That is the point: a difference between two videos
has to be a difference between two solvers, not between two recorders.
"""
import pathlib
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import skinning
import usd_deformable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from asset_checks.kit import reading  # noqa: E402  -- one owner for reading a stage's physics
from asset_checks.kit.reading import copy_gprim  # noqa: E402  -- the one way geometry is copied out of a stage

COLLISION = "/root/collision"
VISUAL = "/root/visual"
GROUND = "/root/ground"
PLATE = "/root/plate"
FIXTURES = "/root/fixtures"
COLLISION_COLOUR = (0.92, 0.78, 0.25)
PLATE_OPACITY = 0.18


def _ground(stage, half):
    """The recording's own floor, a marker of where z = 0 is; the renderer hides it and builds the
    room in its place."""
    plane = UsdGeom.Mesh.Define(stage, GROUND)
    h = float(half)
    plane.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-h, -h, 0.0), Gf.Vec3f(h, -h, 0.0),
                                          Gf.Vec3f(h, h, 0.0), Gf.Vec3f(-h, h, 0.0)]))
    plane.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    plane.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    return plane


def _unique(stage, path):
    candidate, n = path, 1
    while stage.GetPrimAtPath(candidate):
        n += 1
        candidate = f"{path}_{n}"
    return candidate


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
        self.collision = UsdGeom.Mesh.Define(self.stage, COLLISION)
        self.collision.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        self.collision.CreateFaceVertexIndicesAttr(Vt.IntArray([i for face in faces for i in face]))
        self.collision.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*COLLISION_COLOUR)]))
        self.collision.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        self.visual = self.binding = None
        # The asset's own rest pose is what a binding is made in, and it is read back from the
        # reference rather than taken from the runner, which has usually moved the nodes already.
        self.nodes_rest = np.asarray(node_points, dtype=np.float64)
        if asset:
            self._bind_visual(asset, sim_prim_path, None)

        _ground(self.stage, ground_half)

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
        # A typed Xform, not a bare prim: USD's visibility computation walks ancestors through
        # UsdGeomImageable, and skips one that has no type -- an `invisible` authored there was
        # honoured by one renderer and not by another.
        source = UsdGeom.Xform.Define(self.stage, VISUAL).GetPrim()
        source.GetReferences().AddReference(str(asset))
        # The asset's own simulated prim comes along with the reference and would sit inside the
        # moving mesh at its rest pose. The recording already draws that geometry, moving, as
        # /root/collision.
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
        self._write_points(self.collision, nodes, time)
        if self.visual is not None:
            moved = nodes if self.binding is None else skinning.deform(self.binding, self.elements, nodes)
            self._write_points(self.visual, moved, time)
        if self.plate is not None and plate_z is not None:
            where = Gf.Vec3d(self.plate_centre[0], self.plate_centre[1], float(plate_z))
            self.plate_api.SetTranslate(where, time)
            self.plate_edges_api.SetTranslate(where, time)

    def close(self):
        self.stage.GetRootLayer().Save()


class RigidRecording:
    """One rigid run, drawn again from the poses the engine reported.

    Under /root/visual the asset is referenced as it is, textures and all; its root takes the
    recorded pose of the test's asset root (a test may load it lifted), and each rigid body's prim
    takes its recorded world matrix, expressed in its parent's frame. Under /root/collision each
    body's declared colliders -- the gprims carrying CollisionAPI, copied in the body's frame and
    flat-coloured -- ride the same matrices: what the USD says the engine collides with, where the
    engine put it. Under /root/fixtures the test's own furniture (a slope, a gripper's pads) comes
    in from the layer the run saved, each piece placed by its own recorded matrix. Fixtures show in
    both views; only the asset changes clothes.
    """

    def __init__(self, path, fps, asset, result, fixtures_layer=None, ground_half=2.0):
        traj = result["trajectory"]
        self.times = np.asarray(traj["t"], dtype=np.float64)
        self.poses = {p: np.asarray(v, dtype=np.float64) for p, v in traj["pose"].items()}
        # A series that started late -- a gripper built after the asset settled -- is indexed from
        # the step it started at, and its prim is invisible before then.
        self.pose_from = {p: int(traj.get("pose_from", {}).get(p, 0)) for p in self.poses}
        self.asset_prim = result["asset_prim"]
        self.bodies = list(result.get("rigid_bodies") or [])
        self.fixtures = dict((result.get("fixtures") or {}).get("prims") or {})
        self.fps = fps
        self.frames = int(round(self.times[-1] * fps)) + 1

        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/root").GetPrim())
        self.stage.SetTimeCodesPerSecond(fps)
        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(self.frames - 1)
        self._ops = {}
        self._live = {}

        # The asset, as it is. Typed roots throughout, for the reason `_bind_visual` gives.
        visual = UsdGeom.Xform.Define(self.stage, VISUAL).GetPrim()
        visual.GetReferences().AddReference(str(asset))
        UsdGeom.Xform.Define(self.stage, COLLISION)
        self.visual_bodies = {}
        for body in self.bodies:
            rel = body[len(self.asset_prim) + 1:]
            prim = self.stage.GetPrimAtPath(f"{VISUAL}/{rel}")
            if not prim or not prim.IsValid():
                raise SystemExit(f"[recording] {body} is not under the referenced asset as {VISUAL}/{rel}")
            self.visual_bodies[body] = prim

        # Its declared colliders, copied in each body's frame.
        asset_stage = Usd.Stage.Open(str(asset))
        asset_root = asset_stage.GetDefaultPrim().GetPath()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.collision_bodies = {}
        for body in self.bodies:
            rel = body[len(self.asset_prim) + 1:]
            body_prim = asset_stage.GetPrimAtPath(asset_root.AppendPath(rel))
            if not body_prim or not body_prim.IsValid():
                raise SystemExit(f"[recording] {body} is {rel} under {asset_root} in {asset}, which does not exist")
            holder = UsdGeom.Xform.Define(self.stage, _unique(self.stage, f"{COLLISION}/{Sdf.Path(body).name}"))
            self.collision_bodies[body] = holder.GetPrim()
            copied = 0
            for gprim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
                if not (gprim.IsA(UsdGeom.Gprim) and gprim.HasAPI(UsdPhysics.CollisionAPI)):
                    continue
                if gprim != body_prim and str(gprim.GetPath()) in {asset_root.AppendPath(b[len(self.asset_prim) + 1:]) for b in self.bodies}:
                    continue  # a nested body owns its own
                copy = self._collider(gprim, holder.GetPath())
                relative, _ = cache.ComputeRelativeTransform(gprim, body_prim)
                UsdGeom.Xformable(copy).AddTransformOp().Set(relative)
                copied += 1
            if not copied:
                raise SystemExit(f"[recording] {body} declares no collider (no CollisionAPI gprim under it)")

        # The test's furniture.
        self.fixture_prims = {}
        if self.fixtures:
            if not fixtures_layer:
                raise SystemExit("[recording] the run recorded fixtures but no fixtures layer was given")
            root = UsdGeom.Xform.Define(self.stage, FIXTURES).GetPrim()
            root.GetReferences().AddReference(str(fixtures_layer))
            for live, copy_path in self.fixtures.items():
                prim = self.stage.GetPrimAtPath(f"{FIXTURES}/{Sdf.Path(copy_path).name}")
                if not prim or not prim.IsValid():
                    raise SystemExit(f"[recording] fixture {live} was saved as {copy_path} but is not in {fixtures_layer}")
                self.fixture_prims[live] = prim
        self._live = {prim.GetPath(): live for live, prim in self.fixture_prims.items()}
        _ground(self.stage, ground_half)

    def _collider(self, gprim, under):
        """The shape the engine collides with, as its own flat-coloured mesh.

        Not the render mesh with a colour on it: these assets declare no approximation and apply
        no `UsdPhysics.MeshCollisionAPI`, so PhysX cannot use their triangle meshes on a dynamic
        body and cooks a convex hull instead -- it logs that fallback as an error, and drawing the
        triangle mesh would show a shape nothing collides with. `reading.collider_approximation`
        decides what the engine uses and `reading.collider_shape` builds it; where the shape is
        more than geometry can say (a decomposition, an SDF) the mesh is kept and the run says so.
        """
        approximation, why = reading.collider_approximation(gprim)
        points = UsdGeom.PointBased(gprim).GetPointsAttr().Get() if gprim.IsA(UsdGeom.PointBased) else None
        shape = reading.collider_shape(points, approximation) if points else None
        path = _unique(self.stage, f"{under}/{gprim.GetName()}")
        if shape is None:
            copy = copy_gprim(gprim, self.stage, path)
            note = f"as authored ({approximation}: {why})"
        else:
            hull, faces = shape
            mesh = UsdGeom.Mesh.Define(self.stage, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in hull]))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
            copy = mesh.GetPrim()
            note = f"{approximation} of {len(points)} points -> {len(hull)} ({why})"
        g = UsdGeom.Gprim(copy)
        g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*COLLISION_COLOUR)]))
        if copy.IsA(UsdGeom.Mesh):
            UsdGeom.Mesh(copy).CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        print(f"[recording] collider {gprim.GetName()}: {note}")
        return copy

    def _pose_at(self, path, seconds):
        """The recorded matrix nearest `seconds`, or None before the series began."""
        i = int(np.argmin(np.abs(self.times - seconds))) - self.pose_from[path]
        if i < 0:
            return None
        return self.poses[path][min(i, len(self.poses[path]) - 1)]

    def _place(self, prim, flat16, tc):
        """Give `prim` the recorded world matrix at `tc`, as a transform in its parent's frame;
        with no matrix yet, hide it at `tc`."""
        key = prim.GetPath()
        visibility = UsdGeom.Imageable(prim).GetVisibilityAttr() or UsdGeom.Imageable(prim).CreateVisibilityAttr()
        if flat16 is None:
            visibility.Set(UsdGeom.Tokens.invisible, tc)
            return
        if key not in self._ops:
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            self._ops[key] = xf.AddTransformOp()
            if self.pose_from.get(self._live.get(key), 0) > 0:
                visibility.Set(UsdGeom.Tokens.inherited, tc)   # it has just appeared
        parent_world = UsdGeom.XformCache(tc).GetLocalToWorldTransform(prim.GetParent())
        world = Gf.Matrix4d(*[float(v) for v in flat16])
        self._ops[key].Set(world * parent_world.GetInverse(), tc)

    def write(self):
        visual_root = self.stage.GetPrimAtPath(VISUAL)
        ordered = sorted(self.bodies, key=lambda b: b.count("/"))  # parents before children
        for frame in range(self.frames):
            seconds, tc = frame / self.fps, Usd.TimeCode(frame)
            self._place(visual_root, self._pose_at(self.asset_prim, seconds), tc)
            for body in ordered:
                pose = self._pose_at(body, seconds)
                self._place(self.visual_bodies[body], pose, tc)
                self._place(self.collision_bodies[body], pose, tc)
            for live, prim in self.fixture_prims.items():
                self._place(prim, self._pose_at(live, seconds), tc)

    def close(self):
        self.stage.GetRootLayer().Save()


def main():
    """`python recording.py --rigid <result.json> <asset> <out.usda> [--fps 60]`: the animated USD
    of a rigid cell, from the poses its run recorded."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("--rigid", nargs=3, metavar=("RESULT_JSON", "ASSET", "OUT_USDA"), required=True)
    ap.add_argument("--fps", type=int, default=60)
    a = ap.parse_args()
    result_json, asset, out = (pathlib.Path(x) for x in a.rigid)
    result = json.loads(result_json.read_text())
    fixtures = result_json.parent / result["fixtures"]["layer"] if result.get("fixtures") else None
    rec = RigidRecording(out, a.fps, asset.resolve(), result, fixtures)
    rec.write()
    rec.close()
    print(f"[recording] {out}: {rec.frames} frames at {a.fps} fps, {len(rec.bodies)} body(ies), "
          f"{len(rec.fixture_prims)} fixture(s)")


if __name__ == "__main__":
    main()
