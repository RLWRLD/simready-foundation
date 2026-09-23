"""Inside Kit: re-author a Newton-flavour deformable USD so PhysX simulates the same thing.

    ./isaac-run isaac610 native/to_physx.py <source.usda> <out.usda>

Newton reads the AOUSD proposal's public schema names; PhysX reads the same physics under an
`OmniPhysics` prefix and will not translate the other family. So the asset has to be written
twice -- but it has to be the *same asset* twice, or the comparison measures the conversion
instead of the engines.

Two things decide that, and both were wrong before:

  * **The same elements.** `omni.physx`'s auto-hierarchy commands cook their own tetrahedra from
    the render mesh, so PhysX would solve a different discretisation than Newton -- when it
    cooked at all ("Failed to cook PxDeformableVolumeMesh" on this banana). An asset that ships
    a TetMesh already has the elements; `deformableUtils.set_physics_volume_deformable_body`
    takes them as they are. Auto-cooking is kept only for a surface deformable, which has no
    tetrahedra to hand over.
  * **The same material.** With nothing bound, PhysX does not fail -- it quietly runs
    `youngs_modulus = 5e5`, a tenth of what this asset declares, and looks like a softer engine.
    The source's numbers are carried across and the conversion refuses to write an asset that
    would fall back.
"""
import pathlib
import sys

import isaacsim
from isaacsim import SimulationApp

EXPERIENCE = str(pathlib.Path(isaacsim.__file__).parent / "apps" / "isaacsim.exp.full.kit")
app = SimulationApp({"headless": True}, experience=EXPERIENCE)

import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import asset_properties  # noqa: E402
import setups  # noqa: E402
import usd_deformable  # noqa: E402  (the schema names, and PhysX's spelling of them)
from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

SOURCE, OUT = sys.argv[1], sys.argv[2]
VOLUME_SIM, SURFACE_SIM = usd_deformable.VOLUME_SIM, usd_deformable.SURFACE_SIM
# What each kind of deformable material is made of. A volume deformable is described by a
# modulus and a Poisson ratio; a surface one by three stiffnesses and a thickness. PhysX takes
# the surface quantities one-for-one under an `omniphysics:surface*` prefix -- and, unlike
# Newton's isotropic membrane, it can actually honour shearStiffness.
SHARED_FIELDS = ("density", "staticFriction", "dynamicFriction")
VOLUME_FIELDS = ("youngsModulus", "poissonsRatio")
SURFACE_FIELDS = ("stretchStiffness", "shearStiffness", "bendStiffness", "thickness")


def raw_schemas(prim):
    listop = prim.GetMetadata("apiSchemas")
    return list(listop.GetAddedOrExplicitItems()) if listop else []


omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/Asset")
stage.SetDefaultPrim(root.GetPrim())
source_root = stage.DefinePrim("/Asset/Source")
source_root.GetReferences().AddReference(SOURCE)
for _ in range(20):
    app.update()

sim_prim = render_mesh = source_material = None
for prim in Usd.PrimRange(source_root, Usd.TraverseInstanceProxies()):
    schemas = raw_schemas(prim)
    if sim_prim is None and any(s in schemas for s in (VOLUME_SIM, SURFACE_SIM)):
        sim_prim = prim
    if render_mesh is None and prim.IsA(UsdGeom.Mesh) and not any(s in schemas for s in (VOLUME_SIM, SURFACE_SIM)):
        render_mesh = prim
    if source_material is None and prim.IsA(UsdShade.Material) and any("DeformableMaterial" in s for s in schemas):
        source_material = prim
if sim_prim is None:
    raise SystemExit(f"[to-physx] {SOURCE} declares no {VOLUME_SIM} or {SURFACE_SIM} geometry")
if source_material is None:
    raise SystemExit(f"[to-physx] {SOURCE} binds no deformable material; refusing to write an asset that "
                     f"would run on PhysX's default of 5e5 Pa while Newton runs the declared one")

volume = VOLUME_SIM in raw_schemas(sim_prim)
wanted = SHARED_FIELDS + (VOLUME_FIELDS if volume else SURFACE_FIELDS)
values = {}
for name in wanted:
    attr = source_material.GetAttribute(f"physics:{name}")
    if attr and attr.HasAuthoredValue():
        values[name] = float(attr.Get())
required = "youngsModulus" if volume else "stretchStiffness"
if required not in values:
    raise SystemExit(f"[to-physx] {source_material.GetPath()} authors no physics:{required}, which is "
                     f"what a {'volume' if volume else 'surface'} deformable is made of; refusing to "
                     f"guess the stiffness of an asset")

body_path = Sdf.Path("/Asset/Body")
if volume and sim_prim.IsA(UsdGeom.TetMesh):
    # Hand PhysX the asset's own tetrahedra rather than letting it cook new ones.
    source_tets = UsdGeom.TetMesh(sim_prim)
    body = UsdGeom.TetMesh.Define(stage, body_path)
    body.CreatePointsAttr(source_tets.GetPointsAttr().Get())
    body.CreateTetVertexIndicesAttr(source_tets.GetTetVertexIndicesAttr().Get())
    extent = UsdGeom.PointBased(sim_prim).GetExtentAttr().Get()
    if extent:
        body.CreateExtentAttr(extent)
    for _ in range(5):
        app.update()
    if not deformableUtils.set_physics_volume_deformable_body(stage, body_path):
        raise SystemExit(f"[to-physx] set_physics_volume_deformable_body failed on {body_path}")
    points = body.GetPointsAttr().Get()
    tets = body.GetTetVertexIndicesAttr().Get()
    print(f"[to-physx] gave PhysX the asset's own mesh: {len(points)} points, {len(tets)} tets")
else:
    kind = "volume" if volume else "surface"
    cooking_src = render_mesh if (volume and render_mesh is not None) else sim_prim
    print(f"[to-physx] {kind} deformable with no tetrahedra to hand over; cooking from {cooking_src.GetPath()}")
    UsdGeom.Xform.Define(stage, body_path)
    for _ in range(5):
        app.update()
    command = ("CreateAutoVolumeDeformableHierarchyCommand" if volume
               else "CreateAutoSurfaceDeformableHierarchyCommand")
    kwargs = {"root_prim_path": str(body_path), "cooking_src_mesh_path": str(cooking_src.GetPath()),
              "cooking_src_simplification_enabled": False}
    if volume:
        kwargs.update(simulation_tetmesh_path=f"{body_path}/simulation_tetmesh",
                      collision_tetmesh_path=f"{body_path}/collision_tetmesh",
                      simulation_hex_mesh_enabled=False)
    else:
        kwargs["simulation_mesh_path"] = f"{body_path}/simulation_mesh"
    print(f"[to-physx] {command} -> {omni.kit.commands.execute(command, **kwargs)}")
for _ in range(30):
    app.update()

material_path = "/Asset/PhysicsMaterial"
# Friction the asset did not declare must not come from this helper's own defaults (0.5 static,
# 0.25 dynamic): Newton would run the same silent asset at 0.5, and the two engines would be
# sliding on different floors while every other number matched.
friction, _friction_why = asset_properties.friction(asset_properties.read(SOURCE))
shared = {"density": values.get("density"), "static_friction": values.get("staticFriction", friction),
          "dynamic_friction": friction}
if volume:
    made = deformableUtils.add_deformable_material(
        stage, material_path, youngs_modulus=values["youngsModulus"],
        poissons_ratio=values.get("poissonsRatio"), **shared)
else:
    # PhysX takes these as stiffnesses (`omniphysics:surface*Stiffness`: "override for stretching
    # stiffness; by default derived from youngsModulus and thickness"); Newton's importer reads the
    # same authored numbers as moduli times the thickness. PhysX runs the default setup only, so the
    # copy carries that setup's reading -- the numbers Newton builds. Carried raw, a 3 mm film ran
    # 333x stiffer on PhysX than on Newton.
    reading = setups.parse(setups.DEFAULT)["surface"]
    stiff = usd_deformable.surface_stiffnesses(values, reading)
    print(f"[to-physx] surface stiffnesses read as {reading} ({setups.DEFAULT}): {stiff}")
    made = deformableUtils.add_surface_deformable_material(
        stage, material_path, surface_thickness=values.get("thickness"),
        surface_stretch_stiffness=stiff.get("stretchStiffness"),
        surface_shear_stiffness=stiff.get("shearStiffness"),
        surface_bend_stiffness=stiff.get("bendStiffness"), **shared)
if not made:
    raise SystemExit(f"[to-physx] could not author the {'volume' if volume else 'surface'} "
                     f"material at {material_path}")
material = UsdShade.Material.Get(stage, material_path)
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(body_path)).Bind(
    material, UsdShade.Tokens.weakerThanDescendants, "physics")
print(f"[to-physx] carried {source_material.GetPath()} -> {material_path} {values}")

# The source layer still carries the Newton-flavour schemas and a PhysicsCollisionAPI, which
# PhysX would read as a second, static collider sitting inside the body. Nothing should simulate
# it, so it is deactivated rather than left to fight the body we just authored.
source_root.SetActive(False)
print(f"[to-physx] deactivated {source_root.GetPath()} so only the PhysX body is simulated")

for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    applied = [s for s in prim.GetAppliedSchemas() if "Deformable" in s or "Collision" in s]
    if applied:
        print(f"[to-physx] {prim.GetPath()} {prim.GetTypeName()} {applied}")

stage.Export(OUT)
print(f"[to-physx] wrote {OUT}")
app.close()
