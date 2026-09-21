# SPDX-License-Identifier: Apache-2.0
"""Stage setup shared by the experiments, and the asset's runtime physics variant.

SimReady assets keep engine-specific physics in runtime variants (PhysX / Newton / MuJoCo), each
Disabled by default and composing a payload under runnables/physics/ when Enabled; the consumer
selects the variant of the engine it runs ("Multiple Physics Solvers" guide). engine-kit's
load_asset references the asset's default prim at ASSET_PRIM without selecting any variant, so
this module selects it, as SimReady_Metadata.Variants.Physics declares it.
"""

ASSET_PRIM = "/World/AssetRoot/Asset"  # where engine-kit's load_asset references the asset's default prim


async def new_stage():
    """A fresh stage prepared the way engine-kit's execute_single_test prepares one for each test."""
    import omni.kit.app
    import omni.usd
    from pxr import UsdGeom

    try:  # reset the viewport camera before the stage goes, as execute_single_test does
        import omni.kit.viewport.utility as viewport_utility

        viewport = viewport_utility.get_active_viewport()
        if viewport:
            viewport.camera_path = "/OmniverseKit_Persp"
    except ImportError:
        pass
    ok, err = await omni.usd.get_context().new_stage_async()
    if not ok:
        raise RuntimeError(f"could not create a new stage: {err}")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if stage.GetPrimAtPath("/OmniKit_Viewport_LightRig"):
        stage.RemovePrim("/OmniKit_Viewport_LightRig")
    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()
    return stage


def select_runtime_variant(stage, asset_path, engine):
    """Select the asset's runtime physics variant for `engine`; report what it composed.

    Returns {"declared": [...runtimes in the asset's metadata...], "selected": {...} or None,
    "payload_schemas_not_composed": {prim: [schemas]}}. The last entry compares the API schemas the
    variant's payload declares with the ones on the composed prim: an explicit apiSchemas list in a
    stronger layer discards a payload's prepended schemas, leaving only its attribute values.
    """
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(asset_path)
    physics = ((dict(layer.customLayerData).get("SimReady_Metadata") or {}).get("Variants") or {}).get("Physics") or {}
    report = {"declared": sorted(physics), "selected": None, "payload_schemas_not_composed": {}}
    entry = next((value for key, value in physics.items() if key.lower() == engine), None)
    if entry is None:
        return report
    asset_root = "/" + layer.defaultPrim
    if entry.get("prim", asset_root) != asset_root:
        raise RuntimeError(f"variant set {entry} is not on the default prim {asset_root}")
    variant_set = stage.GetPrimAtPath(ASSET_PRIM).GetVariantSets().GetVariantSet(entry["variantSetName"])
    option = entry["activateOption"]
    if option not in variant_set.GetVariantNames():
        raise RuntimeError(f"{entry['variantSetName']} has no option {option!r}: {variant_set.GetVariantNames()}")
    variant_set.SetVariantSelection(option)
    report["selected"] = {"variantSet": entry["variantSetName"], "option": option}

    # The selected variant's specs, wherever the asset authors them (its root layer, a sublayer, a
    # reference): the composed prim's stack holds every spec that contributes, variant specs included.
    selection = (entry["variantSetName"], option)
    variant_specs = [spec for spec in stage.GetPrimAtPath(ASSET_PRIM).GetPrimStack()
                     if spec.path.ContainsPrimVariantSelection() and spec.path.GetVariantSelection() == selection]
    if not variant_specs:
        raise RuntimeError(f"selected {selection} but no spec of it contributes to {ASSET_PRIM}")
    payloads = [(spec.layer, payload) for spec in variant_specs for payload in spec.payloadList.GetAddedOrExplicitItems()]
    for spec_layer, payload in payloads:
        payload_layer = Sdf.Layer.FindOrOpen(spec_layer.ComputeAbsolutePath(payload.assetPath))
        specs = [payload_layer.pseudoRoot]
        while specs:
            spec = specs.pop()
            specs.extend(spec.nameChildren)
            if spec == payload_layer.pseudoRoot or not spec.HasInfo("apiSchemas"):
                continue
            declared = set(spec.GetInfo("apiSchemas").GetAddedOrExplicitItems())
            path = ASSET_PRIM + str(spec.path)[len(asset_root):]
            prim = stage.GetPrimAtPath(path)
            op = prim.GetMetadata("apiSchemas") if prim and prim.IsValid() else None
            composed = set(op.GetAddedOrExplicitItems()) if op else set()
            if declared - composed:
                report["payload_schemas_not_composed"][path] = sorted(declared - composed)
    return report


LOOK_PRIM = "/World/AssetChecksLook"  # visual-only additions; nothing under it collides or has mass
FLOOR_PRIM = "/World/Room/Floor"  # the visible floor of NVIDIA's test room (all three tests)


def add_visual_cues(stage, center_xy, tile_m=0.1, half_extent_m=5.0, key_intensity=1000.0):
    """Make the floor readable in the videos. NVIDIA's room is one grey under one uniform dome light,
    so floor, far walls and the horizon blend and nothing casts a shadow. Adds a checkerboard of
    tile_m squares just above the visible floor -- two meshes, light and dark squares, each with a
    matte (roughness 1) preview material, so the key light leaves no glare -- and a distant key
    light 20 degrees off vertical, so a lifted object's shadow falls under it. The grid has no
    collider, so no physics reads it or its materials. Returns what it added, or why nothing."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    floor = stage.GetPrimAtPath(FLOOR_PRIM)
    if not floor.IsValid():
        return {"added": False, "reason": f"no {FLOOR_PRIM}"}
    top = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"]).ComputeWorldBound(floor).ComputeAlignedRange().GetMax()[2]
    n = int(round(2 * half_extent_m / tile_m))
    z, x0, y0 = top + 0.0002, center_xy[0] - half_extent_m, center_xy[1] - half_extent_m
    UsdGeom.Xform.Define(stage, LOOK_PRIM)
    for name, parity, grey in (("Light", 0, 0.32), ("Dark", 1, 0.18)):
        points, indices = [], []
        for j in range(n):
            for i in range(n):
                if (i + j) % 2 != parity:
                    continue
                base = len(points)
                points += [Gf.Vec3f(x0 + i * tile_m, y0 + j * tile_m, z), Gf.Vec3f(x0 + (i + 1) * tile_m, y0 + j * tile_m, z),
                           Gf.Vec3f(x0 + (i + 1) * tile_m, y0 + (j + 1) * tile_m, z), Gf.Vec3f(x0 + i * tile_m, y0 + (j + 1) * tile_m, z)]
                indices += [base, base + 1, base + 2, base + 3]
        mesh = UsdGeom.Mesh.Define(stage, f"{LOOK_PRIM}/FloorGrid{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (len(indices) // 4)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x0 + n * tile_m, y0 + n * tile_m, z)]))
        material = UsdShade.Material.Define(stage, f"{LOOK_PRIM}/Looks/Floor{name}")
        shader = UsdShade.Shader.Define(stage, f"{LOOK_PRIM}/Looks/Floor{name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(grey, grey, grey))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    key = UsdLux.DistantLight.Define(stage, LOOK_PRIM + "/KeyLight")
    key.CreateIntensityAttr(key_intensity)
    key.CreateAngleAttr(2.0)
    UsdGeom.XformCommonAPI(key).SetRotate(Gf.Vec3f(20.0, 0.0, 30.0))
    return {"added": True, "floor_grid": f"{LOOK_PRIM}/FloorGrid{{Light,Dark}}", "tile_m": tile_m, "extent_m": 2 * half_extent_m,
            "floor_top_z": round(top, 5), "key_light": str(key.GetPath()), "key_intensity": key_intensity}


DEFAULT_NEWTON_SOLVER = "mujoco"  # what Isaac's Newton stage builds when no scene schema selects one

# What an asset declares itself to be, by the API schemas on its geometry. A deformable is not a
# rigid body with soft settings: it is particles, and only a solver that integrates particles can
# run it. The sim schemas are the AOUSD proposal's public names, which Newton's importer reads.
DEFORMABLE_SIM_API = ("PhysicsSurfaceDeformableSimAPI", "PhysicsVolumeDeformableSimAPI",
                      "PhysicsCurvesDeformableSimAPI")
SOLVER_SIMULATES = {  # measured, not assumed: MuJoCo refuses a stage whose bodies are particles
    "physx": {"rigid", "deformable"},   # one engine, both kinds, no solver to choose
    "mujoco": {"rigid"},                # MuJoCo-Warp is a rigid-body engine
    "xpbd": {"rigid", "deformable"},    # XPBD: rigid and soft bodies
    "vbd": {"rigid", "deformable"},     # VBD for particles, AVBD for rigid bodies, and the two coupled
}


def asset_kinds(stage, root_path):
    """{"rigid"} / {"deformable"} / both / empty -- what the asset's own schemas declare it to be."""
    from pxr import Usd, UsdPhysics

    from asset_checks.kit.reading import raw_api_schemas

    kinds = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            kinds.add("rigid")
        applied = set(raw_api_schemas(prim))
        if applied.intersection(DEFORMABLE_SIM_API):
            kinds.add("deformable")
    return kinds


def check_asset_fits_solver(stage, root_path, solver):
    """Refuse a combination the solver cannot simulate, before the test spends a run on it."""
    kinds = asset_kinds(stage, root_path)
    unsupported = kinds - SOLVER_SIMULATES.get(solver, set())
    if unsupported:
        raise RuntimeError(
            f"the asset declares {sorted(kinds)} geometry and the {solver!r} solver simulates "
            f"{sorted(SOLVER_SIMULATES.get(solver, set()))}: {sorted(unsupported)} has no solver here. "
            f"Run it in an environment whose solver takes it.")
    return {"asset_kinds": sorted(kinds), "solver_simulates": sorted(SOLVER_SIMULATES.get(solver, set()))}


def _schema_registered(identifier) -> bool:
    """Whether this Kit registers that schema identifier. A schema's USD identifier is not its Tf
    type name (the deformable proposal ships as OmniUsdPhysicsDeformableSchema* types with
    PhysicsDeformable* identifiers), so ask the registry by identifier."""
    from pxr import Plug, Usd

    registry = Usd.SchemaRegistry()
    return any(registry.GetSchemaTypeName(t) == identifier
               for t in Plug.Registry().GetAllDerivedTypes("UsdAPISchemaBase"))


DEFORMABLE_SUBSTEPS = 5  # what Newton's own deformable examples use; Isaac's default is 1


def _set_substeps(substeps):
    """How many solver steps Isaac takes per frame. It defaults to one, and Newton's own deformable
    examples take five: a particle solver integrates a stiff material, and one step per frame is
    where a large or stiff asset comes apart. Returns what was set."""
    import isaacsim.physics.newton as isaac_newton

    cfg = isaac_newton.acquire_stage().cfg
    was = getattr(cfg, "num_substeps", None)
    cfg.num_substeps = int(substeps)
    return {"num_substeps": int(substeps), "was": was}


def _patch_builder(particle_radius=None):
    """Everything that has to happen while Newton's model is being built, in one wrapper."""
    _register_mujoco_attributes_too()
    _RADIUS["value"] = particle_radius


_RADIUS = {"value": None}


def _register_mujoco_attributes_too():
    """Isaac 6.0.1 hands Newton's USD importer a MuJoCo schema resolver whatever the solver, but
    registers MuJoCo's custom attributes on the builder only when the solver is MuJoCo, so asking
    for any other solver there fails with "MuJoCo custom attributes not registered" and no model is
    built at all. Registering them alongside whatever else is registered costs nothing -- they are
    attribute declarations. Idempotent: registering twice is ignored.

    Isaac 6.0.1 also abandons initialisation when the builder has no rigid body, which drops a
    deformable-only scene outright (6.1.0 counts particles in the same test). That one is not
    patched here: adding a body to get past it made XPBD 1.2.1 fail with an illegal CUDA access, so
    on that Isaac a deformable-only scene is reported as not running rather than forced."""
    import newton

    if getattr(newton.ModelBuilder, "_asset_checks_mjc_attributes", False):
        return  # installed once per Kit process; the wrapper reads _RADIUS each time it runs
    original = newton.ModelBuilder.add_usd

    def add_usd(self, *args, **kwargs):
        try:
            newton.solvers.SolverMuJoCo.register_custom_attributes(self)
        except Exception:  # noqa: BLE001 - already registered, which is what we want
            pass
        out = original(self, *args, **kwargs)
        size_particles_on(self, _RADIUS["value"])
        return out

    newton.ModelBuilder.add_usd = add_usd
    newton.ModelBuilder._asset_checks_mjc_attributes = True


def _select_solver_by_config(solver):
    """Isaac 6.0.1's Newton stage has no schema mapping: `_get_solver` reads `cfg.solver_cfg`, whose
    `solver_type` it switches on. Setting that config before the first play is how a solver is asked
    for there.

    It knows two -- mujoco and xpbd -- and raises for anything else, but the Newton it is pinned to
    ships every solver, so a missing branch is a gap in Isaac's switch rather than a missing
    capability. For such a solver this supplies a config carrying its name and a `_get_solver` that
    builds it from newton.solvers, leaving Isaac's own branches untouched. Returns what was set."""
    import dataclasses

    import newton
    import isaacsim.physics.newton as isaac_newton
    from isaacsim.physics.newton.impl import solver_config
    from isaacsim.physics.newton.impl.newton_stage import NewtonStage

    by_type = {getattr(cls, "__dataclass_fields__", {}).get("solver_type").default: cls
               for cls in vars(solver_config).values()
               if isinstance(cls, type) and "solver_type" in getattr(cls, "__dataclass_fields__", {})}
    stage_handle = isaac_newton.acquire_stage()
    _register_mujoco_attributes_too()
    if solver in by_type:
        stage_handle.cfg.solver_cfg = by_type[solver]()
        return f"cfg.solver_cfg = {by_type[solver].__name__}"

    solver_class = getattr(newton.solvers, f"Solver{solver.upper()}", None) or \
        getattr(newton.solvers, f"Solver{solver.capitalize()}", None)
    if solver_class is None:
        raise RuntimeError(f"this Isaac has no config for {solver!r} and Newton has no solver by that name")
    if not getattr(NewtonStage, "_asset_checks_solver_patched", False):
        original = NewtonStage._get_solver.__func__

        def _get_solver(cls, model, solver_cfg):
            wanted = getattr(solver_cfg, "solver_type", None)
            if wanted in by_type or wanted is None:
                return original(cls, model, solver_cfg)
            built = getattr(newton.solvers, f"Solver{wanted.upper()}", None)
            if built is None:
                return original(cls, model, solver_cfg)
            kwargs = {k: v for k, v in vars(solver_cfg).items() if k != "solver_type"}
            return built(model, **kwargs)

        NewtonStage._get_solver = classmethod(_get_solver)
        NewtonStage._asset_checks_solver_patched = True
    stage_handle.cfg.solver_cfg = dataclasses.make_dataclass(
        f"{solver.upper()}SolverConfigSuppliedByAssetChecks", [("solver_type", str, solver)])()
    return f"cfg.solver_cfg = {solver!r} through asset_checks (this Isaac's _get_solver has no branch for it)"


def _preset_solver_config(solver, settings):
    """Give Isaac the solver's config before it builds one, so settings survive initialisation.

    Isaac replaces `cfg.solver_cfg` during init only when it is not already the right class, so a
    config of that class set now is kept and its values are the ones the solver gets. Without this
    there is nowhere to put a solver setting on Isaac 6.1.0: the scene schema selects the solver and
    a fresh config comes with it."""
    if not settings:
        return None
    import isaacsim.physics.newton as isaac_newton
    from isaacsim.physics.newton.impl import newton_config, solver_config

    by_type = {getattr(cls, "__dataclass_fields__", {}).get("solver_type").default: cls
               for cls in vars(solver_config).values()
               if isinstance(cls, type) and "solver_type" in getattr(cls, "__dataclass_fields__", {})}
    config_class = by_type.get(solver)
    if config_class is None:
        return None
    unknown = set(settings) - set(getattr(config_class, "__dataclass_fields__", {}))
    if unknown:
        raise RuntimeError(f"{config_class.__name__} has no {sorted(unknown)}; it has "
                           f"{sorted(config_class.__dataclass_fields__)}")
    isaac_newton.acquire_stage().cfg.solver_cfg = config_class(**settings)
    return dict(settings)


def select_solver(stage, engine, solver, settings=None, particle_radius=None, substeps=DEFORMABLE_SUBSTEPS):
    """Ask Isaac for `solver` and report how. Isaac 6.1.0 reads a solver's scene API schema off the
    PhysicsScene (`impl/utils.py newton_solver_to_api_schema`) and refuses a stage carrying two of
    them. Under PhysX there is nothing to select. Asking for the Isaac's own default needs nothing
    applied -- Isaac 6.0.1 has no schema mapping at all and always builds SolverMuJoCo -- so it is
    reported as such; asking for anything else on an Isaac that cannot select it is refused here
    rather than silently simulating with another solver. Either way run.py checks the solver that
    actually integrated the scene against the one asked for."""
    from pxr import Usd, UsdPhysics

    from asset_checks.envs import NEWTON_SOLVER_SCENE_API

    fit = check_asset_fits_solver(stage, ASSET_PRIM, solver)
    fit["solver_settings"] = settings or None
    if engine == "newton":
        _patch_builder(particle_radius)  # before anything Isaac builds reads the geometry
        fit["substeps"] = _set_substeps(substeps)
    if engine != "newton":
        return {"requested": solver, "applied": None, "reason": f"{engine} has one solver", **fit}
    try:
        from isaacsim.physics.newton.impl.utils import newton_solver_to_api_schema as mapping
    except ImportError:  # Isaac 6.0.1: the solver comes from the Python config, not from USD
        mapping = None
    schema = NEWTON_SOLVER_SCENE_API[solver]
    if mapping is None:  # Isaac 6.0.1 takes the solver from its Python config, not from USD
        return {"requested": solver, "applied": _select_solver_by_config(solver), **fit,
                "reason": "this Isaac selects a solver from its config, not from a scene schema"}
    if solver == DEFAULT_NEWTON_SOLVER and not _schema_registered(schema):
        return {"requested": solver, "applied": None, **fit,
                "reason": f"this Isaac builds {solver} by default and cannot select from USD"}
    if mapping.get(solver) != schema:
        raise RuntimeError(f"this Isaac cannot select the {solver!r} solver from USD "
                           f"(schema map: {mapping}); run it on an Isaac that maps {schema}")
    if not _schema_registered(schema):
        raise RuntimeError(f"{schema} is not a registered schema in this Isaac, so {solver!r} cannot be asked for")
    scenes = [p for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise RuntimeError("no PhysicsScene on the stage to select a solver on")
    # engine-kit applies MjcSceneAPI when it builds a Newton scene, and Isaac takes a scene with two
    # solver schemas as having none ("Multiple solver APIs detected"), so the other one goes.
    _preset_solver_config(solver, settings)
    replaced = []
    for prim in scenes:
        for other in set(mapping.values()) - {schema}:
            if prim.HasAPI(other):
                prim.RemoveAPI(other)
                replaced.append(f"{prim.GetPath()}:{other}")
        prim.ApplyAPI(schema)
        if not prim.HasAPI(schema):
            raise RuntimeError(f"applied {schema} to {prim.GetPath()} and it did not take")
    return {"requested": solver, "applied": schema, "replaced": replaced,
            "scenes": [str(p.GetPath()) for p in scenes], **fit}


# Newton's defaults are soft_contact_ke 1e3, kd 1e1, mu 0.5, which let a soft body sink through the
# floor, and Isaac sets none of them. Newton's own VBD soft-rigid example uses 1e5 / 1e2 / 0.8 for
# centimetre-scale objects (examples/vbd/example_vbd_soft_rigid_contact.py); those are the floor
# here, and the stiffness is raised from the asset when the asset needs more -- see soft_contact_for.
SOFT_CONTACT_FLOOR = {"soft_contact_ke": 1.0e5, "soft_contact_kd": 1.0e2, "soft_contact_mu": 0.8}
PENETRATION_OF_RADIUS = 0.1  # how far a resting particle may sink into a surface, as a fraction of its radius


def soft_contact_for(particle_mass, radius, gravity=9.81):
    """Contact stiffness that holds this asset up, whatever it weighs.

    A particle resting on a surface sinks until the contact force balances its weight: with a linear
    contact that is m*g/ke, so ke = m*g / (fraction of the radius one is willing to let it sink).
    Fixed numbers cannot do this -- the same 1e5 that suits a 3 g cloth vertex lets a heavy one sink
    straight through -- and the mass comes from the asset's own density and geometry. Damping is
    kept in the same proportion to stiffness as Newton's example uses.
    """
    import numpy as np

    mass = float(np.median(np.asarray(particle_mass)[np.asarray(particle_mass) > 0])) if len(particle_mass) else 0.0
    if mass <= 0.0 or radius <= 0.0:
        return dict(SOFT_CONTACT_FLOOR)
    ke = mass * gravity / (PENETRATION_OF_RADIUS * radius)
    ke = max(ke, SOFT_CONTACT_FLOOR["soft_contact_ke"])
    ratio = SOFT_CONTACT_FLOOR["soft_contact_kd"] / SOFT_CONTACT_FLOOR["soft_contact_ke"]
    return {"soft_contact_ke": ke, "soft_contact_kd": ke * ratio,
            "soft_contact_mu": SOFT_CONTACT_FLOOR["soft_contact_mu"]}


def particle_radius_for(points, radius=None):
    """Half the median distance from a particle to its nearest neighbour, so neighbours touch
    without overlapping, whatever the asset's scale."""
    import numpy as np

    if radius is not None:
        return float(radius)
    points = np.asarray(points, dtype=np.float64)
    picked = np.random.default_rng(0).choice(len(points), size=min(512, len(points)), replace=False)
    distances = np.sqrt(((points[picked][:, None, :] - points[None, :, :]) ** 2).sum(-1))
    distances[distances < 1e-9] = np.inf  # a sampled particle matches itself in the full set
    return float(np.median(distances.min(axis=1)) * 0.5)


def size_particles_on(builder, radius=None):
    """Set the particle radius on the builder, before anything is finalised.

    Newton's ModelBuilder defaults `particle_radius` to 0.1 m and Isaac never changes it, so a 17 cm
    banana imports as 3074 particles each 10 cm across. Setting it on the finished model is too
    late: the solver is constructed from the model and keeps what it was built with. Newton's own
    examples set it while building, which is what this does.
    """
    import numpy as np

    if not builder.particle_count:
        return None
    value = particle_radius_for(np.asarray(builder.particle_q, dtype=np.float64), radius)
    was = float(max(builder.particle_radius))
    for i in range(builder.particle_count):
        builder.particle_radius[i] = value
    PARTICLE_SIZING.update({"radius_m": round(value, 6), "was_m": round(was, 6),
                            "particles": int(builder.particle_count), "set_on": "builder"})
    return value


PARTICLE_SIZING = {}  # what the last build gave its particles; read back into result.json


def size_particles(radius=None, soft_contact=None):
    """Give the model's particles a radius and contact stiffness that fit the asset, and report both.

    Newton's ModelBuilder defaults `particle_radius` to 0.1 m and Isaac never changes it, so a
    17 cm banana imports as 3074 particles each 10 cm across, packed into a body 3 cm thick. Every
    solver then starts from enormous overlap and throws the asset out of the scene -- which is what
    both VBD and XPBD did. Newton's own examples set the radius explicitly (3 mm for centimetre-
    scale objects).

    Left to itself this uses half the median distance from a particle to its nearest neighbour, so
    neighbouring particles touch and do not overlap, whatever the asset's scale.
    """
    import numpy as np

    import isaacsim.physics.newton as isaac_newton

    model = getattr(isaac_newton.acquire_stage(), "model", None)
    if model is None or not getattr(model, "particle_count", 0):
        return None
    points = np.asarray(model.particle_q.numpy())
    if radius is None:
        picked = np.random.default_rng(0).choice(len(points), size=min(512, len(points)), replace=False)
        distances = np.sqrt(((points[picked][:, None, :] - points[None, :, :]) ** 2).sum(-1))
        distances[distances < 1e-9] = np.inf  # a sampled particle matches itself in the full set
        radius = float(np.median(distances.min(axis=1)) * 0.5)
    was = float(np.asarray(model.particle_radius.numpy()).max())
    model.particle_radius.fill_(float(radius))
    contact = dict(soft_contact_for(np.asarray(model.particle_mass.numpy()),
                                    float(np.asarray(model.particle_radius.numpy()).max())),
                   **(soft_contact or {}))
    before = {name: float(getattr(model, name)) for name in contact}
    for name, value in contact.items():
        setattr(model, name, float(value))
    return {"radius_m": round(float(radius), 6), "was_m": round(was, 6), "particles": int(model.particle_count),
            "soft_contact": contact, "soft_contact_was": before}
