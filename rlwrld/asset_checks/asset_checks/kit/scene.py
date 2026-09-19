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
