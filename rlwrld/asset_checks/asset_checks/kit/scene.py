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
    import omni.kit.app
    import omni.usd
    from pxr import UsdGeom

    ok, err = await omni.usd.get_context().new_stage_async()
    if not ok:
        raise RuntimeError(f"could not create a new stage: {err}")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
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

    variant_spec = layer.GetPrimAtPath(asset_root).variantSets[entry["variantSetName"]].variants[option].primSpec
    for payload in variant_spec.payloadList.GetAddedOrExplicitItems():
        payload_layer = Sdf.Layer.FindOrOpen(layer.ComputeAbsolutePath(payload.assetPath))
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
