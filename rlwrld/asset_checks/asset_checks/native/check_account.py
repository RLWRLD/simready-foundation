# SPDX-License-Identifier: Apache-2.0
"""Negative tests of `asset_properties.account` on real assets: the rule that a run refuses an
authored attribute nobody reads, and a quantity stated twice whose statements disagree.

    <isaac venv>/bin/python asset_checks/native/check_account.py <asset.usd> [...]

For each asset (CPU only, pxr needed): the asset as it is must be accepted under the canon setup
with what the Newton runners consume; then each violation is planted in a layer over it and must
be refused, and the refusal must name what was planted --
  * an unknown vendor attribute on a simulated prim,
  * every restated attribute the asset authors, perturbed by 10% (`ACCOUNTED` class `restated`),
  * a structure attribute the runner stops reading (the 2026-09-22 state: edge exclusions unread).
Exit 1 on any case that does not behave. Nothing is written beside the asset.
"""
import argparse
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pxr import Sdf, Usd  # noqa: E402

import asset_properties  # noqa: E402
import setups  # noqa: E402

# What the Newton runners consume beyond `read()`, per the loader, when the asset authors it.
RUNNER_READS = ("newton:particleRadius", "newton:kDamp", "newton:triKd", "newton:edgeKd",
                "newton:triKe", "newton:edgeKe", asset_properties.SEAL_PAIRS, asset_properties.SEAL_KE,
                asset_properties.VERTEX_TRIANGLE_EXCLUSIONS, asset_properties.EDGE_EXCLUSIONS,
                *(n for k in ("self_contact_radius", "self_contact_margin", "seal_ke", "contact_ke",
                              "contact_kd", "friction", "shape_ke", "dt", "iterations")
                  for n in asset_properties.RECIPE[k]))


def verdict(asset, consumed):
    declared = asset_properties.read(asset)
    setup = setups.resolve(setups.parse(setups.CANON), declared)[0]
    asset_properties.consume(declared, *consumed)
    try:
        asset_properties.account(declared, setup)
        return None
    except SystemExit as refusal:
        return str(refusal)


def over(asset, work, name, edits):
    """A layer over `asset` with `edits` [(prim path, attribute, value)] authored; -> its path."""
    source = Usd.Stage.Open(str(asset))
    path = pathlib.Path(work) / f"{name}.usda"
    layer = Sdf.Layer.CreateNew(str(path))
    layer.subLayerPaths.append(str(asset))
    layer.defaultPrim = source.GetDefaultPrim().GetName()
    stage = Usd.Stage.Open(layer)
    for prim_path, attr, value in edits:
        original = source.GetPrimAtPath(prim_path).GetAttribute(attr)
        typ = original.GetTypeName() if original else Sdf.ValueTypeNames.Double
        stage.OverridePrim(prim_path).CreateAttribute(attr, typ, custom=True).Set(value)
    layer.Save()
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("assets", nargs="+")
    args = ap.parse_args()
    bad = 0
    with tempfile.TemporaryDirectory() as work:
        for asset in args.assets:
            name = pathlib.Path(asset).name
            declared = asset_properties.read(asset)
            authored = declared["_authored"]
            reads = [n for names in authored.values() for n in names if n in RUNNER_READS]
            cases = [("as it is", asset, reads, None)]
            first_prim = next(iter(authored)) if authored else str(Usd.Stage.Open(asset).GetDefaultPrim().GetPath())
            cases.append(("unknown attribute", over(asset, work, "unknown", [(first_prim, "vendor:somethingNew", 1.0)]),
                          reads, "vendor:somethingNew"))
            stage = Usd.Stage.Open(asset)
            for path, names in authored.items():
                for attr in names:
                    if asset_properties.ACCOUNTED.get(attr) == "restated":
                        value = stage.GetPrimAtPath(path).GetAttribute(attr).Get()
                        wrong = value * 1.1 if value else 1.0
                        cases.append((f"{attr} off by 10%", over(asset, work, attr.replace(":", "_"),
                                                                 [(path, attr, wrong)]), reads, attr))
            if asset_properties.EDGE_EXCLUSIONS in reads:
                cases.append(("edge exclusions unread (2026-09-22)", asset,
                              [n for n in reads if n != asset_properties.EDGE_EXCLUSIONS],
                              asset_properties.EDGE_EXCLUSIONS))
            for label, path, consumed, expect in cases:
                said = verdict(path, consumed)
                ok = said is None if expect is None else (said is not None and expect in said)
                bad += not ok
                print(f"[account] {name}: {label}: {'ok' if ok else 'WRONG'} -- "
                      + ("accepted" if said is None else said[:140]))
    print(f"[account] {bad} case(s) that did not behave")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
