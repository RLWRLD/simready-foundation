"""The two package-sidecar helpers make_newton_variant.py needs, vendored from DexBench-Arena's conform_simready_basics.py.

A DexBench package is ``<name>/usd/<name>.usd`` with a JSON sidecar ``<name>.meta.json`` beside it
(``asset.version`` and provenance) and a ``CHANGELOG.md`` at the package root. ``bump`` raises the
version (PATCH, or MINOR when asked) and prepends the change to the changelog.
"""

from __future__ import annotations

import json
from pathlib import Path


def sidecar_for(usd: Path) -> Path | None:
    for cand in (usd.with_suffix(".meta.json"), usd.parent / f"{usd.stem}.meta.json"):
        if cand.is_file():
            return cand
    return None


def bump(pkg: Path, usd: Path, note: str, today: str, minor: bool = False) -> str | None:
    """Bump the sidecar version (PATCH, or MINOR when asked) and prepend a changelog entry."""
    sc = sidecar_for(usd)
    if sc is None:
        return None
    meta = json.loads(sc.read_text())
    block = meta.setdefault("asset", {})
    major, mnr, patch = ((block.get("version") or "1.0.0").split(".") + ["0", "0"])[:3]
    version = f"{major}.{int(mnr) + 1}.0" if minor else f"{major}.{mnr}.{int(patch) + 1}"
    block["version"] = version
    sc.write_text(json.dumps(meta, indent=2) + "\n")
    log = pkg / "CHANGELOG.md"
    head = log.read_text() if log.exists() else f"# {pkg.name}\n\nNewest first. A version with no entry here is not a version.\n\n"
    marker = "Newest first. A version with no entry here is not a version.\n\n"
    entry = f"## {today} — {version}\n\n{note}\n\n"
    head = head.replace(marker, marker + entry, 1) if marker in head else head + "\n" + entry
    log.write_text(head)
    return version
