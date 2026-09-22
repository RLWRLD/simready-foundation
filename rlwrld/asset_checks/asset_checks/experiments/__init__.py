# SPDX-License-Identifier: Apache-2.0
"""The experiments, one module each, discovered by being in this directory.

To add an experiment, add a file here. It declares what it is and who drives it:

    NAME        what the command line calls it
    SUMMARY     one line, for `--help` and the report
    KINDS       the asset kinds it means anything for: {"rigid"}, {"deformable"}, or both
    SECONDS     how long it runs, in simulated seconds (deformable path; NVIDIA's tests time
                themselves)
    RIGID       ("module", "function") of the NVIDIA test that is this experiment for a rigid
                asset, or None if it has none
    DEFORMABLE  {engine: "runner.py"} under `native/`, or {} if it has none
    BODIES      optional: the deformable body kinds the experiment means anything for, as
                `usd_deformable` names them ({"volume"}, {"surface"}). An asset that declares no
                body of these kinds is refused before launch. Absent means any body.

Nothing else in the package keeps a list of experiments: `kit/nvidia_test.py` asks here which
NVIDIA test an experiment is, and the deformable runner asks here which script drives it. A file
that declares an asset kind it has no driver for is refused at load, rather than at the end of a
Kit launch.
"""
import importlib
import pkgutil


def _load():
    found = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        name = getattr(module, "NAME", info.name)
        kinds = set(getattr(module, "KINDS", ()))
        if not kinds:
            raise ImportError(f"{module.__name__} declares no KINDS")
        if "rigid" in kinds and not getattr(module, "RIGID", None):
            raise ImportError(f"{module.__name__} claims rigid assets but declares no RIGID test")
        if "deformable" in kinds and not getattr(module, "DEFORMABLE", None):
            raise ImportError(f"{module.__name__} claims deformable assets but declares no DEFORMABLE runner")
        found[name] = module
    return found


ALL = _load()


def get(name):
    """The experiment by name, or a SystemExit naming the ones there are."""
    if name not in ALL:
        raise SystemExit(f"unknown experiment {name!r}; there is " + ", ".join(sorted(ALL)))
    return ALL[name]


def for_kind(kind):
    """The experiments that mean something for an asset of this kind."""
    return {name: module for name, module in ALL.items() if kind in module.KINDS}


def rigid_tests():
    """{experiment: (module, function)} for the NVIDIA tests, as `kit/nvidia_test.py` wants them."""
    return {name: module.RIGID for name, module in ALL.items() if getattr(module, "RIGID", None)}
