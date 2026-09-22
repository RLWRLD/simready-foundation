# SPDX-License-Identifier: Apache-2.0
"""The rules this package keeps, checked rather than written down.

    python asset_checks/native/check_rules.py [package_dir]

Three of them, each of which has been broken at least once and found by measurement:

  1. **No asset is named in code.** A rule that fires on the banana and not on the next asset is
     not a rule. Names in comments and messages are fine -- they are explaining, not deciding.
  2. **The experiment knows no engine.** `drop_shape` and `press_shape` state what the experiment
     asks; each engine arranges to be able to represent it. If an experiment module imports an
     engine or takes one as a parameter, the experiment has started varying by engine.
  3. **Experiments are declared in one place.** `asset_checks/experiments/` is it. Anything else
     holding a list of experiment names is a second copy that will drift from the first.

Exit 1 on any breach, naming the file and the line.
"""
import argparse
import ast
import pathlib
import sys

# The assets we happen to have, by the stem of their file. A rule keyed on one of these is keyed
# on today's data. Words that are also physics -- "cloth" is a kind of body, not an asset -- are
# not here: the thing to catch is a *file* being recognised, not a noun being used.
ASSETS = ("banana", "sm_obs_orange_a01_01", "sm_obs_orange_a02_01", "sm_apple_a01_01",
          "sm_alcohol_a01_01", "sm_coffee_cup_grasp_a01_01", "sm_gen_appliance_toaster_v01_01",
          "sm_gen_cleaning_dishwand_v01_01", "sm_obs_electricians_large_tool_box_a01_01",
          "sm_obs_joystick_a01_01", "sm_obs_lamp_revolute_a01_01", "sm_obs_light_bulb_01",
          "sm_obs_small_sledge_hammer_a01_01", "sm_obs_workbench_tool_a01_01")
# The files that state an experiment's rules, and must know nothing of engines.
EXPERIMENT_RULES = ("drop_shape.py", "press_shape.py")
ENGINES = ("newton", "physx", "warp", "omni", "isaacsim", "mujoco")
# Where an experiment may be named.
DECLARED_IN = ("experiments", "check_rules.py", "README.md")


def strings_and_names(tree):
    """Every string literal and every identifier, with its line: what the code decides on."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr


def asset_named_in_code(path, tree):
    """Rule 1. Docstrings are excluded: they explain, and a measurement is worth naming."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    for line, text in strings_and_names(tree):
        if text in docstrings:
            continue
        low = text.lower()
        for asset in ASSETS:
            # The stem exactly, or a path or filename built from it. Prose is excluded above; what
            # is left is the code recognising one particular file.
            if low == asset or low.startswith(asset + ".") or low.endswith("/" + asset) \
                    or ("/" + asset + ".") in low:
                yield line, f"the asset {asset!r} is named in code as {text!r}"


def experiment_knows_an_engine(path, tree):
    """Rule 2."""
    if path.name not in EXPERIMENT_RULES:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                if name and name.split(".")[0] in ENGINES:
                    yield node.lineno, f"{path.name} imports {name}: the experiment must not know an engine"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg in ENGINES or arg.arg in ("engine", "solver"):
                    yield node.lineno, (f"{path.name}:{node.name} takes {arg.arg!r}: the experiment's "
                                        f"rules must not vary by engine")


def second_list_of_experiments(path, tree, names):
    """Rule 3: a collection literal holding two or more experiment names, outside the one place."""
    if any(part in DECLARED_IN for part in path.parts) or path.name in DECLARED_IN:
        return
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            continue
        found = {e.value for e in node.elts
                 if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value in names}
        if len(found) > 1:
            yield node.lineno, (f"{sorted(found)} is a second list of experiments; they are declared "
                                f"in asset_checks/experiments/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("package", nargs="?", default=str(pathlib.Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    package = pathlib.Path(args.package).resolve()

    sys.path.insert(0, str(package.parent))
    from asset_checks import experiments

    names = set(experiments.ALL)
    problems = []
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(package)
        if relative.name == "check_rules.py":
            continue    # the list of what not to name has to name it
        for check in (asset_named_in_code, experiment_knows_an_engine):
            for line, why in check(path, tree):
                problems.append(f"{relative}:{line}: {why}")
        for line, why in second_list_of_experiments(relative, tree, names):
            problems.append(f"{relative}:{line}: {why}")

    for problem in problems:
        print(f"[rules] {problem}")
    print(f"[rules] {len(problems)} breach(es) of the rules this package keeps")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
