"""Read every runner without running it, and say what would fail when it did.

    python asset_checks/native/check_sources.py [directory]

Three times in this pipeline a change has parsed cleanly and then died on the first run: a
deleted line left `fell` undefined, a constant was inserted below the code that reads it, an
import moved above the Kit app that has to exist first. Each cost a full render pass to discover.
A parser answers "is this Python"; these are the questions that actually decide whether a run
survives its first second, and none of them needs a GPU.

This does not replace running the thing. It replaces *discovering* these three by running it.
"""
import argparse
import ast
import pathlib
import sys

BUILTINS = set(dir(__builtins__)) | {"__file__", "__name__", "__doc__", "__builtins__"}
# Modules that pull in USD and therefore must not be imported before Kit exists, in a file that
# starts Kit at all. Importing them early initialises USD outside Kit, and Kit then cannot
# register its own schema wrappers -- the run dies during startup, long before any physics.
NEEDS_KIT_FIRST = {"pxr", "omni", "isaacsim", "asset_properties", "usd_deformable",
                   "recording", "skinning", "physx_parity", "warp", "newton"}
STARTS_KIT = "SimulationApp("


def module_scope(tree):
    """Names bound at module level, each with the line it is first bound on."""
    bound = {}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.setdefault((alias.asname or alias.name).split(".")[0], line)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.setdefault(node.name, line)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.setdefault(node.id, line)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.setdefault(node.name, line)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                for name in ast.walk(item.optional_vars) if item.optional_vars else []:
                    if isinstance(name, ast.Name):
                        bound.setdefault(name.id, line)
        elif isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            for clause in node.generators:
                for name in ast.walk(clause.target):
                    if isinstance(name, ast.Name):
                        # Bound from the top of the expression, not from wherever `walk` reached
                        # it -- `any(p.IsA(...) for p in stage.Traverse())` reads `p` first.
                        bound[name.id] = min(bound.get(name.id, line), getattr(node, "lineno", line))
    return bound


def local_names(node):
    """Everything a function binds for itself, so its body is not judged against module scope."""
    names = set()
    # Every function in this subtree, not only the outer one: a nested `def place_plate(state,
    # z, speed)` binds three names that the outer function's body never assigns, and judging
    # its body without them reported all three as undefined.
    for scope in [node] + [n for n in ast.walk(node)
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]:
        args = scope.args
        for arg in list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs):
            names.add(arg.arg)
        for arg in (args.vararg, args.kwarg):
            if arg:
                names.add(arg.arg)
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and isinstance(inner.ctx, (ast.Store, ast.Del)):
            names.add(inner.id)
        elif isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(inner.name)
        elif isinstance(inner, ast.ExceptHandler) and inner.name:
            names.add(inner.name)
        elif isinstance(inner, (ast.Import, ast.ImportFrom)):
            for alias in inner.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def enclosing_loop_start(node, parent):
    """The line a module-level loop containing this node starts on, if any.

    A name assigned late in a loop body is available on every pass after the first, so a read
    earlier in that same loop is not a use-before-definition. Judging by source order alone
    reported three of those.
    """
    start = None
    while node in parent:
        node = parent[node]
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            start = getattr(node, "lineno", start)
    return start


def check(path):
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError as bad:
        return [f"{path.name}:{bad.lineno}: will not parse: {bad.msg}"]

    complaints = []
    bound = module_scope(tree)
    # Which nodes sit inside a function. Walking down from each function and collecting ids
    # looks equivalent and is not: `ast.walk` yields the same objects, but a node reachable by
    # two paths made the membership test unreliable. Walking *up* from each node is unambiguous.
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def in_a_function(node):
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return True
        return False

    # 1. a name that is never bound anywhere
    # 2. a module-level name read above the line that binds it
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        if node.id in BUILTINS:
            continue
        if node.id not in bound:
            if not in_a_function(node):
                complaints.append(f"{path.name}:{node.lineno}: `{node.id}` is never defined")
            continue
        # Only a read strictly above the *first* binding is too early. A loop that rebinds a
        # name further down is not a problem -- `v` is set inside the loop and read after it.
        loop = enclosing_loop_start(node, parent)
        if (not in_a_function(node) and node.lineno < bound[node.id]
                and not (loop is not None and bound[node.id] > loop)):
            complaints.append(f"{path.name}:{node.lineno}: `{node.id}` is read here but only "
                              f"defined on line {bound[node.id]}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A nested function sees its enclosing function's names too, so gather every
            # function on the way down rather than this one alone.
            local = BUILTINS | set(bound) | local_names(node)
            walker = node
            while walker in parent:
                walker = parent[walker]
                if isinstance(walker, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    local |= local_names(walker)
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)
                        and inner.id not in local):
                    complaints.append(f"{path.name}:{inner.lineno}: `{inner.id}` is never defined")

    # 3. in a file that starts Kit, nothing that touches USD may be imported before it does
    kit_line = next((i + 1 for i, line in enumerate(text.splitlines()) if STARTS_KIT in line), None)
    if kit_line:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)) or node.lineno >= kit_line:
                continue
            for alias in node.names:
                root = (alias.name if isinstance(node, ast.Import) else (node.module or "")).split(".")[0]
                if root in NEEDS_KIT_FIRST and root != "isaacsim":
                    complaints.append(f"{path.name}:{node.lineno}: `{root}` is imported before "
                                      f"SimulationApp on line {kit_line}; USD initialised outside "
                                      f"Kit stops Kit registering its own schemas")
    return complaints


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", nargs="?", default=str(pathlib.Path(__file__).resolve().parent))
    args = ap.parse_args()
    complaints = []
    for path in sorted(pathlib.Path(args.directory).glob("*.py")):
        complaints += check(path)
    for complaint in sorted(set(complaints)):
        print(f"[sources] {complaint}")
    print(f"[sources] {len(set(complaints))} problem(s) that would only show up at run time")
    return 1 if complaints else 0


if __name__ == "__main__":
    sys.exit(main())
