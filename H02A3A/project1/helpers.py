"""Helper functions for project 1. You do not need to read or change this file,
and you do not submit it."""

import json
import pathlib

from cpmpy.expressions.variables import NDVarArray


def load_instances(filename: str = "instances.json") -> list[dict]:
    """The instances in a JSON file next to this one."""
    path = pathlib.Path(__file__).parent / filename
    with path.open() as f:
        return json.load(f)


def print_solution(data: dict, vars: tuple[NDVarArray, NDVarArray, NDVarArray]) -> None:
    """Print a solved model: the directory tree, then a table with the owner,
    parent and permissions of every file. An unassigned variable (value None)
    is shown as absent: owner `-`, a permission bit `-`, no arc."""
    n_files, n_users = data["n_files"], data["n_users"]
    protected = set(data["protected"])
    owners, permissions, arcs = vars

    def name(f):
        return f"{f}*" if f in protected else str(f)

    parents = {j: [i for i in range(n_files) if arcs[i, j].value()] for j in range(n_files)}
    children = {i: [j for j in range(n_files) if arcs[i, j].value()] for i in range(n_files)}
    shown = set()

    def show_tree(f, prefix, last, top):
        line = prefix + ("" if top else ("`-- " if last else "|-- ")) + name(f)
        if f in shown:  # a second parent, or a cycle
            print(line + " (again, see above)")
            return
        shown.add(f)
        print(line)
        prefix += "" if top else ("    " if last else "|   ")
        for k, c in enumerate(children[f]):
            show_tree(c, prefix, k == len(children[f]) - 1, False)

    def on_cycle(f):
        """Follow parents from f until a file repeats: that file is on a cycle."""
        seen = []
        while f not in seen:
            seen.append(f)
            if not parents[f]:
                return f
            f = parents[f][0]
        return f

    print("Directory tree (* = protected file):")
    for f in range(n_files):
        if not parents[f]:
            show_tree(f, "  ", True, True)
    if len(shown) < n_files:
        print("  not below a root (the arcs contain a cycle):")
        for f in range(n_files):
            if f not in shown:
                show_tree(on_cycle(f), "    ", True, True)

    print("\nOwners and permissions (r = read, w = write, x = execute):")
    header = ["file", "owner", "parent"] + [f"user {u}" for u in range(n_users)]
    rows = [
        [name(f), "-" if owners[f].value() is None else str(owners[f].value()),
         ",".join(map(str, parents[f])) or "-"]
        + ["".join(c if permissions[f, u, p].value() else "-" for p, c in enumerate("rwx"))
           for u in range(n_users)]
        for f in range(n_files)
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  " + "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
