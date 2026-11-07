"""
Constraint Solving [B-KUL-H02A3A] -- Project 1: Passing the security audit

The problem, setup, and grading are described in project1_description.pdf.
Complete the functions stage1 ... stage7: each adds the constraints of that
stage to the given model and returns it. Submit only this file.

The auto-audit imports this file and calls the stage functions on several
instances, so:
  - do not rename the stage functions or change their parameters;
  - do not change make_variables, and use the variables it creates (you may
    add auxiliary variables and helper functions);
  - make your functions work for any instance in the format of the instances
    in `instances.json`;
  - model the requirements of each stage in its own function; a stage may rely on
    the requirements of earlier stages (e.g. stage 4 may use that there is exactly
    one root, from stage 3);
  - put code that solves or prints inside `my_checker()` or at the bottom of
    the file.

Run this file to solve each of the instances and run your own checks.
"""

import cpmpy as cp
from cpmpy.expressions.variables import NDVarArray

from helpers import load_instances, print_solution
import itertools  # optionally, for itertools.combinations


# Permission types: the last index of `permissions`
READ, WRITE, EXECUTE = 0, 1, 2

# The "admin" user
ADMIN = 0


def make_variables(data: dict) -> tuple[NDVarArray, NDVarArray, NDVarArray]:
    """The decision variables of an instance. Do not change."""
    n_files, n_users = data["n_files"], data["n_users"]
    # each file is owned by a user
    owners = cp.intvar(0, n_users - 1, name="owner", shape=n_files)
    # each file, user, and permission type is enabled (True) or disabled (False)
    permissions = cp.boolvar(name="permission", shape=(n_files, n_users, 3))
    # `arcs[i,j]` is True iff file `i` contains file `j`
    arcs = cp.boolvar(name="arcs", shape=(n_files, n_files))
    return owners, permissions, arcs


# ---------------------------------------------------------------------------
# Your own checks (not graded)
# ---------------------------------------------------------------------------

def my_checker(data: dict, vars: tuple[NDVarArray, NDVarArray, NDVarArray]) -> None:
    """Your own checks on one solution of your model (not graded).
    """
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars

    # Convert the values in the decision variables to numpy arrays for easier checking
    np_owners = owners.value()
    np_permissions = permissions.value()
    np_arcs = arcs.value()


    # Stage 1 checks
    assert np_owners[0] == ADMIN, "stage 1: the admin does not own file 0"

    for f in range(n_files):
        assert np_permissions[f, ADMIN, WRITE], (
          f"stage 1: the admin does not have write permissions for file {f}"
        )

    # TODO: we HIGHLY recommend that you add your own checks for every stage!
    # You probably even want to write the check first,
    # before modeling it as constraints.

    # Stage 2 checks
    for f in range(n_files):
        owner = np_owners[f]
        assert np_permissions[f, owner, READ], (
            f"stage 2: owner {owner} of file {f} does not have read permission"
        )
        assert np_permissions[f, owner, WRITE], (
            f"stage 2: owner {owner} of file {f} does not have write permission"
        )

        for u in range(n_users):
            if np_permissions[f, u, WRITE]:
                assert np_permissions[f, u, READ], (
                    f"stage 2: user {u} can write but not read file {f}"
                )
            if np_permissions[f, u, EXECUTE]:
                assert np_permissions[f, u, WRITE], (
                    f"stage 2: user {u} can execute but not write file {f}"
                )

    # Stage 3 checks
    assert (np_arcs != None).all(), "stage 3: an arc is None"
    parents_per_file = np_arcs.astype(int).sum(axis=0)  # How many files contain the same file in column j
    for f in range(n_files):
        assert parents_per_file[f] <= 1, f"stage 3: file {f} has more than one parent directory"
    roots = [f for f in range(n_files) if parents_per_file[f] == 0]
    assert len(roots) == 1, f"stage 3: expected exactly one root file, found {roots}"

    # Stage 4 checks
    visited = set(roots)
    current_size = len(visited)
    last_size = 0
    while current_size > last_size:
        last_size = len(visited)
        for d in range(n_files):
            if d in visited:
                for f in range(n_files):
                    if np_arcs[d, f]:
                        visited.add(f)
        current_size = len(visited)
    assert (visited == set(range(n_files))), f"stage 4: expected all files to be reachable, found {visited}"


# ---------------------------------------------------------------------------
# Permissions and owners
# ---------------------------------------------------------------------------


def stage1(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars

    model.add(owners[0] == ADMIN) # ADMIN needs to own ID 0

    for i in range(n_files):
        model.add(permissions[i][0][WRITE]) # ADMIN needs WRITE permission on every file

    return model


def stage2(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars

    for file in range(n_files):
        model.add(permissions[file, owners[file], READ] == True) # Every user can read their own files
        model.add(permissions[file, owners[file], WRITE] == True) # Every user can write on their own files
        for user in range(n_users):
            model.add((~permissions[file, user, WRITE]) | (permissions[file, user, READ])) # A user who write a file must be able to read it
            model.add((~permissions[file, user, EXECUTE]) | (permissions[file, user, WRITE])) # A user who executes a file must be able to write on it

    return model


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------


def stage3(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars

    # Auxiliary variables needed to keep track of how many columns qualify for root constraint
    aux = cp.boolvar(name="root_auxiliary", shape=n_files)

    for file in range(n_files):
        model.add(((~aux[file]) | (cp.Count(arcs[:, file], False) == n_files)) &
                  ((~(cp.Count(arcs[:, file], False) == n_files)) | (aux[file]))) # if and only if is logically equivalent to the conjunction of two implications

    model.add(cp.Count(aux, True) == 1) # There exists exactly one root directory

    for col in arcs.T:
        model.add(cp.AllDifferentExceptN(col, False)) # Every column has to have at most one True value because every file can only have one parent directory

    return model


def stage4(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars

    # Grows exponentially for n_files, could be better?
    for size in range(1, n_files):
        for subset in itertools.combinations(range(n_files), size):
            complement = list(set(range(n_files)) - set(subset)) # need the complement to check if there is a path from subset to complement
            model.add((cp.Count(arcs[subset, :][:, complement].flatten(), True) > 0) |
                      (cp.Count(arcs[complement, :][:, subset].flatten(), True) > 0)) # check if at least one arc is present
    return model


def stage5(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    files_owned = data["files_owned"]
    owners, permissions, arcs = vars

    for file in range(n_files):
        for user in range(n_users):
            model.add((cp.Count(arcs[file, :], True) > 0).implies(~permissions[file, user, EXECUTE])) # if the file is a directory, it cannot be executable

    for di in range(n_files):
        for user in range(n_users):
            for file in range(n_files):
                for permission in range(3):
                    model.add(((arcs[di, file]) &
                               (permissions[di, user, permission])).implies(permissions[file, user, permission])) # if a file is contained in a directory, it inherits the permissions of the directory (the implication is reversed for simplicity in modelling)

    for user in range(n_users):
        model.add(cp.Count(owners, user) == files_owned[user]) # the number of files owned by a user must be equal to the number provided

    return model


# ---------------------------------------------------------------------------
# Protected files (data["protected"])
# ---------------------------------------------------------------------------


def stage6(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    protected = data["protected"]
    owners, permissions, arcs = vars

    for pf in protected:
        model.add(cp.Count(arcs[pf, :], True) == 0) # A protected file cannot be a directory

    for di in range(n_files):
        for pf in protected:
            model.add((arcs[di, protected[0]]).implies(arcs[di, pf])) # if a directory contains the first protected file, it must contain every other

    return model


def stage7(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    protected = data["protected"]
    owners, permissions, arcs = vars
    for di in range(n_files):
        model.add((arcs[di, protected[0]]).implies(cp.Count(arcs[:, di], True) == 1)) # The directory that contains the protected files is not the root

    aux_owners = [owners[i] for i in protected] # All owners of a protected file
    model.add(cp.AllDifferent(aux_owners)) # The owner of a protected file must be unique

    aux_log = cp.boolvar(name="log_auxiliary", shape=n_files) # Auxiliary variable to find a log file

    for file in range(n_files):
        for u in aux_owners:
            for di in range(n_files):
                model.add(aux_log[file].implies((cp.Count(protected, file) == 0) & # The log file is unprotected
                                             (cp.Count(arcs[file, :], True) == 0) & # The log file is not a directory
                                             (permissions[file, u, WRITE] == True) & # All owners of a protected file can write to the log file
                                             (arcs[di, file].implies(arcs[di, protected[0]])))) # The log file is stored in the same directory as the protected files

    model.add(cp.Count(aux_log, True) > 0) # There exists at least one log file

    return model


def objective(vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict):
    """Bonus (ungraded): the number of permissions granted, to be minimized."""
    owners, permissions, arcs = vars
    return None  # TODO (optional)



if __name__ == "__main__":
    # Every instance in instances.json; use e.g. instances[:1] to look at fewer.
    instances = load_instances()
    for data in instances:
        print(f"Running instance {data['id']}")

        variables = make_variables(data)
        model = cp.Model()

        # All stages together. Remove stages from the tuple to look at fewer.
        stages = (stage1, stage2, stage3, stage4, stage5, stage6, stage7)
        for stage in stages:
            model = stage(model, variables, data)

        # Optional objective: for students that do the non-graded bonus.
        obj = objective(variables, data)
        if obj is not None:
            model.minimize(obj)

        assert model.solve(), f"no solution for {data}"

        try:
            my_checker(data, variables)
        except AssertionError as e:
            print_solution(data, variables)
            raise

    print("Ran 'my_checker()' on all instances, you didn't find any issues.")
