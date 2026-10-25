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
    
    #assert (np_arcs != None).all(), "stage 3: an arc is None"


    # TODO: we HIGHLY recommend that you add your own checks for every stage!
    # You probably even want to write the check first,
    # before modeling it as constraints.


# ---------------------------------------------------------------------------
# Permissions and owners
# ---------------------------------------------------------------------------


def stage1(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars
    # TODO: add the constraints of this stage with  model.add(...)
    return model


def stage2(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars
    # TODO: add the constraints of this stage with  model.add(...)
    return model


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------


def stage3(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars
    # TODO: add the constraints of this stage with  model.add(...)
    return model


def stage4(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    owners, permissions, arcs = vars
    # TODO: add the constraints of this stage with  model.add(...)
    return model


def stage5(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    files_owned = data["files_owned"]
    owners, permissions, arcs = vars
    # TODO: add the constraints of this stage with  model.add(...)
    return model


# ---------------------------------------------------------------------------
# Protected files (data["protected"])
# ---------------------------------------------------------------------------


def stage6(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    protected = data["protected"]
    owners, permissions, arcs = vars
    # TODO: add the constraints of this stage with  model.add(...)
    return model


def stage7(model: cp.Model, vars: tuple[NDVarArray, NDVarArray, NDVarArray], data: dict) -> cp.Model:
    n_files, n_users = data["n_files"], data["n_users"]
    protected = data["protected"]
    owners, permissions, arcs = vars
    # TODO: add the constraints of this stage with  model.add(...)
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
