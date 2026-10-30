"""Minimal complementarity helpers for CasADi Opti and CCOpt."""

import casadi as ca


def complementarity(opti, left, right, left_var = False, right_var = False):
    """Implements complementarity 0 <= left ⟂ right >= 0 by defining slack variables 
    for each side vl = left, vr = right, and recording them to be used to construct 
    complementarity pairs after the problem construction is completed. 

    If left or right are already opti variables, you can use left_var = True and/or right_var = True
    to use them directly instead of creating slacks
    
    Nonnegativity is enforced through lbx returned by construct_ccopt_solver
    """
    # Vectorize constraints and ensure sizes match
    left = ca.vec(left)
    right = ca.vec(right)
    if left.numel() != right.numel():
        raise ValueError("Complementarity sides must have the same number of scalar elements.")

    # Create slack variables and their inequalities
    if left_var:
        vl = left
    else:
        vl = opti.variable(left.numel())
        opti.subject_to(vl == left)

    if right_var:
        vr = right
    else:
        vr = opti.variable(right.numel())
        opti.subject_to(vr == right)

    # Record variable pair for determining indices for cc_pairs later
    # This has to be done after the problem construction is completed because indices can change
    if not hasattr(opti, "_cc_vars"):
        opti._cc_vars = []
    opti._cc_vars.append((vl, vr))
    return vl, vr


def _get_cc_pairs(opti):
    # Establishes flattened ordering for opti.x, opti.g, and constructs other info needed for problem construction
    adv = opti.advanced
    adv.bake()

    # Opti doesn't support variable -> indice lookup, only supports
    # indice -> variable by block (i.e. if variable x is a 4x1 vector, all 4 indices will return x)
    # the dictionary associates var with its ordered indices (this assumes opti preserves ordering)
    var_to_indices = {}
    for i in range(opti.nx):
        var = adv.x_lookup(i)
        var_to_indices.setdefault(var.name(), []).append(i)

    # Use the dictionary to associate the variables to their indices and construct cc_pairs
    cc_pairs = []
    for vl, vr in getattr(opti, "_cc_vars", []):
        il = var_to_indices[vl.name()]
        ir = var_to_indices[vr.name()]
        if len(il) != len(ir):
            raise ValueError("Complementarity variable blocks must have equal lengths.")
        cc_pairs.extend([l, r] for l, r in zip(il, ir))
    return cc_pairs


def construct_ccopt_solver(opti, opts=None):
    """Return (solver, lbx) for a completed Opti model.

    solver is a raw CCOpt nlpsol. lbx is a list with zero at complementarity indices
    and -inf elsewhere; 
    Copy opts and override cc_pairs and detect_simple_bounds as required by the adapter.
    """
    opts = dict(opts or {})

    # Determine indices for complementarity pairs from stored opti._cc_vars
    cc_pairs = _get_cc_pairs(opti)
    opts["cc_pairs"] = cc_pairs

    # The CCOpt adapter requires explicit variable bounds; disable simple-bound detection.
    opts["detect_simple_bounds"] = False
    lbx = [-ca.inf] * opti.nx
    for l, r in cc_pairs:
        lbx[l] = lbx[r] = 0.0

    # Construct nlp
    nlp = {"x": opti.x, "p": opti.p, "f": opti.f, "g": opti.g}
    return ca.nlpsol("solver", "ccopt", nlp, opts), lbx
