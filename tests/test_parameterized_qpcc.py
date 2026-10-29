import casadi as ca
import numpy as np
import pytest

from casadi_ccopt import complementarity, construct_ccopt_solver


def test_parameterized_qpcc():
    """ 
    Tests that parameters propogate to new solver calls after construct_ccopt_solver is called.
    Sets up an optimization problem over x, y, and z scalars with 0 <= x ⟂ y >= 0
    and the objective (x - a)^2 + (y - b)^2 + (z - 1)^2 with bounds on z (-2, -1)
    Solves with (a, b) = (1, 2) and (2, 1) where the expected solution is (0, 2, -1) and (2, 0, -1) """
    opti = ca.Opti()
    x = opti.variable()
    y = opti.variable()
    z = opti.variable()
    opti.subject_to(opti.bounded(-2, z, -1)) # Make sure these persist through solver construction

    # Objective with parameters
    a = opti.parameter()
    b = opti.parameter()
    opti.minimize((x - a)**2 + (y - b)**2 + (z - 1)**2)

    # Complementarity
    complementarity(opti, x, y)
    solver, lbx = construct_ccopt_solver(opti)

    # Make sure complementarity bounds are correct
    np.testing.assert_array_equal(lbx, [-np.inf, -np.inf, -np.inf, 0, 0])

    # Function for solution values
    solution_values = ca.Function("solution_values", [opti.x], [ca.vertcat(x, y, z), opti.g])

    # Reuse one solver, updating only the Opti parameters and numerical arguments.
    for a_value, b_value, expected in [(1, 2, [0, 2]), (2, 1, [2, 0])]:
        # Update objective parameters and resolve
        opti.set_value(a, a_value)
        opti.set_value(b, b_value)
        result = solver(x0=opti.value(opti.x, opti.initial()), p=opti.value(opti.p),
                        lbx=lbx, lbg=opti.value(opti.lbg), ubg=opti.value(opti.ubg))
        assert solver.stats()["success"], solver.stats()

        # Check solution
        xyz, g = solution_values(result["x"])
        xyz = np.array(xyz).ravel()
        np.testing.assert_allclose(xyz, [*expected, -1], atol=1e-5)

        # Check slacks are positive and complementarity holds
        assert np.min(xyz[:2]) >= -1e-6
        assert abs(xyz[0] * xyz[1]) < 1e-6

        # Check bounds
        assert np.all(np.array(result["x"]).ravel() >= np.array(lbx) - 1e-6)
        assert np.all(np.array(g).ravel() >= opti.value(opti.lbg) - 1e-6)
        assert np.all(np.array(g).ravel() <= opti.value(opti.ubg) + 1e-6)
