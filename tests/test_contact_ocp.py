import casadi as ca
import numpy as np
import pytest

from casadi_ccopt import complementarity, construct_ccopt_solver


def test_contact_ocp():
    """
    Sets up a simple 3-step optimal control problem for a point mass hovering over the ground.
    The single degree of freedom is the height of the mass, the control can pull upwards, and
    contact forces are applied when height = 0 using 0 <= height ⟂ force >= 0 complementarity for contact.
    The dynamics are semi-implicit euler, and the cost is a tracking cost on a fixed reference height and
    a control cost. Tests two scenarios, one starting at height = 2 and reference = 2 (no contact) and 
    one starting at height = 0.05, reference = -1 (contact)
    """
    N = 3
    h = 0.1
    gravity = 9.81
    R = 1.0

    opti = ca.Opti()

    # Init position, velocity, controls and contact forces
    q = opti.variable(N + 1)
    v = opti.variable(N + 1)
    u = opti.variable(N)
    lam = opti.variable(N)

    # Initial conditions and constraints
    q_initial = opti.parameter()
    v_initial = opti.parameter()
    opti.subject_to(q[0] == q_initial)
    opti.subject_to(v[0] == v_initial)

    # Reference, tracking cost and semi-implicit euler dynamics with contact forces at q = 0
    # Complementarity is 0 <= q[k+1] ⟂ lam[k] >= 0
    q_ref = opti.parameter()
    cost = 0
    for k in range(N):
        # Semi-implicit Euler with unit mass and a nonnegative ground reaction.
        opti.subject_to(v[k + 1] == v[k] + h * (u[k] - gravity + lam[k]))
        opti.subject_to(q[k + 1] == q[k] + h * v[k + 1])
        complementarity(opti, q[k + 1], lam[k])
        cost += (q[k + 1] - q_ref)**2 + R * u[k]**2
    opti.minimize(cost)

    # Construct solver
    solver, lbx = construct_ccopt_solver(opti, {"print_time": False})

    # Trajectory from solution
    trajectory = ca.Function("trajectory", [opti.x], [q, v, u, lam, opti.g])

    # Run two trajectories, q = 2 with q_ref = 2 (no contact), q = 0.05 with q_ref = -1 (contact)
    for initial_height, reference, contact in [(2.0, 2.0, False), (0.05, -1.0, True)]:
        # Set initial condition and reference
        opti.set_value(q_initial, initial_height)
        opti.set_value(v_initial, 0.0)
        opti.set_value(q_ref, reference)

        # Solve the problem with parameters and bounds from opti
        result = solver(x0=opti.value(opti.x, opti.initial()), p=opti.value(opti.p),
                        lbx=lbx, lbg=opti.value(opti.lbg), ubg=opti.value(opti.ubg))
        assert solver.stats()["success"], solver.stats()

        # Extract solution
        q_sol, v_sol, u_sol, lam_sol, g_sol = [np.array(value).ravel()
                                              for value in trajectory(result["x"])]

        # Check initial condition constraint and dynamics constraints are satisfied
        np.testing.assert_allclose([q_sol[0], v_sol[0]], [initial_height, 0], atol=1e-6)
        np.testing.assert_allclose(v_sol[1:], v_sol[:-1] + h * (u_sol - gravity + lam_sol),
                                   atol=1e-6)
        np.testing.assert_allclose(q_sol[1:], q_sol[:-1] + h * v_sol[1:], atol=1e-6)

        # Check that complementarity between q and lam are satisfied
        assert np.min(q_sol) >= -1e-6
        assert np.min(lam_sol) >= -1e-6
        np.testing.assert_allclose(q_sol[1:] * lam_sol, 0, atol=1e-6)

        # Check variable bounds and constraint bounds
        assert np.all(np.array(result["x"]).ravel() >= np.array(lbx) - 1e-6)
        assert np.all(g_sol >= opti.value(opti.lbg) - 1e-6)
        assert np.all(g_sol <= opti.value(opti.ubg) + 1e-6)

        if contact: # Check that contact was made with normal force
            assert np.max(lam_sol) > 1.0
            assert np.min(q_sol[1:]) < 1e-5
        else: # Check that no contact forces were applied
            assert np.min(q_sol) > 1.0
            np.testing.assert_allclose(lam_sol, 0, atol=1e-5)
