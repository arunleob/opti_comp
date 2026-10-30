import casadi as ca
import numpy as np
import pytest

from casadi_ccopt import _get_cc_pairs, complementarity

def test_complementarity_transcription_and_indexing():
    """
    Runs tests to determine indexing and problem construction works for
    - scalar, vector and matrix complementarity
    - generated equality constraints for scalar and vector complementarity
    """
    # Construct problem with 2 scalar variables, 2 vector (2-d) variables
    # and scalar and vector complementarities
    # 0 <= x+1 ⟂ y >= 0, 0 <= z ⟂ 2*w >= 0
    opti = ca.Opti()
    x = opti.variable()
    y = opti.variable()
    vl, vr = complementarity(opti, x + 1, y)
    z = opti.variable(2)
    w = opti.variable(2)
    zl, zr = complementarity(opti, z, 2 * w)

    # Verify slacks match stored and sizes match
    assert len(opti._cc_vars) == 2
    for stored, expected in zip(opti._cc_vars, [(vl, vr), (zl, zr)]):
        assert all(ca.is_equal(a, b) for a, b in zip(stored, expected))
    assert vl.shape == vr.shape == (1, 1)
    assert zl.shape == zr.shape == (2, 1)

    # Verify total variables (6 primal, 6 slacks) and constraints (6 equalities)
    assert opti.nx == 12
    assert opti.ng == 6

    # Verify indices, should be determined by variable creation order
    # Creation order: x, y, vl, vr, z[0:2], w[0:2], zl[0:2], zr[0:2].
    expected_pairs = [[2, 3], [8, 10], [9, 11]]
    assert _get_cc_pairs(opti) == expected_pairs
    adv = opti.advanced
    for var, indices in [(vl, [2]), (vr, [3]), (zl, [8, 9]), (zr, [10, 11])]:
        found = [i for i in range(opti.nx) if ca.is_equal(adv.x_lookup(i), var)]
        assert found == indices

    # Complementarity adds only equalities; nonnegativity is supplied separately through lbx.
    # Validate constraints with test values (TODO: do this symbolically instead?)
    values = np.arange(1, 13, dtype=float)
    g = ca.Function("constraints", [opti.x], [opti.g])(values)
    expected_g = [3 - (1 + 1), 4 - 2, 9 - 5, 10 - 6, 11 - 2 * 7, 12 - 2 * 8]
    np.testing.assert_allclose(np.array(g).ravel(), expected_g)
    np.testing.assert_array_equal(np.array(opti.value(opti.lbg)).ravel(), np.zeros(6))
    np.testing.assert_array_equal(np.array(opti.value(opti.ubg)).ravel(), np.zeros(6))

    # Test new problem with matrix variables and matrix complementarity (4 variables, 4 complementarities)
    matrix_opti = ca.Opti()
    matrix = matrix_opti.variable(2, 2)
    ml, mr = complementarity(matrix_opti, matrix, ca.DM([[1, 2], [3, 4]]))
    assert ml.shape == mr.shape == (4, 1)
    assert matrix_opti.ng == 8
    # Opti stores a constant right-hand side in the equality bounds.
    matrix_bounds = np.array(matrix_opti.value(matrix_opti.lbg)).ravel()
    np.testing.assert_array_equal(matrix_bounds[4:8], [1, 3, 2, 4])

    # Check empty and incorrect sizes
    empty_opti = ca.Opti()
    assert _get_cc_pairs(empty_opti) == []
    with pytest.raises(ValueError, match="same number of scalar elements"):
        complementarity(empty_opti, ca.DM.zeros(2), ca.DM.zeros(3))
    assert not hasattr(empty_opti, "_cc_vars")

@pytest.mark.parametrize(
    "left_var,right_var,expected_added",
    [
        (False, False, 2),
        (True,  False, 1),
        (False, True,  1),
        (True,  True,  0),
    ],
)
def test_slack_flags_control_auxiliary_variables(left_var, right_var, expected_added):
    opti = ca.Opti()

    x = opti.variable()
    y = opti.variable()
    opti.minimize(x**2 + y**2)

    nx_before = opti.nx

    complementarity(opti, x, y, left_var=left_var, right_var=right_var)

    assert opti.nx == nx_before + expected_added