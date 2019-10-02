import pytest
import numpy as np
from numpy.linalg import inv
import ReverseDistinguishability.helpers as h

@pytest.fixture()
def a_hermetian_matrix():
    return np.array([
        [1, 1 + 2j],
        [1 - 2j, 1]
    ])

def test_diagonalize_matrix(a_hermetian_matrix):
    test_diag, test_unitary = h.diagonalize_matrix(a_hermetian_matrix)
    test_unitary_inverse = inv(test_unitary)
    np.testing.assert_allclose((test_unitary @ test_diag @ test_unitary_inverse), a_hermetian_matrix, rtol=1e-15)
