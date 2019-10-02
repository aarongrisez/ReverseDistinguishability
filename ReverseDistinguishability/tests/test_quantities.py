import pytest
import numpy as np
import ReverseDistinguishability.quantities as q

def test_sanity_qcb():
    """In the case of 2 pure states, regardless of theta, quantity should reduce to cos ** 2 (theta)"""
    x = np.random.uniform(0, 2 * np.pi, 100)
    for i in x:
        np.testing.assert_almost_equal(q.qcb(1,1,i).fun, np.log(np.cos(i / 2) ** 2))