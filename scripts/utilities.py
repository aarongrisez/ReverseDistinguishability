import numpy as np

"""
Defines utility functions for Reverse Distinguishability
"""

def randomHermetian(n):
    """
    Parameters
        n - int
            Dimension of Hermetian matrix
    """
    random1 = np.random.rand(n,n)
    random2 = 1j * np.random.rand(n,n)
    random_complex = random1 + random2
    trace = np.trace(random_complex)
    normalized = np.matrix(random_complex / trace)
    return 1.0 / 2.0 * (normalized + normalized.getH())
