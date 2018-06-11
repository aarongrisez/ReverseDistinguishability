import numpy as np

"""
Defines utility functions for Reverse Distinguishability
"""

def randomHermetian(n):
    """
    Parameters:
        n - int
            Dimension of Hermetian matrix
    Returns: 3 matrices
        1st - Random Hermetian
        2nd - Matrix with Eigenvalues on diagonal
        3rd - Diagonalizing Unitary
    """
    ###Generate 2 Random Matrices, one for Real components, one for Im components
    random1 = np.random.rand(n,n)
    random2 = 1j * np.random.rand(n,n)
    random_complex = np.matrix(random1 + random2)
    ###Symmetrize the matrix
    symmetric = (random_complex + random_complex.getH())
    ###Normalize the trace of the matrix
    trace = np.trace(symmetric)
    normalized = symmetric / trace
    ###Diagonalize the matrix
    (eigenvalues, unitary) = np.linalg.eigh(normalized)
    diagonal = np.eye(n) * eigenvalues
    return (normalized, diagonal, unitary)


def realPower(h, d, u, p):
    """
    Parameters
        h - Hermetian Matrix object
        d - Diagonalized Hermetian Matrix
        u - Diagonalizing Unitary
        p - real number
    Returns
        x - 
    """
    scaled_diagonal = d ** p
    print(scaled_diagonal)
    inverse = np.linalg.inv(u)
    temp = np.matmul(d, inverse)
    return np.matmul(u, temp)
