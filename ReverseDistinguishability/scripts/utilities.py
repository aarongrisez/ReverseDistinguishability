import numpy as np

"""
Defines utility functions for Reverse Distinguishability
"""

def randomPositiveHermetian(n):
    """
    Parameters:
        n - int
            Dimension of Hermetian matrix
    Returns: 3 matrices
        1st - Random Positive Semidefinite Hermetian matrix
        2nd - Matrix with Eigenvalues on diagonal
        3rd - Diagonalizing Unitary

    To Do:
        Ensure that H is positive semidefinite
    """
    ###Generate 2 Random Matrices, one for Real components, one for Im components
    random1 = np.random.rand(n,n)
    random2 = 1j * np.random.rand(n,n)
    random_complex = np.matrix(random1 + random2)
    ###Symmetrize the matrix
    symmetric = (random_complex + random_complex.getH())
    print(symmetric)
    ###Normalize the trace of the matrix
    trace = np.trace(symmetric)
    normalized = symmetric / trace
    ###Diagonalize the matrix
    (eigenvalues, unitary) = np.linalg.eigh(normalized)
    print(eigenvalues)
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
        x - h ** p

    Procedure
        1. Scale the matrix H eigenvalues element-wise on the diagonalized matrix
        2. Calculate the inverse of the diagonalizing unitary: U^-1
        3. Return UHU^-1

    To Do
        Provide alternative implementation if H is not given with its diagonal version and diagonalizing unitary
    """
    scaled_diagonal = d ** p
    print(d)
    print(scaled_diagonal)
    inverse = np.linalg.inv(u)
    temp = np.matmul(d, inverse)
    return np.matmul(u, temp)
