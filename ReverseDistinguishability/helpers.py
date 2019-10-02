import numpy as np
from numpy import linalg as la

def diagonalize_matrix(rho):
    """Diagonalizes a density matrix and returns the diagonal matrix of eigenvalues
    and the diagonalizing unitary as a tuple
    """
    eigval, eigvec = la.eig(rho)
    rho_diag = np.diag(eigval)
    unitary = np.array(eigvec)
    return rho_diag, unitary

def get_qubit_parameter_space():
    """
    Creates a uniform mesh 
    """
    logging.info("Creating Parameter Space...")
    rvals = 1 - np.linspace(r_min, r_max, r_steps) ** 2
    tvals = np.linspace(t_min, t_max, t_steps)
    parameters = np.array(np.dstack(np.array(np.meshgrid(rvals, tvals))).reshape(-1,2))
