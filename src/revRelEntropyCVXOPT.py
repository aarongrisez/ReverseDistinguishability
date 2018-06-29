import numpy as np
import utilities
from scipy.linalg import sqrtm, logm, inv
import cvxpy as cvx
import numba as nb

@nb.jit
def quantity(rho1, rho2):
    """
    For a given pair of qubit states, calculate Matsumoto's reverse relative entropy
    """
    a = np.matmul(inv(rho2), sqrtm(rho1))
    b = np.matmul(sqrtm(rho1, a))
    c = np.matmul(rho1, logm(b))
    return np.trace(c)

@nb.jit
def optimize(n, r1, r2, theta, T):
    """
    For a given qubit pair, optimize Tr(W) where rho-W>0 and sigma-W>0
    """
    dim = 2 ** n
    rho_star = cvx.Variable((dim, dim), PSD=True)
    zeros = np.matrix(np.zeros((dim, dim)))
    rho, sigma = utilities.setUpN2QubitSystems(n, r1, r2, theta)
    D_rho_star_rho = quantity(np.matrix(rho_star), rho)
    D_rho_star_sigma = quantity(np.matrix(rho_star), sigma)
    obj = cvx.Minimize(D_rho_star_rho)
    constraints = [D_rho_star_rho - D_rho_star_sigma == 1.0 / n * T]
    p = cvx.Problem(obj, constraints) 
    return p.solve()
