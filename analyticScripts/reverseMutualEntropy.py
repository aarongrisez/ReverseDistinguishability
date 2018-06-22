import numpy as np
import utilities as util
from scipy.linalg import sqrtm, logm, inv

def quantity(n, r1, r2, theta):
    """
    For a given pair of qubit states, calculate Matsumoto's reverse relative entropy
    """
    rho, sigma = util.setUpN2QubitSystems(n, r1, r2, theta) 
    a = np.matmul(inv(sigma), sqrtm(rho))
    b = np.matmul(sqrtm(rho, a))
    c = np.matmul(rho, logm(b))
    return np.trace(c)
