import numpy as np
from scipy.optimize import shgo

def qcb(r1, r2, theta):
    """
    Returns qcb for qubit Chernoff info as derived by Calsamiglia et al
    In the case of r1=r2, this expression is minimized when s = 1/2
    """
    lambda_0 = (1 + r1) / 2
    lambda_1 = (1 + r2) / 2
    x = quantumChernoffInformationFamily(lambda_0, lambda_1, theta)
    return shgo(x, ((0,1),))

def quantumChernoffInformationFamily(lambda_0, lambda_1, theta):
    """
    Returns s-family of values of the Quantum Chernoff Information as derived by Calsamiglia et al. To find optimal qci, this expression must be minimized over s
    Parameters:
        -lambda_0: float, (0, 1)
            positive eigenvalue of the first qubit in Bloch Representation (WLOG, rotated to be aligned with z-axis)
        -lambda_1: float, (0, 1)
            positive eigenvalue of the second qubit in Bloch Representation
    Returns:
        -callable (s parameter)
            Quantum Chernoff Information (qci) for the two qubits. In a hypothesis testing situation, the probability of error of discriminating
            between the two qubit states decreases ~ exp(-n * min(qci)) where n is the number of tests and the min is taken over all s in [0, 1].
            In general, s is a function of the lambdas and theta. For the case where lambda_0 = lambda_1, s = 1/2 achieves this minimum.
    """
    def _(s):
        return np.log(((lambda_0 ** s * lambda_1 ** (1 - s) +
                (1 - lambda_0) ** s * (1 - lambda_1) ** (1 - s)) * (np.cos(theta/2)) ** 2 +
                (lambda_0 ** s * (1 - lambda_1) ** (1 - s) +
                (1 - lambda_0) ** s * lambda_1 ** (1 - s)) * (np.sin(theta/2)) ** 2))
    return _