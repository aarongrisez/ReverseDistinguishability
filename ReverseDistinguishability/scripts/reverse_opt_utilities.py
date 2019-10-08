import sympy
import sympy.abc
import numpy as np
import numba as nb
from scipy.linalg import sqrtm, inv, logm, expm
from sympy.physics.quantum import TensorProduct

"""
Contains variety of utility functions
"""

@nb.jit
def setUp2Qubits(r1, r2, theta):
    A = 1 / 2 * np.matrix([[1 + r1, 0], [0, 1 - r1]])
    B = 1 / 2 * np.matrix([[1 + r1 * np.sin(theta * np.pi / 360), r2 * np.cos(theta * np.pi / 360)],[r2 * np.cos(theta * np.pi / 360), 1 - r1 * np.sin(theta * np.pi / 360)]])
    return (A, B)

@nb.jit
def setUpN2QubitSystems(n, r1, r2, theta):
    """
    Creates density matrices for N copy pairs of qubits
    """
    A_not, B_not = setUp2Qubits(r1, r2, theta)
    A = A_not
    B = B_not
    if n != 1:
        for i in range(n-1):
            A = np.kron(A, A_not)
            B = np.kron(B, B_not)
    return (A, B)

def qChernoffInformation(r1, r2, theta, s=None):
    """
    Returns qcb for qubit Chernoff info as derived by Calsamiglia et al

    In the case of r1=r2, this expression is minimized when s = 1/2
    """
    lambda_0 = (1 + r1) / 2
    lambda_1 = (1 + r2) / 2
    if r1 == r2:
        return quantumChernoffInformation(lambda_0, lambda_1, theta, 1/2)
    else:
        svals = np.linspace(0, 1, 1000)
        return np.min(quantumChernoffInformation(lambda_0, lambda_1, theta, svals))
    if s != None:
        s = sympy.Symbol('s')
        return quantumChernoffInformation(lambda_0, lambda_1, theta, s)

def quantumChernoffInformation(lambda_0, lambda_1, theta, s):
    """
    Returns s-family of values of the Quantum Chernoff Information as derived by Calsamiglia et al. To find optimal qci, this expression must be minimized over s

    Parameters:
        -lambda_0: float, (0, 1)
            positive eigenvalue of the first qubit in Bloch Representation (WLOG, rotated to be aligned with z-axis)
        -lambda_1: float, (0, 1)
            positive eigenvalue of the second qubit in Bloch Representation

    Returns:
        -float
            Quantum Chernoff Information (qci) for the two qubits. In a hypothesis testing situation, the probability of error of discriminating
            between the two qubit states decreases ~ exp(-n * min(qci)) where n is the number of tests and the min is taken over all s in [0, 1].

            In general, s is a function of the lambdas and theta. For the case where lambda_0 = lambda_1, s = 1/2 achieves this minimum.
    """
    return ((lambda_0 ** s * lambda_1 ** (1 - s) +
            (1 - lambda_0) ** s * (1 - lambda_1) ** (1 - s)) * (np.cos(theta/2)) ** 2 +
            (lambda_0 ** s * (1 - lambda_1) ** (1 - s) +
            (1 - lambda_0) ** s * lambda_1 ** (1 - s)) * (np.sin(theta/2)) ** 2)

def scanFixedR1R2(ufunc, r1, r2, theta_steps=10, s_steps=1E4):
    """
    Takes ufunc and fixed values of r1 and r2 to scan the theta parameter 
    """
    svals = np.linspace(0, 1, s_steps)
    tvals = np.linspace(0, 180, theta_steps) #Theta values in degrees
    qc_info = np.zeros((int(theta_steps), 3))
    for i in range(theta_steps):
        temp = ufunc(r1, r2, tvals[i], svals)
        loc = temp.argmin() #Stores the position of the minimum value of s
        val = temp[loc]
        qc_info[i,0] = (i / theta_steps) * 360
        qc_info[i,1] = loc / s_steps
        qc_info[i,2] = val
    return qc_info #Return form: [(theta, s, Q_min), ...]

def thetaFourier(a, b):
    """
    with a as theta values and b as either s values or Q values, calculate the fft with periodicity in theta
    """
    transform = np.fft.fft(b)
    freq = np.fft.fftfreq(a.shape[-1])
    return (transform, freq)

def setUpNQudits(n, d):
    """
    Returns n density matrices as SymPy matrices and their traces (which must be normalized)
    """
    rho = [] #List of density matrices
    for i in range(n):
        char_num = 97 + i 
        density_matrix = sympy.Matrix(sympy.MatrixSymbol(chr(char_num), d, d))
        trace = density_matrix.trace()
        rho.append(density_matrix) 
    return rho

def setUpPOVMElements(d):
    """
    Returns 2 positive operators that correspond to Qudit systems and trace constraints on each
    """
    e = sympy.Matrix(sympy.MatrixSymbol('e', d, d))
    f = sympy.Matrix(sympy.MatrixSymbol('f', d, d))
    return (e, f)

def setUpTripartiteSystem(d):
    """
    Builds desired tripartite system of 3 qudits
    """
    rho = setUpNQudits(3, d)
    POVM = setUpPOVMElements(d)
    tau = TensorProduct(rho[0], rho[1], rho[2])
    M = TensorProduct(POVM[0], POVM[1], sympy.eye(d,d))
    return (M * tau).trace()
