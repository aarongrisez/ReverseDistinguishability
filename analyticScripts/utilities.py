import sympy
import sympy.abc
import numpy as np
import numba as nb
from scipy.linalg import sqrtm, inv, logm, expm
from sympy.physics.quantum import TensorProduct

"""
Symbolically Calculates tr(x^t*y^(1-t)) in integral representation without performing the actual integral
"""

def CBexp(a, b):
    """
    Takes 2 square matrices of entries, returns trace of the Chernoff Quantity
    """
    t = sympy.Symbol('t') #Parameter to be integrated over
    p = sympy.Symbol('p') #Power to be minimized over
    A = sympy.Matrix(a)
    B = sympy.Matrix(b)
    (dim, dim) = A.shape
    I = sympy.eye(dim)
    integrand1 = sympy.Matrix((t ** (1 / p) * I) + A)
    integrand2 = sympy.Matrix((t ** (1 / (1 - p)) * I) + B)
    total_integrand = A * integrand1 ** (-1) * B * integrand2 ** (-1)
    trace = sympy.Integer(0)
    for i in range(dim):
        trace += total_integrand[i,i]
    return trace

def testQubit(r1=None, r2=None, theta=None):
    """
    Takes parameters for a given 2 qubit system, constructs density matrix, and plots first and second derivatives of CB_exp
    """
    if r1 == None:
        r1 = sympy.Symbol("r_1")
    if r2 == None:
        r2 = sympy.Symbol("r_2")
    if theta == None:
        theta_sym = sympy.abc.theta
    A, B = setUp2Qubits(r1, r2, theta_sym)
    trace = CBexp(A,B)
    return trace

def setUp2QubitsSymbol(r1, r2, theta):
    """
    Returns density matrix representations of the two qubit states as sympy matrices
    """
    A = sympy.Rational(1 / 2) * sympy.Matrix([[1 + r1, 0], [0, 1 - r1]])
    B = sympy.Rational(1 / 2) * sympy.Matrix([[1 + r1 * sympy.sin(theta * np.pi / 360), r2 * sympy.cos(theta)],[r2 * sympy.cos(theta * np.pi / 360), 1 - r2 * sympy.sin(theta * np.pi / 360)]])
    return (A, B)

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
        for i in range(n):
            A = np.kron(A, A_not)
            B = np.kron(B, B_not)

    return (A, B)

@nb.jit
def compareMatsumotoLeifer(systems, r1, r2, theta_min=1.0E-30, theta_max=360, steps=1000):
    """
    Scans theta parameter and returns the Leifer and Matsumoto reverse fidelities for the two states

    r1 - Radius of first qubit
    r2 - Radius of second qubit
    systems - Number of copies of qubit systems
    """
    tvals = np.linspace(theta_min, theta_max, steps)
    matsumoto = np.zeros(steps)
    leifer = np.zeros(steps)
    for (i, j) in enumerate(tvals):
        states = setUpN2QubitSystems(systems, r1, r2, j)
        matsumoto[i] = matsumotoReverseFidelity(states[0],states[1])
        leifer[i] = leiferReverseFidelity(states[0],states[1])
    return (matsumoto, leifer)


def qChernoffInformation(r1, r2, theta, s):
    """
    Returns non-logarithmic, unminimized variety of q. chernoff info as derived by Calsamiglia et al

    Can be done ussympy.integrate(integrand, (t, 0, sympy.oo))ing SymPy if arguments are passed as symbols
    """
    lambda_0 = (1 + r1) / 2
    lambda_1 = (1 + r2) / 2
    return ((lambda_0 ** s * lambda_1 ** (1 - s) +
            (1 - lambda_0) ** s * (1 - lambda_1) ** (1 - s)) * (sympy.cos(theta)) ** 2 +
            (lambda_0 ** s * (1 - lambda_1) ** (1 - s) +
            (1 - lambda_0) ** s * lambda_1 ** (1 - s)) * (sympy.sin(theta)) ** 2)

def sDerivQChernoffInformation(r1=sympy.Symbol('r_1'), r2=sympy.Symbol('r_2'), theta=sympy.abc.theta, asUfunc = False):
    """
    Zeroth, first and second derivative expressions w.r.t. s of qChernoffInformation symbolically
    """
    s = sympy.Symbol('s')
    info = qChernoffInformation(r1, r2, theta, s)
    first = info.diff(s)
    second = first.diff(s)
    if asUfunc == False:
        return [info, first, second] #Option to return SymPy Expressions
    if asUfunc == True:
        from sympy.utilities.autowrap import ufuncify
        return [ufuncify([r1, r2, theta, s], info), ufuncify([r1, r2, theta, s], first), ufuncify([r1, r2, theta, s], second)]

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

@nb.jit
def matsumotoReverseFidelity(rho, sigma):
    """
    Returns Matsumoto's value for reverse fidelity, trace(rho*sqrt(rho**-1/2*sigma*rho**-1/2))
    """
    a = np.matrix(rho)
    b = np.matrix(sigma)
    c = np.matmul(inv(sqrtm(a)), b) #c = rho**(-1/2)*sigma
    d = np.matmul(c, inv(sqrtm(a))) #d = c*rho**(-1/2)
    e = np.matmul(a, sqrtm(d)) #e = a * sqrt(d)
    return np.trace(e)

@nb.jit
def leiferReverseFidelity(rho, sigma):
    """
    Returns Leifer's value for reverse fidelity, trace(exp(1/2 * (log(rho) + log(sigma))))
    """
    a = np.matrix(rho)
    b = np.matrix(sigma)
    c = logm(a)
    d = logm(b)
    e = expm(1/2 * (c + d))
    return np.trace(e)

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
