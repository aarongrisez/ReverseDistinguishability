import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from scipy.linalg import inv, sqrtm, logm

DATA_PATH = './Data/'

def getRandomChunk():
    files = glob.glob(DATA_PATH + 'chunk*')
    return random.choice(files)

def compareQcbRqcb(chunk=None):
    """
    From chunk filename, plot data versus n and the analytic qcb, if no chunk given, random one will be selected
    """
    if chunk==None:
        chunk = getRandomChunk()
    array = np.load(chunk)
    r1 = array[0]
    r2 = array[1]
    theta = array[2]
    sequence = array[3:array.size]
    analytic = qcb(r1, r2, theta)
    rre = mrre(r1, r2, theta, sequence.size)
    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.plot(sequence)
    ax.plot(rre)
    ax.hlines(analytic, 0, sequence.size)
    return fig

def qcb(r1, r2, theta, s=None):
    """
    Returns qcb for qubit Chernoff info as derived by Calsamiglia et al

    In the case of r1=r2, this expression is minimized when s = 1/2
    """
    lambda_0 = (1 + r1) / 2
    lambda_1 = (1 + r2) / 2
    if r1 == r2:
        return quantumChernoffInformation(lambda_0, lambda_1, theta, 1/2)
    else:
        svals = np.linspace(0, 1, 10000)
        return np.min(quantumChernoffInformation(lambda_0, lambda_1, theta, svals))

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
            
def mrre(r1, r2, theta, n):
    """
    Returns Matsumoto's reverse relative entropy for sequences of pairs of states
    """
    sequence = np.zeros(n)
    for i in range(1, n+1):
        rho, sigma = setUpN2QubitSystems(i, r1, r2, theta, i)
        sequence[i-1] = np.log(quantity(rho, sigma)) / i
    return sequence

def quantity(rho1, rho2):
    """
    For a given pair of qubit states, calculate Matsumoto's reverse relative entropy
    """
    a = np.matmul(inv(rho2), sqrtm(rho1))
    b = np.matmul(sqrtm(rho1), a)
    c = np.matmul(rho1, logm(b))
    return np.trace(c)

def setUp2Qubits(r1, r2, theta):
    A = 1 / 2 * np.matrix([[1 + r1, 0], [0, 1 - r1]])
    B = 1 / 2 * np.matrix([[1 + r1 * np.cos(theta * np.pi / 180), r2 * np.sin(theta * np.pi / 180)],[r2 * np.sin(theta * np.pi / 180), 1 - r1 * np.cos(theta * np.pi / 180)]])
    return (A, B)

def setUpN2QubitSystems(n, r1, r2, theta, m=1):
    """
    Creates density matrices for N copy pairs of qubits
    """
    A_not, B_not = setUp2Qubits(r1, r2, theta)
    A = A_not
    B = B_not
    zeroz = np.zeros((2,2))
    zeroz[0,0] = 1
    if n >= 1:
        for i in range(n-1):
            A = np.kron(A, A_not)
            B = np.kron(B, B_not)
    if m > n:
        for i in range(m-n):
            A = np.kron(zeroz, A)
            B = np.kron(zeroz, B)
    dim = 2 ** m #Dimension of matrices returned
    return (np.matrix(A.reshape((dim,dim))), np.matrix(B.reshape((dim,dim))))
