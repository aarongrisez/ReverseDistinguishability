import numpy as np
import cvxpy as cvx
import numpy as np

def setUpProblem(dim, r1, r2, theta, m):
    W = cvx.Variable(shape=(dim, dim), PSD=True)
    rho = cvx.Parameter(shape=(dim, dim), name='rho', PSD=True)
    sigma = cvx.Parameter(shape=(dim, dim), name='sigma', PSD=True)
    obj = cvx.Maximize(1/2 * cvx.trace(W))
    rho, sigma = setUpN2QubitSystems(m, r1, r2, theta, m)
    constraints = [rho - W >> 0, sigma - W >>0]
    p = cvx.Problem(obj, constraints) 
    return (rho, sigma, p)

def calculate(params_tuple, depth):
    """
    Creates sequence of data; parameters passed as (r, theta)
    """
    sequence = np.zeros(depth)
    r = params_tuple[0]
    t = params_tuple[1]
    for a in range(1, depth + 1):
        rho, sigma, p = setUpProblem(2**a, r, r, t, a)
        x = -1 / a * np.log(p.solve())
        sequence[a - 1] = x
    return sequence

def calculateSingleN(params_tuple, n):
    """
    Creates sequence of data; parameters passed as (r, theta)
    """
    r = params_tuple[0]
    t = params_tuple[1]
    rho, sigma, p = setUpProblem(2**a, r, r, t, a)
    x = -1 / a * np.log(p.solve())
    return sequence

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
