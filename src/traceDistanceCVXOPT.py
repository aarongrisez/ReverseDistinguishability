import cvxpy as cvx
import numpy as np
import utilities
import numba as nb
import psutil
import time

@nb.jit
def optimize(p):
    """
    Runs solver for cvxpy
    """
    return p.solve(warm_start=True)

def setUpProblem(dim, r1, r2, theta):
    W = cvx.Variable(shape=(dim, dim), PSD=True)
    rho = cvx.Parameter(shape=(dim, dim), name='rho', PSD=True)
    sigma = cvx.Parameter(shape=(dim, dim), name='sigma', PSD=True)
    rho, sigma = utilities.setUpN2QubitSystems(1, r1, r2, theta, dim)
    obj = cvx.Maximize(1/2 * cvx.trace(W))
    constraints = [rho - W >= 0, sigma - W >=0]
    p = cvx.Problem(obj, constraints) 
    return (rho, sigma, p)

@nb.jit
def mSequenceOptimize(m, r1, r2, theta, l=0):
    """
    For a given qubit pair, get sequence of optimizations for n<=m
    Tested for m <= 9, runtime reasonable
    l is starting point
    """
    sequence = np.zeros(m-1-l)
    dim = 2 ** m
    rho, sigma, p = setUpProblem(dim, r1, r2, theta)
    for (j,i) in enumerate(range(1+l, m)):
        rho, sigma = utilities.setUpN2QubitSystems(i, r1, r2, theta, m)
        sequence[j] = optimize(p)
    return sequence

@nb.jit
def mSequenceOptimizeLog(m, r1, r2, theta):
    """
    For a given qubit pair, get sequence of optimizations for n<=m
    Tested for m <= 9, runtime reasonable
    """
    sequence = np.zeros(m-1)
    for i in range(1, m):
        sequence[i-1] = -np.log(optimize(i, r1, r2, theta)) / i
    return sequence

@nb.jit
def scanThetaOptimize(max_systems, r1, r2, theta_min, theta_max, steps):
    """
    Scans theta parameter and returns the mSequence of convex optimization problem solutions for the two states

    r1 - Radius of first qubit
    r2 - Radius of second qubit
    max_systems - Number of copies of qubit systems
    """
    tvals = np.linspace(theta_min, theta_max, steps)
    solutions = np.zeros((steps, max_systems-1))
    for (i, j) in enumerate(tvals):
        solutions[i,:] = mSequenceOptimize(max_systems, r1, r2, j)
    return [tvals, solutions]
