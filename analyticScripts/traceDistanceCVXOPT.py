import cvxpy as cvx
import numpy as np
import utilities
import numba as nb
import psutil
import time

@nb.jit
def optimize(n, r1, r2, theta):
    """
    For a given qubit pair, optimize Tr(W) where rho-W>0 and sigma-W>0
    """
    dim = 2 ** n
    W = cvx.Variable((dim, dim), PSD=True)
    zeros = np.matrix(np.zeros((dim, dim)))
    rho, sigma = utilities.setUpN2QubitSystems(n, r1, r2, theta)
    mem = psutil.virtual_memory()
    print("States Generated. n = " + str(n) + ": " + str(mem.used) + " used, " + str(mem.available) + " available")
    print("Time: " + time.asctime( time.localtime(time.time())))
    obj = cvx.Maximize(1/2 * cvx.trace(W))
    constraints = [rho - W >= 0, sigma - W >=0]
    mem = psutil.virtual_memory()
    print("Constraints set. : " + str(mem.used) + " used, " + str(mem.available) + " available")
    p = cvx.Problem(obj, constraints) 
    mem = psutil.virtual_memory()
    print("Problem set. : " + str(mem.used) + " used, " + str(mem.available) + " available")
    return p.solve()

@nb.jit
def mSequenceOptimize(m, r1, r2, theta, l):
    """
    For a given qubit pair, get sequence of optimizations for n<=m
    Tested for m <= 9, runtime reasonable
    l is starting point
    """
    sequence = np.zeros(m-1-l)
    for (j,i) in enumerate(range(1+l, m)):
        sequence[j] = optimize(i, r1, r2, theta)
        print("Time: " + time.asctime( time.localtime(time.time())))
        mem = psutil.virtual_memory()
        print("Optimum reached. : " + str(mem.used) + " used, " + str(mem.available) + " available")
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
