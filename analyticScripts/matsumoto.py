import numpy as np
import utilities as util

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
        states = util.setUpN2QubitSystems(systems, r1, r2, j)
        matsumoto[i] = matsumotoReverseFidelity(states[0],states[1])
        leifer[i] = leiferReverseFidelity(states[0],states[1])
    return (matsumoto, leifer)

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
    c = np.logm(a)
    d = np.logm(b)
    e = np.expm(1/2 * (c + d))
    return np.trace(e)
