import sympy as sym
from sympy import cos, sin, log
from sympy.physics.quantum import TensorProduct
from sympy.utilities.autowrap import ufuncify

def setUpN2QubitSystems(n):
    r1 = sym.Symbol('r_1')
    r2 = sym.Symbol('r_2')
    t = sym.Symbol('theta')
    rho = sym.Matrix([[1 + r1, 0], [0, 1 - r1]])
    sigma = sym.Matrix([[1 + r2 * cos(t), r2 * sin(t)],[r2 * sin(t), 1 - r2 * cos(t)]])
    rho_not = rho
    sigma_not = sigma
    for i in range(1, n):
        rho = TensorProduct(rho_not, rho)
        sigma = TensorProduct(sigma_not, sigma)
    return (rho, sigma, r1, r2, t)

def mrre(rho, sigma, n):
    """
    Returns symbolic version of Matsumoto's reverse relative entropy
    """
    a = rho ** (1/2) * sigma.inv() * rho ** (1/2)
    b = a.diagonalize()
    c = b[1].applyfunc(sym.log)
    temp = []
    for i in range(n+1):
        temp.append(c[i,i])
    z = rho * sym.diag(*temp)
    f = z.trace()
    return f

def generateUfunc(n):
    """
    Generates the dim=n ufunc for evaluating Matsumoto's expression
    """
    rho, sigma, r1, r2, t = setUpN2QubitSystems(n)
    expr = mrre(rho, sigma, n)
    ufuncify([r1, r2, t], expr, tmpdir='mrre_ufuncs')
