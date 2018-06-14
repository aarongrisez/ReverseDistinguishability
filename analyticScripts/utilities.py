import sympy
import sympy.abc

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
    A, B = setUpQubit(r1, r2, theta_sym)
    trace = CBexp(A,B)
    return trace

def setUpQubit(r1, r2, theta):
    """
    Returns density matrix representations of the two qubit states
    """
    A = sympy.Rational(1 / 2) * sympy.Matrix([[1 + r1, 0], [0, 1 - r1]])
    B = sympy.Rational(1 / 2) * sympy.Matrix([[1 + r1 * sympy.sin(theta), r2 * sympy.cos(theta)],[r2 * sympy.cos(theta), 1 - r2 * sympy.sin(theta)]])
    return (A, B)
