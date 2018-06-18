import cvxopt
import utilities

def optimization(n, r1, r2, theta):
    """
    Returns max(tr(W)) st. rho-W>0 and sigma-W>0
    """
    c = cvxopt.matrix([1. for i in range(n)])
    rho, sigma = utilities.setUpN2QubitSystems(n, r1, r2, theta)
    #G = [cvxopt.matrix([[1.] for i in range(n*4)]) for i in range(2)]
    h = [cvxopt.matrix(rho), cvxopt.matrix(sigma)]
    sol = cvxopt.solvers.sdp(c, Hl=h)
