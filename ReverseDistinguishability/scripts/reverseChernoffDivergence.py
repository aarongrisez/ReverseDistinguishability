import numpy as np
import utilities as util

"""
Functions for looking at the quantity min(Trace(rho(rho**-1/2*sigma*rho**-1/2)))
"""

def quantity(n, r1, r2, theta, alpha):
    states = util.setUpN2QubitSystems(n, r1, r2, theta)
    a = 
