import hashlib
from ReverseDistinguishability import paths
import numpy as np
import glob
import worker
from argparse import ArgumentParser
import time
import pandas as pd
import logging

"""
Set Up filestructure, parameters files and logging
"""

def setUpParameterSpace(r_min, r_max, r_steps, t_min, t_max, t_steps):
    """
    Creates parameter space as a dataframe
    """
    logging.info("Creating Parameter Space...")
    rvals = 1 - np.linspace(r_min, r_max, r_steps) ** 2
    tvals = np.linspace(t_min, t_max, t_steps)
    parameters = np.array(np.dstack(np.array(np.meshgrid(rvals, tvals))).reshape(-1,2))
    df = pd.DataFrame(parameters)
    logging.info("Creating Parameter Space...success")
    logging.info("Parameter space has " + str(int(parameters.size)) + " elements")
    return df

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-ri', help='lowest r value, float (0,1)',
                        type=float, default=0.1)
    parser.add_argument('-ra', help='highest r value, >rmin float (0,1)',
                        type=float, default=0.9)
    parser.add_argument('-rs', help='r mesh steps, int',
                        type=float, default=10)
    parser.add_argument('-ti', help='lowest t value, float (0,180)',
                        type=float, default=0.1)
    parser.add_argument('-ta', help='highest t value, >tmin float (0,180)',
                        type=float, default=179)
    parser.add_argument('-ts', help='t mesh steps, int',
                        type=int, default=10)
    logging.info('Beginning setup')
    a = parser.parse_args()
    params = setUpParameterSpace(r_min = a.ri,
                                 r_max = a.ra,
                                 r_steps = a.rs,
                                 t_min = a.ti,
                                 t_max = a.ta,
                                 t_steps = a.ts)
    params.to_csv(paths['params'] / 'params.csv')