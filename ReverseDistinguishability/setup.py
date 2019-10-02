import hashlib
import numpy as np
import glob
import worker
from argparse import ArgumentParser
import time
import logging

"""
Set Up filestructure, parameters files and logging
"""

DATA_PATH = './Data/'
PARAM_PATH = './Params/'
LOG_PATH = './Log/'
logfile = LOG_PATH + 'log' + time.strftime('%b%a%I_%M_%S') + '.log'
logging.basicConfig(filename=logfile,
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def setUpParameterSpace(r_min, r_max, r_steps, t_min, t_max, t_steps, num_processes):
    """
    Creates and partitions parameter space, returns main block AND overflow
    """
    logging.info("Creating Parameter Space...")
    rvals = 1 - np.linspace(r_min, r_max, r_steps) ** 2
    tvals = np.linspace(t_min, t_max, t_steps)
    parameters = np.array(np.dstack(np.array(np.meshgrid(rvals, tvals))).reshape(-1,2))
    partition = partitionParameterSpace(parameters, num_processes)
    logging.info("Creating Parameter Space...success")
    logging.info("Parameter space has " + str(int(parameters.size)) + " elements")
    return partition

def partitionParameterSpace(parameters, num_processes):
    """
    Returns partitioned numpy array where the first index provides the process number
    """
    logging.info("Partitioning Parameter Space...")
    pp_per_process = np.floor(parameters.shape[0] / (num_processes - 1)) #Total number of parameter points for each thread
    main_block_size = int((num_processes - 1) * pp_per_process) #Calculates the main block size for the parameter space, 1 thread reserved for overflow
    overflow_len = parameters.shape[0] - main_block_size
    main_block = parameters[0:main_block_size]
    main_block_shaped = main_block.reshape(num_processes-1,int(pp_per_process),2)
    overflow = parameters[int(main_block_size):parameters.shape[0]] 
    logging.info("Partitioning Parameter Space...success")
    return main_block_shaped, overflow

def paramsToFile(params, overflow, num_processes):
    """
    Takes partitioned parameter space and saves individual partitions as .npy instances
    """
    logging.info("Writing Parameters to file...")
    for i in range(num_processes-1):
        param_chunk = params[i]
        ID = hashlib.sha1()
        ID.update(np.array2string(param_chunk).encode('utf-8'))
        np.save(PARAM_PATH + 'param_chunk' + ID.hexdigest() + '.npy', param_chunk)
    param_chunk = overflow
    ID = hashlib.sha1()
    ID.update(np.array2string(param_chunk).encode('utf-8'))
    np.save(PARAM_PATH + 'param_chunk' + ID.hexdigest() + '.npy', param_chunk)
    logging.info("Writing Parameters to file...success")

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
    parser.add_argument('-fs', help='num of param files to use, int',
                        type=int, default=3)
    parser.add_argument('-profile', help='whether to profile the script or not, bool',
                        type=bool, default=False)
    logging.info('Beginning setup')
    a = parser.parse_args()
    params = setUpParameterSpace(r_min = a.ri,
                                 r_max = a.ra,
                                 r_steps = a.rs,
                                 t_min = a.ti,
                                 t_max = a.ta,
                                 t_steps = a.ts,
                                 num_processes = a.fs)
    paramsToFile(params[0], params[1], a.fs)
