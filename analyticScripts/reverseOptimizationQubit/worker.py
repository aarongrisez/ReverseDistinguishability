import numpy as np
import traceDistanceCVXOPT as opt
import hashlib
import os
import logging

def workerFunction(filename, depth, DATA_PATH, i):
    logging.info('Worker ' + str(i) + ' started on ' + filename)
    a = loadParameters(filename)
    if a.size == 0:
        logging.info('Encountered empty param file, ignoring...') 
    else:
        process(a, depth, DATA_PATH, i)
    os.remove(filename)
    logging.info('Worker ' + str(i) + ' Done')

def loadParameters(filename):
    return np.load(filename)

def process(params_list, depth, DATA_PATH, l):
    chunk = np.zeros(depth+3)
    for j in params_list:
        chunk[0] = j[0]
        chunk[1] = j[0]
        chunk[2] = j[1]
        chunk[3:depth+3] = calculate(j, depth)
        saveData(np.array(chunk), DATA_PATH, l)

def calculate(params_tuple, depth):
    """
    Creates sequence of data; parameters passed as (r, theta)
    """
    depth = depth
    sequence = np.zeros(depth)
    r = params_tuple[0]
    t = params_tuple[1]
    for a in range(1, depth + 1):
        rho, sigma, p = opt.setUpProblem(2**a, r, r, t, a)
        x = -1 / a * np.log(p.solve())
        sequence[a - 1] = x
    return sequence

def saveData(chunk, DATA_PATH, i):
    """
    Saves data in chunks
    """
    #Write to log "saving Chunk", time, and system spec
    #Save to file
    ID = hashlib.sha1()
    ID.update(np.array2string(chunk).encode('utf-8'))
    chunk_file = DATA_PATH + 'chunk' + '_' + ID.hexdigest() + '.npy'
    np.save(chunk_file, chunk)
