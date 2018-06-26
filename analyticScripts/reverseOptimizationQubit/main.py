import traceDistanceCVXOPT as opt
import hashlib
from tempfile import TemporaryFile
import numpy as np

DATA_PATH = './Data/'
LOG_PATH = './Log/'

def setUpParameterSpace(r_min=.1, r_max=.9, r_steps=5, t_min=.1, t_max=359, t_steps=10, num_threads=4):
    """
    Creates and partitions parameter space
    """
    parameter_points = r_steps * t_steps #Total number of points in parameter space to be tested
    pp_per_thread = parameter_points / num_threads #Total number of parameter points for each thread
    rvals = np.linspace(r_min, r_max, r_steps)
    tvals = np.linspace(t_min, t_max, t_steps)
    parameters = np.dstack(np.array(np.meshgrid(rvals, tvals))).reshape(-1,2)
    partition = partitionParameterSpace(parameters, num_threads, pp_per_thread)
    return partition

def partitionParameterSpace(parameters, num_threads, pp_per_thread):
    """
    Returns partitioned numpy array where the first index provides the thread number
    """
    main_block_size = (num_threads - 1) * pp_per_thread #Calculates the main block size for the parameter space, 1 thread reserved for overflow
    overflow = len(parameters) - main_block_size
    print(parameters)
    main_block = parameters[0:main_block_size].reshape(num_threads, pp_per_thread, 2)
    return parameters

def setUpFileSystem():
    """
    Creates folders
    """
    pass

def setUpProcesses(parameters, num_threads):
    """
    Sets up threads for running script

    Takes the partitioned parameter space and the desired number of threads, assigns each thread a segment of the parameter space
    """
    pass

def process(params_list, depth):
    i = 0
    p_length = len(params_list)
    while i < p_length:
        chunk = []
        while len(chunk) < 50:
            chunk.append(calculate(params_list[i], depth))
            i += 1
        saveData(np.array(chunk))

def calculate(params_tuple, depth):
    """
    Creates sequence of data; parameters passed as (r, theta)
    """
    sequence = np.zeros(depth + 3)
    sequence[0:3] = np.array([params_tuple[0], params_tuple[0], params_tuple[1]])
    sequence[4:depth+3] = opt.mSequenceOptimizeLog(depth, params_tuple[0], params_tuple[0], params_tuple[1])
    return sequence

def saveData(chunk):
    """
    Saves data in chunks
    """
    #Write to log "saving Chunk", time, and system specs
    #Save to file
    ID = hashlib.sha1()
    ID.update(np.array2string(chunk).encode('utf-8'))
    np.save(DATA_PATH + 'chunk' + ID.hexdigest() + '.npy', chunk)

def execute(processes):
    """
    Executes processes
    """
    pass

if __name__ == "__main__":
    params = setUpParameterSpace()
    process(params, 4)
