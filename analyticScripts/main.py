import traceDistanceCVXOPT as opt
import numpy as np

def setUpParameterSpace(r_min=.1, r_max=.9, r_steps=5, t_min=.1, t_max=359, t_steps=10, num_threads=4):
    """
    Creates and partitions parameter space
    """
    parameter_points = r_steps * t_steps #Total number of points in parameter space to be tested
    threads = 
    pp_per_thread = parameter_points / n #Total number of parameter points for each thread
    rvals = np.linspace(r_min, r_max, r_steps)
    tvals = np.linspace(t_min, t_max, t_steps)
    parameters = np.meshgrid(rvals, tvals).reshape(-1, 2)
    partition = partitionParameterSpace(parameters, n)
    return partition

def partitionParameterSpace(parameters, n, pp_per_thread):
    """
    Returns partitioned numpy array where the first index provides the thread number
    """

def setUpFiles():
    """
    Creates folders
    """


def setUpProcesses(parameters, n):
    """
    Sets up threads for running script

    Takes the partitioned parameter space and the desired number of threads
    """

def process(params_list, depth):
    i = 0
    p_length = len(params_list)
    while i < p_length:
        chunk = []
        while len(chunk) < 100:
            chunk.append(calculate(params_list[i], depth))
            i += 1
        saveData(chunk)

def calculate(params_tuple, depth):
    """
    Creates sequence of data; parameters passed as (r, theta)
    """
    sequence = opt.mSequenceOptimizeLog(depth, params_tuple[0], params_tuple[0], params_tuple[1])
    return (params[0], params[0], params[1], sequence)

def saveData(chunk):
    """
    Saves data in chunks
    """
    print("Saving Chunk")

def execute(processes):
    """
    Executes processes
    """

if __name__ == "__main__":
    setUpFiles()
    params = setUpParameterSpace(r_min, r_max, r_steps, theta_min, theta_max, theta_steps, num_threads)
    processes = setUpProcesses(params, num_threads, depth)
    execute(processes)
