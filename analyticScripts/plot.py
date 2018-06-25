import matplotlib.pyplot as plt
import traceDistanceCVXOPT as opt
import utilities

def plotMSequenceOptimize(m, r1, r2, theta, logarithm=False):
    """
    For a given qubit pair, plot mSequence
    """
    n_vals = [i for i in range(1, m)]
    if logarithm == False:
        sequence = opt.mSequenceOptimize(m, r1, r2, theta)
    else:
        sequence = opt.mSequenceOptimizeLog(m, r1, r2, theta)
        qcb = utilities.qChernoffInformation(r1, r2, theta, 1/2)
        plt.hlines(qcb, 0, m)
    plt.scatter(n_vals, sequence, c='black', s=.6)
    
def plotMSequenceOptimizeFromData(data):
    """
    For a given qubit pair, plot mSequence
    """
    n_vals = [i for i in range(1, m)]
    sequence = opt.mSequenceOptimizeLog(m, r1, r2, theta)
    if r1 == r2:
        #This next line makes the assumption that r1=r2 to give minimum value of s for QCB
        qcb = utilities.qChernoffInformation(r1, r2, theta, 1/2)
    else:
        qcb = utilities.minimizeQChernoffInformation(r1, r2, theta)
    plt.hlines(qcb, 0, m)
    plt.scatter(n_vals, sequence, c='black', s=.6)

def plotScanThetaOptimize(max_systems, r1, r2, theta_min=1.0E-30, theta_max=180, steps=100):
    """
    Plots theta parameter scan for optimization problem
    """
    tvals, x = opt.scanThetaOptimize(max_systems, r1, r2, theta_min, theta_max, steps)
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            plt.scatter(tvals[j], x[j,i], c='black', s=.5)
