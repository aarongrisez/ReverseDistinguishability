import matplotlib.pyplot as plt
import numpy as np
import glob

DATA_PATH='./Data/'

def loadData():
    """
    Loads data from consolidated array, assumes only one consolidated file is in the directory!
    """
    datfile = glob.glob(DATA_PATH + 'consolidated_*.npy')
    return np.load(datfile[0])

def getPlottingSequence(data, column, dtype='Polar'):
    """
    Extracts a sequence for plotting from consolidated data
    """
    rvals = data[:,0]
    tvals = data[:,2]
    fvals = data[:,column]
    return (rvals, tvals, fvals)

def main():
    data = loadData()
    last_column = data.shape[1] - 1
    rvals, tvals, fvals = getPlottingSequence(data, last_column)
    print(rvals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    c = ax.scatter(tvals, rvals, c=fvals, s=2, cmap='hsv')
    return c

if __name__ == "__main__":
    plot = main()
    plt.show()
