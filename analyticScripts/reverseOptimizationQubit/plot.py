import matplotlib.pyplot as plt
import numpy as np
import glob

DATA_PATH='./Data/'
PLOT_PATH='./Graphs/'

def loadData():
    """
    Loads data from consolidated array, assumes only one consolidated file is in the directory!
    """
    datfile = glob.glob(DATA_PATH + 'consolidated.npy')
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
    for i in range(3, data.shape[1] - 1):
        rvals, tvals, fvals = getPlottingSequence(data, i)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        plt.title('N = ' + str(i-2))
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.scatter(tvals * np.pi / 180, rvals, c=(1 - fvals) * 80, s=2, cmap='gnuplot')
        plt.savefig(fname=PLOT_PATH + 'n_equals_' + str(i-2) + '.png', format='png')

if __name__ == "__main__":
    plot = main()
