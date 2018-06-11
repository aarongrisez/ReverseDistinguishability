import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import funm

def plotRowData(n, i, title, data):
    """
    n - num of subplots
    i - this row
    title - title for this row
    data - row of data
    """
    x = plt.subplot(n, 1, i)
    x.plot(data)
    x.title(title + "; Largest " + title + "is : " + str(np.max(data)))
    return x

def inverseIntegerTest(matrix, integer):
    """
    Returns (matrix - (matrix ** (1/integer)) ** integer) elementwise norm 
    """
    approximation = funm(matrix, lambda x: x ** (1.0 / integer))
    rebuilt = approximation ** integer
    return np.absolute(matrix - rebuilt)
