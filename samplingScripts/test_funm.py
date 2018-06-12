from scipy.linalg import funm
import pprint
import numpy as np
import utilities as util
import unittest
import numericalStabilityFunm as nsf
import matplotlib.pyplot as plt

"""
Tests the numerical stability of using scipy.linalg.funm to calculate fractional powers of a matrix
"""

class NumericalTestFunm(unittest.TestCase):
    
    def setUp(self):
        """
        Creates Hermetian test matrices, instantiates means, maxes, mins lists
        """
        n = 2 #Dimension of matrices to test
        l = 4 #Number of matrices to test
        m = 100 #Number of powers to test
        self.matrices = [util.randomHermetian(n) for i in range(l)]
        self.integers = np.random.randint(2, 9E7, m)
        self.mins = [[]]
        self.maxes = [[]]
        self.means = [[]]

    def test_inverse_integer(self):
        """
        Tests implementation for the case M**(1/p) where p is an integer
        """
        ###WORKING - I want to get some plots going to better visualize the data here. Need to debug that this is capturing the error for an approximation properly
        for matrix in self.matrices:
            for integer in self.integers:
                temp_max = []
                temp_min = []
                temp_mean = []
                difference = nsf.inverseIntegerTest(matrix, integer)
                temp_max.append(np.max(difference))
                temp_min.append(np.min(difference))
                temp_mean.append(np.mean(difference))
            self.maxes.append(temp_max)
            self.mins.append(temp_min)
            self.means.append(temp_mean)
        print(self.maxes)
        nsf.plotRowData(3, 1, "Max", self.maxes)
        nsf.plotRowData(3, 2, "Min", self.mins)
        nsf.plotRowData(3, 3, "Mean", self.means)
        plt.show()
        #print(self.maxes)
        #print(self.mins)
        #print(self.means)
        assert False

if __name__ == '__main__':
    unittest.main()
