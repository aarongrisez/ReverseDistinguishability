from scipy.linalg import funm
import numpy as np
import utilities as util
import unittest

"""
Tests the numerical stability of using scipy.linalg.funm to calculate fractional powers of a matrix
"""

class NumericalTestFunm(unittest.TestCase):
    
    def setUp(self):
        """
        Creates Hermetian test matrices, instantiates means, maxes, mins lists
        """
        n = 2 #Dimension of matrices to test
        l = 1 #Number of matrices to test
        m = 1 #Number of powers to test
        self.matrices = [util.randomHermetian(n) for i in range(l)]
        self.integers = [2,3] #np.random.randint(2, 9E7, m)
        self.mins = []
        self.maxes = []
        self.means = []
        pass

    def test_inverse_integer(self):
        """
        Tests implementation for the case M**(1/p) where p is an integer
        """
        ###WORKING - I want to get some plots going to better visualize the data here. Need to debug that this is capturing the error for an approximation properly
        for i in self.matrices:
            for j in self.integers:
                print(i)
                approximation = funm(i, lambda x: x ** (1.0 / j))
                print(approximation)
                rebuilt = approximation ** j #This line simply inverts the prior step recreating i
                difference = i - rebuilt
                print(difference)
                self.maxes.append(np.max(difference))
                self.mins.append(np.min(difference))
                self.means.append(np.mean(difference))
        #print(self.maxes)
        #print(self.mins)
        #print(self.means)
        assert False

if __name__ == '__main__':
    unittest.main()
