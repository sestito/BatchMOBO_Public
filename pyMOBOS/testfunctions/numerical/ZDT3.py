from ..TestFunction import TestFunction

'''
https://pymoo.org/problems/multi/zdt.html
'''

import numpy as np

class ZDT3(TestFunction):
    name = "ZDT3"

    def __call__(self, x):
        return self.__f(x)

    def set_bounds(self) -> None:
        '''
        Bounds must be saved to self.parameter_bounds
            First row is the lower bounds
            Second row is the upper bounds

        For the ZDT3 problem, the lower bound is 0 and the upper bound is 1 for all parameters
        '''

        bounds =  np.zeros((2, self.number_of_parameters))
        bounds[1,:] = 1

        self.parameter_bounds = bounds

    def __f(self, x):
        output = np.zeros((x.shape[0], 2))
        output[:,0] = self.__f1(x)
        output[:,1] = self.__f2(x)
        return output

    def __g(self, x):
        n = x.shape[1]
        return 1 + (9 / (n-1)) * np.sum(x[:, 1:], axis = 1)

    def __h(self, x):
        return 1 - np.power(self.__f1(x) / self.__g(x), 0.5) - (self.__f1(x) / self.__g(x)) * np.sin(10*np.pi*self.__f1(x))

    def __f1(self, x):
        return x[:, 0] # First column, or first value in every row

    def __f2(self, x):
        return self.__g(x)*self.__h(x)
 
    def plot_data(self, number_of_data_points = 10000) -> np.ndarray:
        part_number_of_data_points = int(number_of_data_points / 5)

        x1 = np.linspace(0.0000, 0.0830, part_number_of_data_points)
        x2 = np.linspace(0.1822, 0.2577, part_number_of_data_points)
        x3 = np.linspace(0.4093, 0.4538, part_number_of_data_points)
        x4 = np.linspace(0.6183, 0.6525, part_number_of_data_points)
        x5 = np.linspace(0.8233, 0.8518, part_number_of_data_points)

        # Initialize output as a (n x 2) matrix
        output = np.zeros((part_number_of_data_points*5, 2))

        

        X = np.zeros((part_number_of_data_points*5, 2))
        X[:, 0] = np.hstack((x1, x2, x3, x4, x5))

        output = self.__f(X)

        return output