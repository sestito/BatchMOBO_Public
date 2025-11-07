from ..TestFunction import TestFunction

'''
Given input x = [x1, x2, x3, ..., xn]
Optimum occurs for any value of 0 <= x1 <= 1 and xi = 0 for i = 2, ..., n
'''

import numpy as np

# Class inherits from TestFunction
class ZDT1(TestFunction):
    name = "ZDT1"
    def __call__(self, x):
        return self.__f(x)

    def set_bounds(self) -> None:
        '''
        Bounds must be saved to self.parameter_bounds
            First row is the lower bounds
            Second row is the upper bounds

        For the ZDT1 problem, the lower bound is 0 and the upper bound is 1 for all parameters
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
        return 1 + 9*np.sum(x[:, 1:], axis = 1)/(n-1)

    def __h(self, x):
        return 1 - np.power(self.__f1(x)/self.__g(x), (1/2))

    def __f1(self, x):
        return x[:, 0] # First column, or first value in every row

    def __f2(self, x):
        return self.__g(x)*self.__h(x)

    def plot_data(self, number_of_data_points = 10000) -> np.ndarray:
        # Initialize output as a (n x 2) matrix
        output = np.zeros((number_of_data_points, 2))

        # Pareto front is when f1 = 1 - f2 ^ 2
        output[:, 0] = np.linspace(0, 1, number_of_data_points)
        output[:, 1] = 1 - np.power(output[:, 0], 0.5)

        return output



'''
def ZDT1(x):

    def f(x):
        output = np.zeros((x.shape[0], 2))
        output[:,0] = __f1(x)
        output[:,1] = __f2(x)
        return output
    
    def __g(x):
        n = x.shape[1]
        return 1 + 9*np.sum(x[:, 1:], axis = 1)/(n-1)

    def __h(x):
        return 1 - np.power(__f1(x)/__g(x), (1/2))

    def __f1(x):
        return x[:, 0] # First column, or first value in every row

    def __f2(x):
        return __g(x)*__h(x)

    return f(x)
'''
'''
class ZDT1:

    def __call__(self, x):
        return self.f(x)     

    def f(self, x):
        output = np.zeros((x.shape[0], 2))
        output[:,0] = self.__f1(x)
        output[:,1] = self.__f2(x)
        return output
    
    def __g(self, x):
        n = x.shape[1]
        return 1 + 9*np.sum(x[:, 1:], axis = 1)/(n-1)

    def __h(self, x):
        return 1 - np.power(self.__f1(x)/self.__g(x), (1/2))

    def __f1(self, x):
        return x[:, 0] # First column, or first value in every row

    def __f2(self, x):
        return self.__g(x)*self.__h(x)
'''