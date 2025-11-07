from ..TestFunction import TestFunction

import numpy as np
import warnings

class FON(TestFunction):
    name = "FON"
    '''
    Computes the FON test problem.

    :param x np.ndarray: x must be a 2d numpy array in the form of [number of samples, 3] where each value is [-4, 4]
    :return: A 2d numby array in the form of [number of samples, objectives] where there are two objectives.
    '''
    number_of_parameters = 3
    number_of_objectives = 2

    def __init__(self, number_of_parameters: int = 3):
        if number_of_parameters != 3:
            warnings.warn("FON only uses 3 parameters. Number of parameters was set to 3.")
        self.set_bounds()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != 3:
            raise Exception('FON requires 3 parameters.')

        return self.__f(x)

    def set_number_of_parameters(self, number_of_parameters: int) -> None:
        warnings.warn("FON only uses 3 parameters. Number of parameters was set to 3.")

    def set_bounds(self) -> None:
        '''
        Bounds must be saved to self.parameter_bounds
            First row is the lower bounds
            Second row is the upper bounds

        For the FON problem, the lower bound is -4 and the upper bound is 4 for all parameters
        '''

        bounds =  np.zeros((2, self.number_of_parameters))
        bounds[0,:] = -4
        bounds[1,:] = 4

        self.parameter_bounds = bounds

    def __f(self, x):
        output = np.zeros((x.shape[0], 2))
        output[:,0] = self.__f1(x)
        output[:,1] = self.__f2(x)
        return output

    def __f1(self, x):

        exponent = (x - (1 / (3**0.5)))
        exponent =  exponent ** 2
        exponent = np.sum(exponent, axis = 1)
        return 1 - np.exp(-1*exponent)

    def __f2(self, x):
        exponent = (x + (1 / (3**0.5)))
        exponent =  exponent ** 2
        exponent = np.sum(exponent, axis = 1)
        return 1 - np.exp(-1*exponent)

