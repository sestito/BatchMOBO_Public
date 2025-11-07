from ..TestFunction import TestFunction

import numpy as np


class DTLZ2(TestFunction):
    name = "DTLZ2"

    def __init__(self, number_of_parameters: int = 3, number_of_objectives: int = 2):
        if number_of_parameters <= number_of_objectives:
            raise Exception('The number of objectives must be less than the number of parameters')
        
        self.number_of_objectives = number_of_objectives
        self.number_of_parameters = number_of_parameters
        self.set_bounds()

    def set_bounds(self):
        '''
        Bounds must be saved to self.parameter_bounds
            First row is the lower bounds
            Second row is the upper bounds
        For the DTLZ1 problem, the lower bound is 0 and the upper bound is 1 for all parameters
        '''
        bounds =  np.zeros((2, self.number_of_parameters))
        bounds[1,:] = 1

        self.parameter_bounds = bounds   

    def __call__(self, x: np.ndarray):
        number_of_objectives = self.number_of_objectives
        n = x.shape[1]

        if n <= number_of_objectives:
            raise Exception('Number of objectives must be less than length of x vector')

        xm = x[:, (number_of_objectives-1):]

        g = np.sum((xm-0.5)**2, axis=1)
        g = g.reshape(-1,1)

        output = np.ones((x.shape[0], number_of_objectives))
        output *= (1 + g)

        for i in range(number_of_objectives):
            xprod = x[:,:number_of_objectives-(i+1)]
            xprod = np.cos(0.5*np.pi*xprod)
            output[:,i] *= np.prod(xprod, axis=1)
            if i != 0:
                output[:,i] *= np.sin(0.5*x[:, number_of_objectives-(i+1)]*np.pi)


        return output
