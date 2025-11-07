from ..TestFunction import TestFunction

import numpy as np

class DTLZ1(TestFunction):
    name = "DTLZ1"

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
        n = x.shape[1]

        if n <= self.number_of_objectives:
            raise Exception('Number of objectives must be less than length of x vector')

        xm = x[:, (self.number_of_objectives-1):]


        sum_value = (xm - 0.5)**2 - np.cos(20*np.pi*(xm-0.5))
        g = xm.shape[1] + np.sum(sum_value, axis=1)
        g = g*100
        g = g.reshape(-1,1)


        output = np.ones((x.shape[0], self.number_of_objectives))
        output *= 0.5*(1+g)



        for i in range(self.number_of_objectives):
            xprod = x[:,:self.number_of_objectives-(i+1)]
            output[:,i] *= np.prod(xprod, axis=1)
            if i != 0:
                output[:,i] *= (1 - x[:, self.number_of_objectives-(i+1)])


        return output