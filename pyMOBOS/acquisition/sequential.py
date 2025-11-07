# TODO Is LCB correct? Should it be negative?

import time
from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm

from . import __acquisition_common as acquisition_common


class Sequential:
    
    time: bool = False # If set to true, will output the time it takes for the recommendation calculation

    def __init__(self, X: np.ndarray, Y: np.ndarray, X_Bounds: np.ndarray, surrogate: object, n_restarts: int = 25, aquisition_function: str = 'EI', epsilon: float = 0.1, beta: float = 1, mo_method: str = 'Euclidean') -> None:
        '''
        Initialization of the EIMe class


        Sphinx Markup
        ------------
        :param X np.ndarray: Design Variables. This is a 2d array in the form of [number of samples, number of parameters]
        :param Y np.ndarray: Objectives. This is a 2d array in the form of [number of samples, number of objectives]
        :param X_Bounds np.ndarray: These are the bounds of your design variabls in the form of [2, number of design variables]. X_Bounds[0,:] corresponds to your minimum and X_Bounds[1,:] corresponds to your maximum.
        :param surrogate object: Surrogate Object. This is the initialized surrogate to be used.
        :param n_restarts int: Number of restarts during the optimization process. More restarts provide a higher chance to find the true optimal at increased computation time. Default is 25.
        :param aquisition_function str: Name of the aquisition function to use. Options include Expected Improvement 'EI', Probability Improvement 'PI', Upper Confidence Bound 'UCB', and Lower Confidence Bound 'LCB'
        :param mo_method str: Name of the multi-objective combination method to use. Options include 'Euclidean' and 'Mean'. Ex. 'Euclidean' -> value = sqrt( value_1^2 + value_2^2 + ... )
        :param epsilon float: Only for PI. This is the value added onto the max f(x) value. Higher epsilon values query locations with larger standard deviation.
        :param beta float: Only for UCB and LCB. This value effects UCB and LCB such that value = mean += beta * std
        '''
        #styles_lookup = {
        #    'Euclidean': self.__euclidean,
        #    'Mean': self.__mean
        #}
        styles_lookup = {
            'Euclidean': 'Euclidean',
            'Mean': 'Mean'
        }

        #acquisition_dict = {
        #    'EI': self.ExpectedImprovement,
        #    'PI': self.ProbabilityImprovement,
        #    'UCB': self.UpperConfidenceBound,
        #    'LCB': self.LowerConfidenceBound
        #    }

        acquisition_dict = {
            'EI': acquisition_common.ExpectedImprovement,
            'PI': acquisition_common.ProbabilityImprovement,
            'UCB': acquisition_common.UpperConfidenceBound,
            'LCB': acquisition_common.LowerConfidenceBound
            }

        self.X = X
        self.Y = Y
        self.X_Bounds = X_Bounds
        self.surrogate = surrogate
        self.n_restarts = n_restarts
        self.beta = beta
        self.epsilon = epsilon

        #self.surrogate.init(self.X, self.Y)

        try:
            self.aquisition = acquisition_dict[aquisition_function]
        except KeyError:
            raise Exception('Acquisition function ' + str(aquisition_function) + ' does not exist!')

        try:
            self.mo_method = styles_lookup[mo_method]
        except KeyError:
            raise Exception('Multi-objective method ' + str(mo_method) + ' does not exist!')

    def __call__(self) -> np.ndarray:
        return self.propose_location()


    def propose_location(self) -> np.ndarray:
        """
        This method proposes the next recommended set of design variables that will maximize the expected improvement.

        Sphinx Markup
        ------------
        :return np.ndarray: This is a 1d array in the form of [design variables]
        """

        if self.time:
            start_time = time.time()


        number_of_parameters = self.X.shape[1]



        def min_obj(x: np.ndarray) -> float:
            # Minimization objective is the negative acquisition function
            value = -self.aquisition(self, x.reshape(-1, number_of_parameters)) # This comes out as a 2D array
            return value


        # Find the best optimum by starting from n_restart different random points.
        X0 = np.zeros((self.n_restarts, number_of_parameters))
        result = np.zeros(self.n_restarts)
        restart_number = 0
        # TODO Parallelize loop
        for x0 in np.random.uniform(self.X_Bounds[0, :], self.X_Bounds[1, :], size=(self.n_restarts, number_of_parameters)):
            res = minimize(min_obj, x0=x0, bounds=self.X_Bounds.transpose(), method='L-BFGS-B')
            result[restart_number] = res.fun # This pulls out the value. res.x is the x0.
            X0[restart_number,:] = res.x
            restart_number += 1

        #Force values smaller thatn 10**-16 to be 0
        X0[np.abs(X0) < 10**-16] = 0

        # TODO Check to see if xvalues are within bounds

        index_of_min = np.where(result == np.amin(result))[0][0]
        x_next = X0[index_of_min, :]
        

        '''
        if min_x.size > 0:
            proposed_point = np.transpose(min_x.reshape(-1, 1))
        else:
            proposed_point = np.array([[]])
        '''

        if self.time:
            elapsed_time = int(time.time() - start_time)
            print('Calculation took ' + str(elapsed_time) + ' seconds.' )


        return np.array([x_next])


