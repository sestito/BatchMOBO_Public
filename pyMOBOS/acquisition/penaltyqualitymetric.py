import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from scipy import special

import matplotlib.pyplot as plt # Only for phi plotting, can remove

from pyMOBOS.utilities import ParetoEfficient



class PenaltyQualityMetric:
    def __init__(self, X: np.ndarray, Y: np.ndarray, X_Bounds: np.ndarray, surrogate: object, quality_metrics: list, n_restarts: int = 25):
        '''
        Initialization

        Sphinx Markup
        ------------
        :param X np.ndarray: Design Variables. This is a 2d array in the form of [number of samples, number of parameters]
        :param Y np.ndarray: Objectives. This is a 2d array in the form of [number of samples, number of objectives]
        :param X_Bounds np.ndarray: These are the bounds of your design variabls in the form of [2, number of design variables]. X_Bounds[0,:] corresponds to your minimum and X_Bounds[1,:] corresponds to your maximum.
        :param surrogate object: Surrogate Object. This is the initialized surrogate to be used.
        :param quality_metrics List: List of quality metric functions for this acquisition function
        :param n_restarts int: Number of restarts during the optimization process. More restarts provide a higher chance to find the true optimal at increased computation time. Default is 25.
        '''
        
        # Store X, Y, X_Bounds, and surrogate
        self.X = X
        self.Y = Y
        self.X_Bounds = X_Bounds
        self.surrogate = surrogate

        # Specify Quality Metrics to use
        self.quality_metrics = quality_metrics

        self.n_restarts = n_restarts

        self.L = self.lipschitz_constant()


    def __call__(self, batch_size: int = 1) -> np.ndarray:
        return self.propose_location(batch_size)
        
    def caclulate_quality_metrics(self, pareto: np.ndarray):
        number_of_quality_metrics = len(self.quality_metrics)
        quality_metric_values = np.zeros((number_of_quality_metrics, 1)) # Initialize an n x 1 array
        i = 0
        for qm in self.quality_metrics:
            quality_metric_values[i] = qm(pareto)
            i += 1

        return quality_metric_values


    def acquisition(self, x: np.ndarray) -> np.ndarray:
        '''
       
        Sphinx Markup
        -----------
        :param x np.ndarray: A n x number_of_parameters shape array.
        :return np.ndarray: Returns a n x 1 array of acquisition function values
        '''
        
        # Get number of samples
        number_of_parameters = self.X.shape[1]

        # Retrieve the pareto solutions
        pareto_Y = ParetoEfficient(self.Y)

        QM_values = self.caclulate_quality_metrics(pareto_Y)

        a = [] # Output
        for x_val in x: # This will iterate through each row
            # Get mean and std of the sample (x)
            mean, std = self.surrogate(x_val.reshape(-1, number_of_parameters), return_std = True)
            

            # See if new mean is in the Pareto Front
            Y_new = np.append(self.Y, mean, axis = 0)
            pareto_Y_new = ParetoEfficient(Y_new)

            if not np.array_equal(pareto_Y_new, pareto_Y): # Pareto new has changed with sample
                # Calculate quality metrics
                QM_values_new = self.caclulate_quality_metrics(pareto_Y_new)

                # Calculate relative metric change


                # Percentage Difference
                # https://en.wikipedia.org/wiki/Relative_change_and_difference
                # https://stats.stackexchange.com/questions/86708/how-to-calculate-relative-error-when-the-true-value-is-zero
                #RelativeMetrics = np.abs((QM_values - QM_values_new) / ((QM_values + QM_values_new) / 2))

                '''
                # Percentage Change
                RelativeMetrics = np.abs((QM_values - QM_values_new)/QM_values)

                PercentDifferenceMetric = np.abs((QM_values - QM_values_new) / ((QM_values + QM_values_new) / 2))
                mask_zero = QM_values == 0
                RelativeMetrics[mask_zero] = PercentDifferenceMetric[mask_zero]


                PercentDifferenceMagnituteMetric = np.abs((QM_values - QM_values_new) / ((np.abs(QM_values) + np.abs(QM_values_new)) / 2))
                mask_oposite = QM_values + QM_values_new == 0
                mask_oposite = np.logical_and(mask_oposite, mask_zero)
                RelativeMetrics[mask_oposite] = PercentDifferenceMagnituteMetric[mask_oposite]

                
                mask_equal = QM_values == QM_values_new
                RelativeMetrics[mask_equal] = 0
                '''

                RelativeMetrics = np.abs(QM_values - QM_values_new) / (( np.abs(QM_values) + np.abs(QM_values_new) ) / 2)
                mask_equal = QM_values == QM_values_new
                RelativeMetrics[mask_equal] = 0
                
                if np.isnan(RelativeMetrics).any():
                    raise Exception('Error! NaN! Most likely wrong dimensional problem (2+ objectives required) OR one of the quality metrics is 0!')
               
                # Calculate max change of metrics
                a.append(np.max(RelativeMetrics))

            else:
                #Calculate f_LCB
                f_LCB = mean - std #Corresponds to ff in ac_fun.m

                #Calculate minimum Euclidean distance between fLCB and non-dominated solution
                #a = -min(||fLCB - f(xi)||)
                distance = f_LCB - pareto_Y
                Euclidean_Distance = np.sum(np.power(distance,2), axis = 1)
                a.append(-1*np.min(np.power(Euclidean_Distance,0.5)))

        a = np.array(a)
        return a.reshape(-1, 1)




    def penalized_acquisition(self, x):
        val = self.acquisition(x)


        if self.proposed_locations != []:
            xj_array = np.array(self.proposed_locations)
            phi_funcs = self.phi(xj_array)
              
            for phi_func in phi_funcs:

                # Phi func will return a phi value for each y
                # so a phi correspoinding to [y1, y2, y3, ....]
                phi_adjustment = phi_func(x)
                phi_adjustment = np.mean(phi_adjustment, axis = 1)

                val = val * phi_adjustment   

        return val


    def phi(self, xj: np.ndarray):
        '''
        Given a list of points, xj, return the mean and standard deviations for the normal distribution

        Sphinx Markup
        ------------
        :param xj np.ndarray: x array in the form of number of points as rows, and [x1, x2, x3, ...] as the columns. xj is the ocation of the penalizer.
        :return list: This a list of phi functions
        '''

        [num_functions, num_inputs] = xj.shape

        mean, std = self.surrogate(xj, return_std = True)

        # This is the minimum value of the outputs.
        # Should be in the form of [y1, y2, y3, ...]
        # In the Gonzalez paper, M was max
        M = np.min(self.Y, axis = 0)

        L = self.L

        output_phi_functions = [0]*num_functions

        for i in range(num_functions):
            xj_temp = xj[i, :]
            mean_temp = mean[i,:]
            std_temp = std[i,:]

            output_phi_functions[i] = Phi(M, L, xj_temp, mean_temp, std_temp)

        return output_phi_functions


    # Information about how to implement this can be found at the following:
    # https://github.com/SheffieldML/GPyOpt/blob/0be0508f00934043815dd46b9a331e3847070aae/GPyOpt/core/evaluators/batch_local_penalization.py#L52
    # https://github.com/SheffieldML/GPy/blob/devel/GPy/core/gp.py
    # https://stackoverflow.com/questions/16078818/calculating-gradient-with-numpy
    def lipschitz_constant(self) -> np.ndarray:
        '''
        Calculates the Lipschitz constant based on Gonzalez et al. (2016) implementation called GP-LCA.
        The self.surrogate must already be fitted.
        This will output a Lipschitz constant for each objective.

        Sphinx Markup
        ------------
        :return np.ndarray: 1D array of Lipschitz constants for each objective.
        '''

        def df(x, model, dx = 0.00001, objective=0):
            # Make sure x array is 2d
            x = np.atleast_2d(x)

            # Calculate gradiant
            # Comes out in form of [number of inputs = 1][number of objectives][number of parameters]

            dydx = model.calc_gradient(x, dx = dx)
            norm = np.linalg.norm(dydx, axis=2)

            return -1* norm[:, objective] # Negative because we are minimizing



        # Create starting guess
        number_of_parameters = self.X.shape[1]
        number_of_objectives = self.Y.shape[1]

        L_output = [0] * number_of_objectives

        ranges = np.diff(self.X_Bounds, axis=0)
        minimum_range = np.min(ranges)
     
        dx_val = minimum_range / 1e-6
        dx_val = 1e-5

        for obj in range(number_of_objectives):
        
            # Create a set of random x0 values
            number_of_initial_samples = 500
            x0_array = np.random.uniform(self.X_Bounds[0,:], self.X_Bounds[1,:], size=(number_of_initial_samples, number_of_parameters))

            # Find the minimum value
            y0_array = df(x0_array, self.surrogate)

            x0 = x0_array[np.argmin(y0_array)]


            res = minimize(df, x0, method='L-BFGS-B', bounds=self.X_Bounds.transpose(), args = (self.surrogate, dx_val, obj), options = {'maxiter': 200})
            
            negative_L = float(res.fun)
            L_val = -1*negative_L

            if L_val < 1e-7:
                L_val = 10 # To avoid problems when the model is flat

            L_output[obj] = L_val
        
        # got L = 400 from the Gonzalez paper Figure 1 about the Forrester funciton
        #slope = 20

        #return slope*np.ones(x.shape)
        return np.array(L_output)


    def propose_location(self, batch_size: int = 1):


        # Reset proposed locations to nothing.
        # Penalized acquisition Function Calls this
        self.proposed_locations = []

        number_of_parameters = self.X.shape[1]

        def min_obj(x: np.ndarray) -> float:
            # Minimization objective is the negative acquisition function
            value = -self.penalized_acquisition(x.reshape(-1, number_of_parameters)) # This comes out as a 2D array
            return value

        for i in range(batch_size):
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

            self.proposed_locations.append(x_next)


        proposed_locations = np.array(self.proposed_locations)
        self.proposed_locations = []
        return proposed_locations



class Phi:
    def __init__(self, M, L, xj, mean, std):
        self.M = M
        self.L = L
        self.xj = xj
        self.mean = mean
        self.std = std


    def __call__(self, x):
            return self.phi_call(x)

    def phi_call(self, x):
            [num_samples, num_inputs] = x.shape
            phi = [0]*num_samples

            for i in range(num_samples):
                # Calculate ||xj - x||
                val_to_norm = self.xj - x
                normed_val = np.linalg.norm(val_to_norm)

                # Calculate numerator
                # L ||xj - x|| - M + un(xj)
                numerator = self.L * normed_val - (self.mean - self.M)

                # Calculate denominator
                # sqrt( 2 * sigma(xj)^2)
                denominator = self.std ** 2
                denominator = 2 * denominator
                denominator = denominator ** 0.5

                # Calculate z score
                z = numerator / denominator

                # Calculate phi
                # phi = 0.5 * erfc(-z)
                phi[i] = 0.5 * special.erfc(-z)

            return phi


    def plot(self, x_min, x_max, n = 1000):
        x = np.linspace(x_min, x_max, n)
        y = []
        for val in x:
            pass_in = np.array([[val]])
            to_append = self.phi_call(pass_in)
            y.append(to_append[0][0])
        plt.plot(x, y)
        plt.show()