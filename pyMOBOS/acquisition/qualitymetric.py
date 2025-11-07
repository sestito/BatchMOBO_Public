import numpy as np
import time
from scipy.optimize import minimize

from pyMOBOS.utilities import ParetoEfficient

class QualityMetric:
    time = False    

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


    def __call__(self) -> np.ndarray:
        return self.propose_location()

    def caclulate_quality_metrics(self, pareto: np.ndarray):
        number_of_quality_metrics = len(self.quality_metrics)
        quality_metric_values = np.zeros((number_of_quality_metrics, 1)) # Initialize an n x 1 array
        i = 0
        for qm in self.quality_metrics:
            quality_metric_values[i] = qm(pareto)
            i += 1

        return quality_metric_values


    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        '''
       
        Sphinx Markup
        -----------
        :param x np.ndarray: A n x number_of_parameters shape array.
        :return np.ndarray: Returns a n x 1 array of aquisition function values
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
            value = -self.acquisition_function(x.reshape(-1, number_of_parameters)) # This comes out as a 2D array
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

