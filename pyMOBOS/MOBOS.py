'''
Python Multi-Objective Bayesian Optimization Suite

'''

'''
Having __ in front of a function makes it private
Having _  in front of a function makes it so import * does not by default import that function
https://www.geeksforgeeks.org/args-kwargs-python/
'''
from numpy.random.mtrand import triangular
import numpy as np

import time

from scipy.optimize import minimize




class MOBOS:
    def __init__(self, surrogate, aquisition, f, parameter_dict, n_jobs=1): #TODO: Fix this
        self.GP = surrogate # List of surrogates based on number of objectives
        self.A = aquisition
        self.function = f

        ''' Do a similar aspect to the acquision function where we look up types. or maybe we catch it from the aquisition input?
                This will be serial or batch'''




    '''If serial, we will optimize the aquisition function'''

    def _optimizeAcq(self, method='L-BFGS-B', n_start = 100): 
        pass
        #Implement parallelization

    def _acquisitionWrapper(self, xnew):

        tau = np.max(self.Y, axis = 1) # Move so you only have to calculate this once
        mean, std = self.GP.predict(xnew, return_std = True)
        return self.A(tau, mean, std)

        pass


    # Specify mode: Sequential or Batch

    ''' If it's batch, we will calculate the pareto front and recommend x number of points'''



    

    def recommend(self, X, Y):
        """
            X samples and Y samples.
        """
        pass


    def expected_value(self,X):
        mu, sigma = self.gpr.predict(X, return_std=True)
        return mu

    def calculate_non_dominated(self,X = [], Y = []):
        if X == [] or Y == []:
            X_Sample = self.Design_Variables
            Y_Sample = self.Objectives

        else:
            X_Sample = X
            Y_Sample = Y

        # Taken from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        def is_pareto_efficient(costs, return_mask = True):
            """
            Find the pareto-efficient points
            :param costs: An (n_points, n_costs) array
            :param return_mask: True to return a mask
            :return: An array of indices of pareto-efficient points.
                If return_mask is True, this will be an (n_points, ) boolean array
                Otherwise it will be a (n_efficient_points, ) integer array of indices.
            """
            is_efficient = np.arange(costs.shape[0])
            n_points = costs.shape[0]
            next_point_index = 0  # Next index in the is_efficient array to search for
            while next_point_index<len(costs):
                nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
                nondominated_point_mask[next_point_index] = True
                is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
                costs = costs[nondominated_point_mask]
                next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
            if return_mask:
                is_efficient_mask = np.zeros(n_points, dtype = bool)
                is_efficient_mask[is_efficient] = True
                return is_efficient_mask
            else:
                return is_efficient


        def non_dominated(X_Sample, Y_Sample):
            mask = is_pareto_efficient(Y_Sample)
            return X_Sample[mask], Y_Sample[mask]

        return non_dominated(X_Sample, Y_Sample)


    def propose_location(self, n_restarts=25):
        '''
        Proposes the next sampling point by optimizing the acquisition function.

        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        '''

        # Old data loaded testing function
        '''
        if self.bData_Loaded == False:
            raise Exception('Make sure all data has been loaded! This includes Design Variables, Objectives, and minimum/maximum Bounds for the Design Variables!')
        if self.bGPM_Set == False:
            raise Exception('Data has not been fitted to Gaussian Process Model!')
        '''

        start_time = time.time()


        dim = self.Design_Variables.shape[1]
        #min_val = 1
        min_val = 10**10
        min_x = np.array([[]])

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -self.aquisition(X.reshape(-1, dim), self.Design_Variables, self.Objectives, self.gpr)

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.BoundsDesignVariable[:, 0], self.BoundsDesignVariable[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=self.BoundsDesignVariable, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        if min_x.size > 0:
            min_x[np.abs(min_x) < 10**-16] = 0  #Force values smaller thatn 10**-16 to be 0

            #Catch things that are outside of the bounds
            i = 0
            for val in min_x:
                if np.abs(val - self.BoundsDesignVariable[i,0]) < (10**-15):
                    min_x[i] = self.BoundsDesignVariable[i,0]
                if np.abs(val - self.BoundsDesignVariable[i,1]) < (10**-15):
                    min_x[i] = self.BoundsDesignVariable[i,1]
                i += 1


        if min_x.size > 0:
            self.Proposed_Point = np.transpose(min_x.reshape(-1, 1))
        else:
            self.Proposed_Point = np.array([[]])


        if self.time_optimization == True:
            elapsed_time = int(time.time() - start_time)
            print('Calculation took ' + str(elapsed_time) + ' seconds.' )

        return self.Proposed_Point
