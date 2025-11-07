from .Surrogate import Surrogate

import numpy as np

# Example Code https://www.egr.msu.edu/coinlab/blankjul/pydacefit/_usage.html?highlight=dacefit%20predict

from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.regr import regr_constant

class GaussianDace(Surrogate):
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        
        

        [m, n] = Y.shape
        self.numObjectives = n
        self.numSamples = m

        [m, n] = X.shape
        self.numParameters = n

        self.fit(X, Y)

        self.X = X
        self.Y = Y



    def fit(self, X, Y):
        self.fitted_models = [None]*self.numObjectives # Create empty list of n models.
        regression = regr_constant
        correlation = corr_gauss

        # TODO Paralelize
        for modelNumber in range(self.numObjectives):
            dacefit = DACE(regr=regression, corr=correlation, theta=1e-1, thetaL=1e-3, thetaU=300)
            dacefit.fit(X, Y[:,modelNumber])
            # Set the kernel to the predefined kernel
            self.fitted_models[modelNumber] = dacefit
            
            

    def __call__(self, Xpredict, return_std=False):
        if not self.fitted_models: 
            raise Exception('The Gaussian process model has not been fitted! Use GP.fit() first.')
        
        return self.predict(Xpredict, return_std=return_std)

    def predict(self, Xpredict, return_std=False):
        [number_of_samples, number_of_parameters] = Xpredict.shape
        
        # Check to make sure shape is correct
        if number_of_parameters != self.numParameters:
            raise Exception('Number of parameters is inconsistent with fit data!')

        mu_array = np.zeros((number_of_samples, self.numObjectives))
        if return_std:
            sigma_array = np.zeros((number_of_samples, self.numObjectives))

        # TODO Paralelize
        for j in range(self.numObjectives):
            if return_std:

                [mu, sigma] = self.fitted_models[j].predict(Xpredict, return_mse=return_std)
                mu_array[:,j] = mu[:,0]
                sigma_array[:,j] = sigma[:,0]

            else:
                 mu = self.fitted_models[j].predict(Xpredict, return_mse=return_std)
                 mu_array[:,j] = mu[:,0]

        '''
        # This code is if the sklearn cannot do multiple predicts at a time.
        # Could be useful for paralelization
        mu_array = np.zeros((number_of_samples, self.numObjectives))
        sigma_array = np.zeros((number_of_samples, self.numObjectives))
        for i in range(number_of_samples):
            x = Xpredict[i]
            for j in range(self.numObjectives):
                [mu_array[i,j], sigma_array[i,j]] = self.fitted_models[j].predict(np.array([x]), return_std=return_std)
        '''

        if return_std:
            return mu_array, sigma_array
        else:
            return mu_array

    def calc_gradient(self, x0_array: np.ndarray, dx=0) -> np.ndarray:
        '''
        This calculates the gradiant for each point in the x0_array as well as for each objective.
        This is similar to the centered first finite difference method.
        This is done using the np.gradient function with a mesh of an xn1 = x0 - dx, x0, and x1 = x0 + dx around each x0 point.

        Sphinx Markup
        ------------
        :param x0_array np.ndarray: 2D array of number of points (rows) by number of parameters (columns)
        :param dx float: dx value to use when calculating the mesh for the gradiant.
        :return np.ndarray: Gradiant in the form of [number of points, number of objectives, number of parameters] where the last index is the gradiant vector.
        '''
        raise Exception('Derivative calculation not implemented yet!')

        # Return the gradiant for each x0 data point passed in
 
        dydx_array = np.zeros((x0_array.shape[0], self.numObjectives))


        # TODO Paralelize
        for j in range(self.numObjectives):

            [mu, sigma] = self.fitted_models[j].predict(x0_array, return_gradient=True)


 

       
        return np.array(dydx_array) # Outputs in form of [number of inputs][number of objectives][number of parameters]