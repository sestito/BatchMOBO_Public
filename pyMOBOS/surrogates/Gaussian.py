'''
This function holds the Gaussian Process Model Surrogate
'''

# We need to build one gaussian process model per objective
'''
Build a list of gaussian process models based on number of objectives
When you predict, you return a list of mu and a list of sigma
Parallelize for each model.

Build this using sklearn
Having a kernel input
'''
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
import numpy as np

from sklearn.gaussian_process.kernels import Matern

from .Surrogate import Surrogate


class Gaussian(Surrogate):
    # Initialization function to get user defined constants
    #   What we need: kernel, 
    def __init__(self,  X: np.ndarray, Y: np.ndarray, method: str = 'sklearn', kernel = None, n_restarts_optimizer: int = 10,**params):
        if kernel is None:
            #self.kernel = ConstantKernel(1.0) * RBF(1.0)
            self.kernel = Matern()
        else:
            self.kernel = kernel
        
        self.n_restarts_optimizer = n_restarts_optimizer
        
        self.params = params
        # NOTE: Don't set just 1 GPR. Just set the parameters and make the GPR when it's fit. (We don't know size yet until Y is passed in)
        #self.gpr = gpr(kernel = self.kernel)
        self.method = method
        self.fit(X, Y, method)

    def __call__(self, Xpredict, return_std=False):
        return self.predict(Xpredict, return_std=return_std)

    # Initializes and fits models
    def fit(self, X, Y, method = 'sklearn'): # Maybe method isn't here but in params?
        '''
        Inputs:
            X: [m, o] where m is the number of samples and o is the number of design variables

            Y: numpy array of shape [m, n] where m is the number of samples and n is the number of objectives
        '''
        # TODO: Check if Kernel has been set


        self.method = method # This is the method for the fit and predict algorithms.
        self.method_dict = {
            'sklearn': {
                'fit': self._fit_sklearn,
                'predict': self._predict_sklearn
            }
        }
        # TODO: Add a check to make sure the method exists in the method_dict

        # TODO: may need to move the following to the sklearn method
        # Save a variable for the number of objectives and number of samples
        [m, n] = Y.shape
        self.numObjectives = n
        self.numSamples = m

        [m, n] = X.shape
        self.numParameters = n

        self.method_dict[self.method]['fit'](X,Y)

        self.X = X
        self.Y = Y

    def _fit_sklearn(self, X, Y):
        self.fitted_models = [None]*self.numObjectives # Create empty list of n models.
        
        # TODO Paralelize
        for modelNumber in range(self.numObjectives):
            # Set the kernel to the predefined kernel
            self.fitted_models[modelNumber] = gpr(kernel=self.kernel, n_restarts_optimizer=self.n_restarts_optimizer).fit(X, Y[:,modelNumber])

    
    
    # Return a list of the predictions based on Xpredict
    def predict(self, Xpredict, return_std=False):
        '''
        Inputs:
            Xpredict: a 2D numpy array of values you want to predict at
        '''
        # Could check if things are set, and if not then error?
        if not self.fitted_models: 
            raise Exception('The Gaussian process model has not been fitted! Use GP.fit() first.')
            # TODO: Check that this exception actually works as intended.

        
        return self.method_dict[self.method]['predict'](Xpredict, return_std)
            

        # NOTE: Maybe we parallelize this? Might not be worth it.

    def _predict_sklearn(self, Xpredict, return_std=False):
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
                [mu_array[:,j], sigma_array[:,j]] = self.fitted_models[j].predict(Xpredict, return_std=return_std)
            else:
                mu_array[:,j] = self.fitted_models[j].predict(Xpredict, return_std=return_std)

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

    # Clears the gaussian process models
    def clear(self):

        del self.fitted_models, self.numObjectives, self.numSamples, self.method, self.numParameters, self.X, self.Y

        # Delete properties that were set
        # See about if we need to delete properties from memory or if setting to None is enough
            # To my understand del makes it so that the references are deleted which is the same as setting the value to None.
            # However, the memory isn't returned to the OS until garbage collection, which doesn't seem to be something easily optimized
            # In short: I believe setting to None (del) should be good enough for our purposes.
    
    def calc_gradient(self, x0_array: np.ndarray, dx: float = 0.0000001) -> np.ndarray:
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

        # Return the gradiant for each x0 data point passed in
        dydx_array = [0] * x0_array.shape[0]

        for k in range(x0_array.shape[0]):
            x0 = x0_array[k]

            # Initialize output
            dydx = [0] * self.numObjectives

            # Create meshgrid
            xn1 = x0 - dx
            x1 = x0 + dx

            input = np.stack((xn1, x0, x1))

            # Creates a list of inputs, [x0, x1, x2, x3, ...., xn] in a 3 x 3 x 3 x ... for n dimensions
            z = np.meshgrid(*(input[:,i] for i in range(input.shape[1])))

            # turn z into array
            z_array = np.array(z)

            # Change shape to  number of Parameters wide by number of inputs
            GP_input_array = z_array.reshape(self.numParameters, -1).transpose()
        
            y_array = self(GP_input_array)

            # Convert y back into the same shape as z_array
            y_array_shape = list(z_array.shape) # z_array is in form of (num_parameters, 3, 3, 3, ...)
            y_array_shape[0] = self.numObjectives
            y_array_mesh = y_array.transpose().reshape(y_array_shape)

            # Extract the gradient
            for i in range(self.numObjectives):
                # Calculate gradient
                grad = np.gradient(y_array_mesh[i], dx) 
                # If there is only 1 parameter, then it returns as an ndarray
                if self.numParameters == 1:
                    grad = [grad]

                # Gradiant for each x dimension
                gradiant_values_in_each_dimension = np.zeros(self.numParameters)
                for j in range(self.numParameters):
                    data = grad[j]

                    # Get center value from data. This is going to be a (3, 3, 3, ...) shape
                    index = [1] * self.numParameters
                    index = tuple(index)

                    gradiant_center = data[index]

                    # We are performing the centered difference quotient



                    # Get the midpoint
                    gradiant_values_in_each_dimension[j] = gradiant_center

                dydx[i] = gradiant_values_in_each_dimension

            dydx_array[k] = dydx           
        return np.array(dydx_array) # Outputs in form of [number of inputs][number of objectives][number of parameters]

