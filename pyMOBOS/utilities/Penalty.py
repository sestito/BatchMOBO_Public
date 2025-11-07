from scipy import special
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt # Only for phi plotting, can remove

def phi(surrogate, X, Y, X_Bounds, xj: np.ndarray):
    '''
    Given a list of points, xj, return the mean and standard deviations for the normal distribution

    Sphinx Markup
    ------------
    :param xj np.ndarray: x array in the form of number of points as rows, and [x1, x2, x3, ...] as the columns. xj is the location of the penalizer.
    :return list: This a list of phi functions
    '''

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
                    #numerator = numerator * -1
                    #print(numerator)

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
            print(np.max(y))
            print(np.min(y))
            plt.plot(x, y)
            plt.show()

    # Create starting guess
    number_of_parameters = X.shape[1]
    number_of_objectives = Y.shape[1]

    [num_functions, num_inputs] = xj.shape

    mean, std = surrogate(xj, return_std = True)

    # This is the minimum value of the outputs.
    # Should be in the form of [y1, y2, y3, ...]
    # In the Gonzalez paper, M was max
    M = np.min(Y, axis = 0)


    L = lipschitz_constant(number_of_parameters, number_of_objectives, X_Bounds, surrogate)

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
def lipschitz_constant(number_of_parameters, number_of_objectives, X_Bounds, surrogate) -> np.ndarray:
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



    L_output = [0] * number_of_objectives

    ranges = np.diff(X_Bounds, axis=0)
    minimum_range = np.min(ranges)
    
    dx_val = minimum_range / 1e-6
    dx_val = 1e-5

    for obj in range(number_of_objectives):
    
        # Create a set of random x0 values
        number_of_initial_samples = 500
        x0_array = np.random.uniform(X_Bounds[0,:], X_Bounds[1,:], size=(number_of_initial_samples, number_of_parameters))

        # Find the minimum value
        y0_array = df(x0_array, surrogate)

        x0 = x0_array[np.argmin(y0_array)]


        res = minimize(df, x0, method='L-BFGS-B', bounds=X_Bounds.transpose(), args = (surrogate, dx_val, obj), options = {'maxiter': 200})
        
        negative_L = float(res.fun)
        L_val = -1*negative_L

        if L_val < 1e-7:
            L_val = 10 # To avoid problems when the model is flat

        L_output[obj] = L_val
    
    # got L = 400 from the Gonzalez paper Figure 1 about the Forrester funciton
    #slope = 20

    #return slope*np.ones(x.shape)
    return np.array(L_output)

