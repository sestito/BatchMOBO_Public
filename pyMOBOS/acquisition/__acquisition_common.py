#from termios import XCASE
import numpy as np
from scipy.stats import norm
from pyMOBOS.utilities import ParetoEfficient


def __get_mo_method(mo_method='NoneMethod'):
    styles_lookup = {
        'Euclidean': __euclidean,
        'Mean': __mean,
        'NoneMethod':__nonemethod
    }
    return styles_lookup[mo_method]


def ExpectedImprovementArray(self, x: np.ndarray) -> np.ndarray:
    """
    This function calculates the expected improvement of the array x.


    Sphinx Markup
    ------------
    :param x np.ndarray: This is a 2d array in the form of [number of values to calculate, number of parameters]
    :param output_array bool: This tells the aquisition function to output as an array instead of scalar value
    :return np.ndarray: This is a 1D array in the form of [expected imporovement values]  
    """

    mean_array, std_array = self.surrogate(x, return_std = True)

    #tau = np.max(self.surrogate.Y, axis = 0)
    tau = ParetoEfficient(self.surrogate.Y)


    [num_rows, _] = x.shape
    expected_imp_to_return = np.zeros((num_rows, 1))

    for itteration in range(num_rows):
        mean = mean_array[itteration, :].reshape(1,-1)
        std = std_array[itteration, :].reshape(1,-1)

        with np.errstate(divide='ignore'):
            # Maximum
            #z = (mean - tau) / std
            #expected_imp = (mean - tau) * norm.cdf(z) + std * norm.pdf(z)
            
            # Minimum
            z = (tau - mean) / std
            expected_imp = (tau - mean) * norm.cdf(z) + std * norm.pdf(z)
            for i in range(std.shape[1]):
                if std[0, i] == 0.0:
                    expected_imp[:, i] = 0.0
            #expected_imp[std == 0.0] = 0.0

        expected_imp_combined = __get_mo_method(self.mo_method)(expected_imp)
        expected_imp_to_return[itteration] = np.min(expected_imp_combined)

    return expected_imp_to_return

    # TODO Figure out why I did this????
    #minimums = np.min(expected_imp_combined)
    #return np.array([[minimums]])

def ExpectedImprovement(self, x: np.ndarray, output_array: bool = False) -> np.ndarray:
    """
    This function calculates the expected improvement of the array x.


    Sphinx Markup
    ------------
    :param x np.ndarray: This is a 2d array in the form of [number of values to calculate, number of parameters]
    :param output_array bool: This tells the aquisition function to output as an array instead of scalar value
    :return np.ndarray: This is a 1D array in the form of [expected imporovement values]  
    """

    mean, std = self.surrogate(x, return_std = True)

    #tau = np.max(self.surrogate.Y, axis = 0)
    tau = ParetoEfficient(self.surrogate.Y)



    with np.errstate(divide='ignore'):
        # Maximum
        #z = (mean - tau) / std
        #expected_imp = (mean - tau) * norm.cdf(z) + std * norm.pdf(z)
        
        # Minimum
        z = (tau - mean) / std
        expected_imp = (tau - mean) * norm.cdf(z) + std * norm.pdf(z)
        for i in range(std.shape[1]):
            if std[0, i] == 0.0:
                expected_imp[:, i] = 0.0
        #expected_imp[std == 0.0] = 0.0



    if output_array:
        # This outputs it as an array of expected imp values
        expected_imp_to_return = expected_imp       

    else:
        expected_imp_combined = __get_mo_method(self.mo_method)(expected_imp)
        expected_imp_to_return = np.min(expected_imp_combined)

    return expected_imp_to_return

    # TODO Figure out why I did this????
    #minimums = np.min(expected_imp_combined)
    #return np.array([[minimums]])

def ProbabilityImprovement(self, x: np.ndarray) -> np.ndarray:
    """
    This function calculates the expected improvement of the array x.


    Sphinx Markup
    ------------
    :param x np.ndarray: This is a 2d array in the form of [number of values to calculate, number of parameters]
    :return np.ndarray: This is a 1D array in the form of [expected imporovement values]  
    """

    mean, std = self.surrogate(x, return_std = True)

    #fmax = np.max(self.surrogate.Y, axis = 0) # Get the max f value for each objective. 1d array of length number of objectives
    #tau = fmax + self.epsilon
    fmax = ParetoEfficient(self.surrogate.Y)
    tau = fmax - self.epsilon

    with np.errstate(divide='ignore'):
        z = (tau - mean) / std
        PI = norm.cdf(z)
        #PI[std == 0.0] = 0
        for i in range(std.shape[1]):
            if std[0, i] == 0.0:
                PI[:, i] = 0.0
    

    PI_combined = __get_mo_method(self.mo_method)(PI)
    PI_to_return = np.min(PI_combined)

    return PI_to_return   

def ProbabilityImprovementArray(self, x: np.ndarray) -> np.ndarray:
    """
    This function calculates the expected improvement of the array x.


    Sphinx Markup
    ------------
    :param x np.ndarray: This is a 2d array in the form of [number of values to calculate, number of parameters]
    :return np.ndarray: This is a 1D array in the form of [expected imporovement values]  
    """

    mean_array, std_array = self.surrogate(x, return_std = True)

    [num_rows, _] = x.shape
    PI_to_return = np.zeros((num_rows, 1))

    #fmax = np.max(self.surrogate.Y, axis = 0) # Get the max f value for each objective. 1d array of length number of objectives
    #tau = fmax + self.epsilon
    fmax = ParetoEfficient(self.surrogate.Y)
    tau = fmax - self.epsilon

    for itteration in range(num_rows):
        mean = mean_array[itteration, :].reshape(1,-1)
        std = std_array[itteration, :].reshape(1,-1)

        with np.errstate(divide='ignore'):
            z = (tau - mean) / std
            PI = norm.cdf(z)
            #PI[std == 0.0] = 0
            for i in range(std.shape[1]):
                if std[0, i] == 0.0:
                    PI[:, i] = 0.0
        

        PI_combined = __get_mo_method(self.mo_method)(PI)
        PI_to_return[itteration] = np.min(PI_combined)

    return PI_to_return   


def UpperConfidenceBound(self, x: np.ndarray) -> np.ndarray:
    
    mean, std = self.surrogate(x, return_std = True)
    UCB = mean + self.beta * std
    UCB_combined = __get_mo_method(self.mo_method)(UCB)
    UCB_to_return = np.min(UCB_combined)
    return UCB_to_return
    

def LowerConfidenceBound(self, x: np.ndarray) -> np.ndarray:
    mean, std = self.surrogate(x, return_std = True)
    LCB = mean - self.beta * std
    LCB_combined = __get_mo_method(self.mo_method)(LCB)
    LCB_to_return = np.min(-1*LCB_combined)
    return LCB_to_return

def LowerConfidenceBoundArray(self, x: np.ndarray) -> np.ndarray:
    mean_array, std_array = self.surrogate(x, return_std = True)
    [num_rows, _] = x.shape
    LCB_to_return = np.zeros((num_rows, 1))    

    for itteration in range(num_rows):
        mean = mean_array[itteration, :].reshape(1,-1)
        std = std_array[itteration, :].reshape(1,-1)
        LCB = mean - self.beta * std
        LCB_combined = __get_mo_method(self.mo_method)(LCB)
        LCB_to_return[itteration] = np.min(-1*LCB_combined)
    return LCB_to_return


def __euclidean(X: np.ndarray) -> np.ndarray:
    """
    This combines the expected improvement values for each objective by the Euclidean distance such that return = sqrt( x[1]^2 + x[2]^2 + ...) where x is X[i,:].

    Sphinx Markup
    ------------
    :param X np.ndarray: This is a 2d array in the form of [number of x values to calculate, number of parameters]
    :return: This is a 1d array in the form of [expected improvement values]
    """
    # NOTE: the origin is currently just zero, but this can be changed in the future
    origin = np.zeros(X.shape)

    #distance = (origin-X)
    #distance = distance ** 2
    #distance = np.sum(distance, axis = 1)
    #distance = np.sqrt(distance)
    #distance = distance.reshape(-1,1) # Convert to 1 column


    distance_2 = np.linalg.norm(origin - X, axis=1)
    distance_2 = distance_2.reshape(-1,1)

    return distance_2

def __mean(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis = 1).reshape(-1,1)

def __nonemethod(X: np.ndarray) -> np.ndarray:
    return X
