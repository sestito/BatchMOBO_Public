from abc import ABC, abstractmethod
import numpy as np

class Surrogate(ABC):
    # Properties

    ### Abstract Methods required in child ###
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, Xpredict, return_std=False):
        pass


    @abstractmethod
    def calc_gradient(self, x0_array: np.ndarray) -> np.ndarray:
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
        pass
