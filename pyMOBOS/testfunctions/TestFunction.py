from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from scipy.stats import qmc

class TestFunction(ABC):
    # Properties
    number_of_parameters = None
    number_of_objectives = None
    parameter_bounds = None
    name = "Undefined"

    ### Abstract Methods required in child ###
    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def set_bounds(self):
        '''
        Bounds must be saved to self.parameter_bounds
            First row is the lower bounds
            Second row is the upper bounds
        '''
        pass


    def __init__(self, number_of_parameters: int = 2):
        # Set the number of parameters
        self.number_of_parameters = number_of_parameters
        self.set_bounds()

    # Permits the user to update the number of parameters
    def set_number_of_parameters(self, number_of_parameters: int) -> None:
        self.number_of_parameters = number_of_parameters
        self.set_bounds()

    def set_number_of_objectives(self, number_of_objectives: int) -> None:
        self.number_of_objectives = number_of_objectives

    # Quick method to return the bounds for the teset problem
    def bounds(self) -> np.ndarray:
        return self.parameter_bounds

    # Enum that stores the different random styles available
    class Random(Enum):
        RANDOM = 1 # Numpy random
        LHS = 2 # Latin Hypercube Sampling

    def initial_samples(self, number_of_samples: int = 10, random_style: Random = Random.RANDOM) -> tuple[np.array, np.array]:
        '''
        Creates a set of random samples and solutions

        Sphinx Markup
        ------------
        :param number_of_samples int: Number of samples to create
        :param random_style Random: Which randomization method you would like to use. call self.random._member_names_ to get all possible values.
        '''

        # Create random numbers based on style
        # Should be an array of rows = number_of_samples and columns = number_of_parameters
        match random_style:
            # Purely random numbers
            case self.Random.RANDOM:
                samples = np.random.rand(number_of_samples, self.number_of_parameters)
            
            # Latin Hypercube Sampling
            case self.Random.LHS:
                sampler = qmc.LatinHypercube(d = self.number_of_parameters)
                samples = sampler.random(n = number_of_samples)

            case _:
                raise Exception('Please use a random style that exists. Call self.Random._member_names_')

        # Results from above are given in range of [0, 1].
        # Must convert them into appropriate numbers based on the bounds
        range_of_bounds = self.parameter_bounds[1, :] - self.parameter_bounds[0, :]
        minimum_bounds = self.parameter_bounds[0, :]
        samples = samples * range_of_bounds + minimum_bounds
        solutions = self(samples) # Calculate the solution for each sample

        return samples, solutions # Returns as a tuple