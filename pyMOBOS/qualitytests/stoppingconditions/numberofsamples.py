from xmlrpc.client import Boolean
import numpy as np

class NumberOfSamples:
    def __init__(self, number_of_samples, initial_samples):
        self.number_of_samples = number_of_samples
        self.initial_samples = initial_samples

    def __call__(self, X, Y):
        [num_samples, _] = X.shape
        stop: Boolean = False
        if num_samples >= (self.initial_samples + self.number_of_samples):
            stop = True
        return stop