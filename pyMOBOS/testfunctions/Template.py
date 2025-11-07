'''
Must modify __init__.py to include new Test Function
'''

# Parent object
from .TestFunction import TestFunction

# Imports
import numpy as np



class Template(TestFunction):
    '''
    DESCRIPTION


    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        
        '''


        return

    def bounds(self) -> np.ndarray:
        '''
        Bounds must be saved to self.parameter_bounds
            First row is the lower bounds
            Second row is the upper bounds
        '''


        self.parameter_bounds = 