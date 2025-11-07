from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from scipy.stats import qmc

class AcquisitionParent(ABC):
    pass

    #@abstractmethod
    #def __call__(self):
    #    pass

    def __call__(self) -> np.ndarray:
        pass