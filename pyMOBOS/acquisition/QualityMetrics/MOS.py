import numpy as np

def MOS(Y):
    y_min = np.amin(Y, axis=0)
    y_max = np.amax(Y, axis=0)
    MOS = np.prod(y_max - y_min)
    return MOS