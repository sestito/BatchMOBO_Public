import numpy as np
from scipy.interpolate import griddata

def _cartesian(arrays, out=None):
    """
    Pulled from https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/1235363#1235363

    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        _cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def IHD(Y, num = 100):
    numberOfSamples, numberOfObjectives = Y.shape

    YMin = np.amin(Y,axis=0)
    YMax = np.amax(Y,axis=0)

    dl = (YMax - YMin) / (num)
    dA = np.prod(dl[:-1]) #all but the last one

    arrs = []
    for i in range(numberOfObjectives-1):
        arrs.append(np.linspace(YMin[i],YMax[i],num=num,endpoint=False)) #Left hand reimann sums

    if numberOfSamples <= (numberOfObjectives-1):
        # If 2 objectives and 1 point, MHD = 0
        # If 3 objectives and 2 points, MHD = 0
        # If 3 objectives and 1 point, MHD = -1 (better)
        # If 4 objectives and 1 points, MHD = -2 (best)
        # If 4 objectives and 2 points, MHD = -1 (okay)
        # If 4 objectives and 3 points, MHD = 0 (worst)
        IHD = 1-numberOfObjectives+numberOfSamples #There is no area under a point
    else:
        #Interpolate for the last objective and sum up the area/volume/ect for all non nan values
        mu = griddata(Y[:,0:numberOfObjectives-1],Y[:,numberOfObjectives-1],_cartesian(arrs),method='linear')
        IHD = np.nansum((mu-YMin[-1])*dA)
    return IHD