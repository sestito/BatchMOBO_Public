import numpy as np


# Taken from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient



def ParetoEfficient(Y, X = None, return_mask = False, return_rank = False):
    """
    Calculates the Pareto front of the Y data. Returns the corresponding Y Samples.

    The return rank feature is not tested!!!!!!

    Sphinx Markup
    ----------------
    :param Y np.ndarray: Solutions: This is a 2d array in the form of [number of samples, number of objectives]
    :param X np.ndarray: Samples: This is a 2d array in the form of [number of samples, number of paramters]
    :param return_mask bool: Specifies if the function should return the mask for the Pareto front.
    :param return_rank bool: Specifies if the function should return an array of the rank of each solution.
    """

    # Handle the situation where they just want the Pareto front
    if not return_rank:
        mask = is_pareto_efficient(Y)
        if X is None:
            if return_mask:
                return Y[mask], mask
            else:
                return Y[mask]
        else:
            if return_mask:
                return Y[mask], X[mask], mask
            else:
                return Y[mask], X[mask]

    else:
        # Create blank ranks array
        n_points = Y.shape[0]
        ranks = np.zeros(n_points)
        current_data_mask = np.ones(n_points, dtype=bool)
        
        # Initialize rank
        rank = 1


        while np.sum(current_data_mask) > 0:
            mask = is_pareto_efficient(Y[current_data_mask])
            ranks[current_data_mask][mask] = rank
            rank += 1

            current_data_mask[mask] = False
        
        
        mask = is_pareto_efficient(Y)
        if X is None:
            if return_mask:
                return Y[mask], mask, ranks
            else:
                return Y[mask], ranks
        else:
            if return_mask:
                return Y[mask], X[mask], mask, ranks
            else:
                return Y[mask], X[mask], ranks
        
    raise Exception('This function should never reach this point!')
