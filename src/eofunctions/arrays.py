import numpy as np
import sys
import warnings
import functools
import operator


def flatten(data):
    if data.ndims > 1:
        warn_message = "Array has more than one dimension. Flattening will be applied."
        warnings.warn(warn_message)
        data = data.flatten()

    return data


def array_contains(data, element):
    return np.any(np.isin(element, data))


def array_element(data, index, return_nodata=False):

    data = flatten(data)
    element = None
    if index >= len(data):
        if not return_nodata:
            err_message = "The array has no element with the specified index."
            sys.exit(err_message)
    else:
        element = data[index]

    return element


def count(data, expression=None):
    if expression:  # expression True or not None
        if type(expression) == 'boolean':
            num_of_elems = functools.reduce(operator.mul, data.shape, 1)
        elif type(expression) == 'str':
            num_of_elems = np.sum(eval(expression.replace('x', 'data')))
        else:
            err_message = "Expression data type is not supported."
            sys.exit(err_message)
    else:  # expression False or None
        num_of_elems = np.sum(~np.isnan(data) & ~np.isinf(data))

    return num_of_elems


def order(data, asc=True, nodata=None):
    data = flatten(data)

    if asc:
        permutation_idxs = np.argsort(data)
    else:
        permutation_idxs = np.argsort(data)[::-1]

    data_sorted = data[permutation_idxs]
    nan_idxs = np.isnan(data_sorted)
    permutation_idxs_subset = permutation_idxs[~nan_idxs]
    if nodata == True:  # boolean comparison necessary, since nodata can not be only of type boolean
        return permutation_idxs_subset + permutation_idxs[nan_idxs]
    elif nodata == False:
        return permutation_idxs[nan_idxs] + permutation_idxs_subset
    else:
        return permutation_idxs_subset


def first(data):
    data = flatten(data)
    first_elem = None
    if len(data) != 0:
        first_elem = data[0]

    return first_elem


def last(data):
    data = flatten(data)
    last_elem = None
    if len(data) != 0:
        last_elem = data[-1]

    return last_elem