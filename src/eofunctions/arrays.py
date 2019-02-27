import numpy as np
import sys
import warnings
import functools
import operator
import copy
from eofunctions.math import min_, max_, is_empty

# TODO: Flattening is applied in some function calls, is this necessary?
# TODO: Where should the data types be checked? In each function?
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


def first(data):
    data = flatten(data)
    first_elem = None
    if not is_empty(data):
        first_elem = data[0]

    return first_elem


def last(data):
    data = flatten(data)
    last_elem = None
    if not is_empty(data):
        last_elem = data[-1]

    return last_elem


def order(data, asc=True, nodata=None):
    data = flatten(data)

    if asc:
        permutation_idxs = np.argsort(data)
    else:
        permutation_idxs = np.argsort(data)[::-1]

    data_sorted = data[permutation_idxs]
    nan_idxs = np.isnan(data_sorted)
    if nodata == True:  # boolean comparison necessary, since nodata can not be only of type boolean
        return permutation_idxs[~nan_idxs] + permutation_idxs[nan_idxs]
    elif nodata == False:
        return permutation_idxs[nan_idxs] + permutation_idxs[~nan_idxs]
    else:
        return permutation_idxs[~nan_idxs]


def rearrange(data, order):
    data = flatten(data)
    if len(data) != len(order):
        raise Exception('The number of data and order elements has to match!')
    return data[order]


# rearrange(data, order(data, nodata)) could be used, but is probably slower than sorting the array directly
def sort(data, asc=True, nodata=None):
    data = flatten(data)

    if asc:
        data_sorted = np.sort(data)
    else:
        data_sorted = np.sort(data)[::-1]

    nan_idxs = np.isnan(data_sorted)
    if nodata == True:  # boolean comparison necessary, since nodata can not be only of type boolean
        return data_sorted[~nan_idxs] + data_sorted[nan_idxs]
    elif nodata == False:
        return data_sorted[nan_idxs] + data_sorted[~nan_idxs]
    else:
        return data_sorted[nan_idxs]


def _and(expressions, ignore_nodata=True):
    if is_empty(expressions):
        return None
    expressions_copy = copy.deepcopy(expressions)
    if not ignore_nodata:
        expressions_copy[np.isnan(expressions)] = False
        if np.all(expressions) and not np.all(expressions_copy):
            return None
        else:
            return np.all(expressions_copy)
    else:
        return np.all(expressions)

def _or(expressions, ignore_nodata=True):
    if is_empty(expressions):
        return None
    expressions_copy = copy.deepcopy(expressions)
    if not ignore_nodata:
        expressions_copy[np.isnan(expressions)] = False
        if np.any(expressions) and not np.any(expressions_copy):
            return None
        else:
            return np.any(expressions_copy)
    else:
        return np.any(expressions)

def _xor(expressions, ignore_nodata=True):
    if is_empty(expressions):
        return None
    if not ignore_nodata:
        if np.any(np.isnan(expressions)):
            return None
        else:
            return np.nansum(expressions) == 1
    else:
        return np.nansum(expressions) == 1


def clip(data, min, max):
    data = np.where(data < min, min, data)
    data = np.where(data > max, max, data)

    return data

def sum_first_elem(data):
    for i in range(1, len(data)):
        data[0] += float(data[i])
    return data[0]

def sum(data, ignore_nodata):
    if len(data) < 2:
        err_message = "Addition requires at least two numbers."
        sys.exit(err_message)
    if not ignore_nodata:
        if np.any(np.isnan(data)):
            return None
        else:
            data = data[~np.isnan(data)]
            sum_first_elem(data)
    else:
        data = data[~np.isnan(data)]
        sum_first_elem(data)


def subtract_first_elem(data):
    for i in range(1, len(data)):
        data[0] -= float(data[i])
    return data[0]

def subtract(data, ignore_nodata):
    if len(data) < 2:
        err_message = "Subtraction requires at least two numbers (a minuend and one or more subtrahends)."
        sys.exit(err_message)
    if not ignore_nodata:
        if np.any(np.isnan(data)):
            return None
        else:
            data = data[~np.isnan(data)]
            subtract_first_elem(data)
    else:
        data = data[~np.isnan(data)]
        subtract_first_elem(data)


def multiply_first_elem(data):
    for i in range(1, len(data)):
        data[0] *= float(data[i])
    return data[0]

def multiply(data, ignore_nodata):
    if len(data) < 2:
        err_message = "Multiplication requires at least two numbers."
        sys.exit(err_message)
    if not ignore_nodata:
        if np.any(np.isnan(data)):
            return None
        else:
            data = data[~np.isnan(data)]
            multiply_first_elem(data)
    else:
        data = data[~np.isnan(data)]
        multiply_first_elem(data)

def product(data, ignore_nodata):
    return multiply(data, ignore_nodata=ignore_nodata)

def divide_first_elem(data):
    for i in range(1, len(data)):
        data[0] /= float(data[i])
    return data[0]

def divide(data, ignore_nodata):
    if len(data) < 2:
        err_message = "Division requires at least two numbers (a dividend and one or more divisors)."
        sys.exit(err_message)
    if not ignore_nodata:
        if np.any(np.isnan(data)):
            return None
        else:
            data = data[~np.isnan(data)]
            divide_first_elem(data)
    else:
        data = data[~np.isnan(data)]
        divide_first_elem(data)


def extrema(data, ignore_nodata=True):
    min_val = min_(data, ignore_nodata=ignore_nodata)
    max_val = max_(data, ignore_nodata=ignore_nodata)
    return [min_val, max_val]



