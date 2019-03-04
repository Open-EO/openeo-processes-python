import numpy as np
import pandas as pd
import sys
import warnings
import functools
import operator
import copy
from eofunctions.math import min_, max_, is_empty, is_valid


# TODO: Flattening is applied in some function calls, is this necessary?
# TODO: Where should the data types be checked? In each function?
def flatten(data, dtype=object):
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=dtype)
    if data.ndim > 1:
        warn_message = "Array has more than one dimension. Flattening will be applied."
        warnings.warn(warn_message)
        data = data.flatten()

    return data


def array_contains(data, element):
    return element in data


def array_element(data, index, return_nodata=False):

    data = flatten(data)
    element = np.nan
    if index >= len(data):
        if not return_nodata:
            err_message = "The array has no element with the specified index."
            sys.exit(err_message)
    else:
        element = data[index]

    return element


# TODO: This function does not work fully yet as specified in
# https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#count
# link to a function understanding a process graph is needed
def count(data, expression=None):
    if expression:  # expression True or not None
        if type(expression) == bool:
            num_of_elems = functools.reduce(operator.mul, np.array(data).shape, 1)
        else:
            num_of_elems = np.sum(eval(expression.replace('x', 'data')))
    else:  # expression False or None
        num_of_elems = np.sum([is_valid(elem) for elem in data])

    return num_of_elems


def first(data, ignore_nodata=True):
    data = flatten(data)
    first_elem = np.nan
    if not is_empty(data):
        if ignore_nodata:
            first_elem = data[~pd.isnull(data)][0]
        else:
            first_elem = data[0]

    return first_elem


def last(data, ignore_nodata=True):
    data = flatten(data)
    last_elem = np.nan
    if not is_empty(data):
        if ignore_nodata:
            last_elem = data[~pd.isnull(data)][-1]
        else:
            last_elem = data[-1]

    return last_elem


def order(data, asc=True, nodata=None):
    data = flatten(data, dtype=float)

    if asc:
        permutation_idxs = np.argsort(data, kind='mergesort')
    else:  # [::-1] not possible
        permutation_idxs = np.argsort(-data, kind='mergesort')

    data_sorted = data[permutation_idxs]
    nan_idxs = pd.isnull(data_sorted)
    if nodata == True:  # boolean comparison necessary, since nodata can not be only of type boolean
        return permutation_idxs[~nan_idxs].tolist() + permutation_idxs[nan_idxs].tolist()
    elif nodata == False:
        return permutation_idxs[nan_idxs].tolist() + permutation_idxs[~nan_idxs].tolist()
    else:
        return permutation_idxs[~nan_idxs].tolist()


def rearrange(data, order):
    data = flatten(data)
    return data[order]


# rearrange(data, order(data, nodata)) could be used, but is probably slower than sorting the array directly
def sort(data, asc=True, nodata=None):
    data = flatten(data, dtype=float)

    if asc:
        data_sorted = np.sort(data)
    else:  # [::-1] not possible
        data_sorted = -np.sort(-data)

    nan_idxs = pd.isnull(data_sorted)
    if nodata == True:  # boolean comparison necessary, since nodata can not be only of type boolean
        return data_sorted[~nan_idxs].tolist() + data_sorted[nan_idxs].tolist()
    elif nodata == False:
        return data_sorted[nan_idxs].tolist() + data_sorted[~nan_idxs].tolist()
    else:
        return data_sorted[~nan_idxs].tolist()


def and_(expressions, ignore_nodata=True):
    if is_empty(expressions):
        return np.nan
    expressions_copy = np.array(expressions)
    if not ignore_nodata:
        expressions_copy[np.isnan(expressions)] = False
        if np.all(expressions) and not np.all(expressions_copy):
            return np.nan
        else:
            return np.all(expressions_copy)
    else:
        expressions_valid = expressions_copy[~np.isnan(expressions)]
        return False if len(expressions_valid) == 0 else np.all(expressions_valid)


def or_(expressions, ignore_nodata=True):
    if is_empty(expressions):
        return np.nan
    expressions_copy = np.array(expressions)
    if not ignore_nodata:
        expressions_copy[np.isnan(expressions)] = False
        if np.any(expressions) and not np.any(expressions_copy):
            return np.nan
        else:
            return np.any(expressions_copy)
    else:
        expressions_valid = expressions_copy[~np.isnan(expressions)]
        return False if len(expressions_valid) == 0 else np.any(expressions_valid)


def xor_(expressions, ignore_nodata=True):
    if is_empty(expressions):
        return np.nan
    if not ignore_nodata:
        if np.any(np.isnan(expressions)):
            return np.nan
        else:
            return np.nansum(expressions) == 1
    else:
        return np.nansum(expressions) == 1


def clip(data, min, max):
    data = np.array(data)
    data = np.where(data < min, min, data)
    data = np.where(data > max, max, data)

    return data.tolist()


# old function, not needed anymore
def __sum_first_elem(data):
    for i in range(1, len(data)):
        data[0] += float(data[i])
    return data[0]


def sum_(data, ignore_nodata=True):
    if len(data) < 2:
        err_message = "Addition requires at least two numbers."
        sys.exit(err_message)
    if not ignore_nodata:
        return np.sum(data)
    else:
        return np.nansum(data)


# old function, not needed anymore
def __subtract_first_elem(data):
    for i in range(1, len(data)):
        data[0] -= float(data[i])
    return data[0]


def subtract(data, ignore_nodata=True):
    if len(data) < 2:
        err_message = "Subtraction requires at least two numbers (a minuend and one or more subtrahends)."
        sys.exit(err_message)
    data = np.array(data)
    if not ignore_nodata:
        if np.any(np.isnan(data)):
            return np.nan
        else:
            data = data[~np.isnan(data)]
            return functools.reduce(operator.sub, data)
    else:
        data = data[~np.isnan(data)]
        return functools.reduce(operator.sub, data)


# old function, not needed anymore
def __multiply_first_elem(data):
    for i in range(1, len(data)):
        data[0] *= float(data[i])
    return data[0]


def multiply(data, ignore_nodata=True):
    if len(data) < 2:
        err_message = "Multiplication requires at least two numbers."
        sys.exit(err_message)
    data = np.array(data)
    if not ignore_nodata:
        if np.any(np.isnan(data)):
            return np.nan
        else:
            data = data[~np.isnan(data)]
            return functools.reduce(operator.mul, data, 1)
    else:
        data = data[~np.isnan(data)]
        return functools.reduce(operator.mul, data, 1)


def product(data, ignore_nodata=True):
    return multiply(data, ignore_nodata=ignore_nodata)


# old function, not needed anymore
def __divide_first_elem(data):
    for i in range(1, len(data)):
        data[0] /= float(data[i])
    return data[0]


def divide(data, ignore_nodata=True):
    if len(data) < 2:
        err_message = "Division requires at least two numbers (a dividend and one or more divisors)."
        sys.exit(err_message)
    data = np.array(data)
    if not ignore_nodata:
        if np.any(np.isnan(data)):
            return np.nan
        else:
            data = data[~np.isnan(data)]
            return functools.reduce(operator.truediv, data)
    else:
        data = data[~np.isnan(data)]
        return functools.reduce(operator.truediv, data)


def extrema(data, ignore_nodata=True):
    min_val = min_(data, ignore_nodata=ignore_nodata)
    max_val = max_(data, ignore_nodata=ignore_nodata)
    return [min_val, max_val]



