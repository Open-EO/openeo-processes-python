import numpy as np
import pandas as pd
import sys
import warnings
import functools
import operator
import copy
from eofunctions.math import min_, max_, is_empty, is_valid, list2nparray


# TODO: documentation missing!
# TODO: restructure function packages, they currently interfere with each other

# def flatten(data, dtype=object):
#     if not isinstance(data, np.ndarray):
#         data = np.array(data, dtype=dtype)
#     if data.ndim > 1:
#         warn_message = "Array has more than one dimension. Flattening will be applied."
#         warnings.warn(warn_message)
#         data = data.flatten()
#
#     return data
def __build_multi_dim_index(index_name, shape, axis):
    dims = len(shape)
    expand_dim_exprs = [["None"] * (dims - 1)] * (dims - 1)
    for i, elem in enumerate(expand_dim_exprs):
        elem_cp = copy.deepcopy(elem)
        elem_cp[i] = ":"
        expand_dim_exprs[i] = ",".join(elem_cp)
    expand_dim_exprs.insert(axis, None)
    strings_select = []
    for i, n in enumerate(shape):
        if i == axis:
            strings_select.append(index_name)
        else:
            strings_select.append("np.arange({})[{}]".format(n, expand_dim_exprs[i]))

    return ",".join(strings_select)


def array_contains(data, element):
    return element in data


def array_element(data, index, return_nodata=False):
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
def count(data, axis=0, expression=None):
    if isinstance(data, list):
        data = list2nparray(data)

    if expression == True:
        if axis is not None:
            num_of_elems = data.shape[axis]
        else:
            num_of_elems = functools.reduce(operator.mul, data.shape, 1)
    elif isinstance(expression, str):
        if axis is not None:
            num_of_elems = np.sum(eval(expression.replace('x', 'data')), axis=axis)
        else:
            num_of_elems = np.sum(eval(expression.replace('x', 'data')))
    else:
        if axis is not None:
            num_of_elems = np.sum(is_valid(data), axis=axis)
        else:
            num_of_elems = np.sum(is_valid(data))

    return num_of_elems


def first(data, axis=0, ignore_nodata=True):
    if not isinstance(data, np.ndarray):
        data = list2nparray(data)

    first_elem = np.nan
    if not is_empty(data):
        dims = len(data.shape)
        if ignore_nodata:
            nan_mask = ~pd.isnull(data)
            first_elem_idx = np.argmax(nan_mask, axis=axis)
            string_select = __build_multi_dim_index("first_elem_idx", data.shape, axis)
            first_elem = eval("data[{}]".format(string_select))
        else:
            strings_select = [":"]*dims
            strings_select[axis] = "0"
            first_elem = eval("data[{}]".format(",".join(strings_select)))

    return first_elem


def last(data, axis=0, ignore_nodata=True):
    if not isinstance(data, np.ndarray):
        data = list2nparray(data)

    last_elem = np.nan
    if not is_empty(data):
        dims = len(data.shape)
        if ignore_nodata:
            nan_mask = ~pd.isnull(data)
            last_elem_idx = np.argmax(np.flip(nan_mask, axis=axis), axis=axis)
            string_select = __build_multi_dim_index("last_elem_idx", data.shape, axis)
            last_elem = eval("data[{}]".format(string_select))
        else:
            strings_select = [":"] * dims
            strings_select[axis] = "-1"
            last_elem = eval("data[{}]".format(",".join(strings_select)))

    return last_elem


def order(data, axis=0, asc=True, nodata=None):
    if isinstance(data, list):
        data = list2nparray(data)

    data = data.astype(float)

    if asc:
        permutation_idxs = np.argsort(data, kind='mergesort', axis=axis)
    else:  # [::-1] not possible
        permutation_idxs = np.argsort(-data, kind='mergesort', axis=axis)

    if nodata == False:  # TODO: can this be done in an easier way?
        string_select = __build_multi_dim_index("permutation_idxs", data.shape, axis)
        string_select_flip = __build_multi_dim_index("permutation_idxs_flip", data.shape, axis)
        data_sorted = eval("data[{}]".format(string_select))
        nan_idxs = pd.isnull(data_sorted)
        permutation_idxs_flip = np.flip(permutation_idxs, axis=axis)
        data_sorted_flip = eval("data[{}]".format(string_select_flip))
        nan_idxs_flip = pd.isnull(data_sorted_flip)
        permutation_idxs_flip[~nan_idxs_flip] = permutation_idxs[~nan_idxs]
        return permutation_idxs_flip
    else:
        return permutation_idxs


def rearrange(data, order, axis=0):
    if isinstance(data, list):
        data = list2nparray(data)
    string_select = __build_multi_dim_index("order", data.shape, axis)
    return eval("data[{}]".format(string_select))


# rearrange(data, order(data, nodata)) could be used, but is probably slower than sorting the array directly
def sort(data, axis=0, asc=True, nodata=None):
    if isinstance(data, list):
        data = list2nparray(data)

    data = data.astype(float)

    if asc:
        data_sorted = np.sort(data, axis=axis)
    else:  # [::-1] not possible
        data_sorted = -np.sort(-data, axis=axis)

    if nodata == False:
        nan_idxs = pd.isnull(data_sorted)
        data_sorted_flip = np.flip(data_sorted, axis=axis)
        nan_idxs_flip = pd.isnull(data_sorted_flip)
        data_sorted_flip[~nan_idxs_flip] = data_sorted[~nan_idxs]
        return data_sorted_flip
    else:
        return data_sorted


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
    if isinstance(data, list):
        data = list2nparray(data)

    data = np.where(data < min, min, data)
    data = np.where(data > max, max, data)

    return data


# old function, not needed anymore
def __sum_first_elem(data):
    for i in range(1, len(data)):
        data[0] += float(data[i])
    return data[0]


def sum_(data, axis=0, ignore_nodata=True):
    number_elems = count(data, axis=axis, expression=True)
    if number_elems < 2:
        err_message = "Addition requires at least two numbers."
        sys.exit(err_message)
    if not ignore_nodata:
        return np.sum(data, axis=axis)
    else:
        return np.nansum(data, axis=axis)


# old function, not needed anymore
def __subtract_first_elem(data):
    for i in range(1, len(data)):
        data[0] -= float(data[i])
    return data[0]


# TODO: faster implemention?
def subtract(data, axis=0, ignore_nodata=True):
    if isinstance(data, list):
        data = list2nparray(data)

    number_elems = count(data, axis=axis, expression=True)
    if number_elems < 2:
        err_message = "Subtraction requires at least two numbers (a minuend and one or more subtrahends)."
        sys.exit(err_message)

    nan_idxs = pd.isnull(data)
    if not ignore_nodata:
        if np.any(nan_idxs):
            return np.nan
        else:
            data = data[~nan_idxs]
            return np.apply_along_axis(lambda data: functools.reduce(operator.sub, data), axis, data)
    else:
        data = data[~nan_idxs]
        return np.apply_along_axis(lambda data: functools.reduce(operator.sub, data), axis, data)


# old function, not needed anymore
def __multiply_first_elem(data):
    for i in range(1, len(data)):
        data[0] *= float(data[i])
    return data[0]


# TODO: faster implemention?
def multiply(data, axis=0, ignore_nodata=True):
    if isinstance(data, list):
        data = list2nparray(data)

    number_elems = count(data, axis=axis, expression=True)
    if number_elems < 2:
        err_message = "Multiplication requires at least two numbers."
        sys.exit(err_message)

    nan_idxs = pd.isnull(data)
    if not ignore_nodata:
        if np.any(nan_idxs):
            return np.nan
        else:
            data = data[~nan_idxs]
            return np.apply_along_axis(lambda data: functools.reduce(operator.mul, data, 1), axis, data)
    else:
        data = data[~nan_idxs]
        return np.apply_along_axis(lambda data: functools.reduce(operator.mul, data, 1), axis, data)


def product(data, ignore_nodata=True):
    return multiply(data, ignore_nodata=ignore_nodata)


# old function, not needed anymore
def __divide_first_elem(data):
    for i in range(1, len(data)):
        data[0] /= float(data[i])
    return data[0]


# TODO: faster implemention?
def divide(data, axis=0, ignore_nodata=True):
    if isinstance(data, list):
        data = list2nparray(data)

    number_elems = count(data, axis=axis, expression=True)
    if number_elems < 2:
        err_message = "Division requires at least two numbers (a dividend and one or more divisors)."
        sys.exit(err_message)

    nan_idxs = pd.isnull(data)
    if not ignore_nodata:
        if np.any(nan_idxs):
            return np.nan
        else:
            data = data[~nan_idxs]
            return np.apply_along_axis(lambda data: functools.reduce(operator.truediv, data), axis, data)
    else:
        data = data[~nan_idxs]
        return np.apply_along_axis(lambda data: functools.reduce(operator.truediv, data), axis, data)


def extrema(data, axis=0, ignore_nodata=True):
    min_val = min_(data, axis=axis, ignore_nodata=ignore_nodata)
    max_val = max_(data, axis=axis, ignore_nodata=ignore_nodata)
    return [min_val, max_val]



