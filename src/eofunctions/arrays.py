import numpy as np
import pandas as pd
import sys
import functools
import operator
from eofunctions.math import eo_min, eo_max, eo_is_valid
from eofunctions.eo_utils import is_empty, list2nparray, build_multi_dim_index

from eofunctions.eo_utils import process


def eo_array_contains(data, element):
    return element in data


def eo_array_element(data, index, return_nodata=False):
    element = np.nan
    if index >= len(data):
        if not return_nodata:
            err_message = "The array has no element with the specified index."
            sys.exit(err_message)
    else:
        element = data[index]

    return element


########################################################################################################################
# First Process
########################################################################################################################

@process
def eo_first():
    return eoFirst()


class eoFirst(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        first_elem = np.nan
        if not is_empty(data):
            dims = len(data.shape)
            if ignore_nodata:
                nan_mask = ~pd.isnull(data)
                first_elem_idx = np.argmax(nan_mask, axis=dimension)
                string_select = build_multi_dim_index("first_elem_idx", data.shape, dimension)
                first_elem = eval("data[{}]".format(string_select))
            else:
                strings_select = [":"] * dims
                strings_select[dimension] = "0"
                first_elem = eval("data[{}]".format(",".join(strings_select)))

        return first_elem

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Last Process
########################################################################################################################

@process
def eo_last():
    return eoLast()


class eoLast(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        last_elem = np.nan
        if not is_empty(data):
            dims = len(data.shape)
            if ignore_nodata:
                nan_mask = ~pd.isnull(data)
                last_elem_idx = np.argmax(np.flip(nan_mask, axis=dimension), axis=dimension)
                string_select = build_multi_dim_index("last_elem_idx", data.shape, dimension)
                last_elem = eval("data[{}]".format(string_select))
            else:
                strings_select = [":"] * dims
                strings_select[dimension] = "-1"
                last_elem = eval("data[{}]".format(",".join(strings_select)))

        return last_elem

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Order Process
########################################################################################################################

@process
def eo_order():
    return eoOrder()


class eoOrder(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, asc=True, nodata=None):
        data = data.astype(float)

        if asc:
            permutation_idxs = np.argsort(data, kind='mergesort', axis=dimension)
        else:  # [::-1] not possible
            permutation_idxs = np.argsort(-data, kind='mergesort', axis=dimension)

        if nodata is None:
            string_select = build_multi_dim_index("permutation_idxs", data.shape, dimension)
            data_sorted = eval("data[{}]".format(string_select))
            return permutation_idxs[~np.isnan(data_sorted)]
        elif nodata == False:  # TODO: can this be done in an easier way?
            string_select = build_multi_dim_index("permutation_idxs", data.shape, dimension)
            string_select_flip = build_multi_dim_index("permutation_idxs_flip", data.shape, dimension)
            data_sorted = eval("data[{}]".format(string_select))
            nan_idxs = pd.isnull(data_sorted)
            permutation_idxs_flip = np.flip(permutation_idxs, axis=dimension)
            data_sorted_flip = eval("data[{}]".format(string_select_flip))
            nan_idxs_flip = pd.isnull(data_sorted_flip)
            permutation_idxs_flip[~nan_idxs_flip] = permutation_idxs[~nan_idxs]
            return permutation_idxs_flip
        else:
            return permutation_idxs

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass

########################################################################################################################
# Rearrange Process
########################################################################################################################

@process
def eo_rearrange():
    return eoOrder()


class eoRearrange(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, order, dimension=0):
        string_select = build_multi_dim_index("order", data.shape, dimension)
        return eval("data[{}]".format(string_select))

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


# rearrange(data, order(data, nodata)) could be used, but is probably slower than sorting the array directly
def eo_sort(data, axis=0, asc=True, nodata=None):

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


def eo_and(expressions, ignore_nodata=True):
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


def eo_or(expressions, ignore_nodata=True):
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


def eo_xor(expressions, ignore_nodata=True):
    if is_empty(expressions):
        return np.nan
    if not ignore_nodata:
        if np.any(np.isnan(expressions)):
            return np.nan
        else:
            return np.nansum(expressions) == 1
    else:
        return np.nansum(expressions) == 1


def eo_clip(data, min, max):

    data = np.where(data < min, min, data)
    data = np.where(data > max, max, data)

    return data


def extrema(data, axis=0, ignore_nodata=True):
    min_val = eo_min(data, axis=axis, ignore_nodata=ignore_nodata)
    max_val = eo_max(data, axis=axis, ignore_nodata=ignore_nodata)
    return [min_val, max_val]



