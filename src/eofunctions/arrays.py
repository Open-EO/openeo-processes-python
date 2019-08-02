import sys
import functools
import operator

import numpy as np
import pandas as pd

from eofunctions.eo_utils import eo_is_valid
from eofunctions.eo_utils import eo_is_empty
from eofunctions.eo_utils import build_multi_dim_index
from eofunctions.eo_utils import process

from eofunctions.errors import IndexOutOfBounds


def eo_array_contains(data, element):
    return element in data


########################################################################################################################
# Array Element Process
########################################################################################################################

@process
def eo_array_element():
    return eoArrayElement()


class eoArrayElement(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, index, return_nodata=False):
        """
        Returns the element of 'data' corresponding to the given index.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        index: int
            Array index.
        return_nodata: bool, optional
            If true (default is False), np.nan will be returned if the element cannot be find. If false and the
            element cannot be found, an IndexOutOfBounds error will be raised.

        Returns
        -------
        count: int
            Count of the data.

        Raises
        ------
        IndexOutOfBounds
            If the element specified by the index cannot be found and 'return_nodata' is false.

        """
        if index >= len(data):
            if not return_nodata:
                raise IndexOutOfBounds()
            else:
                element = np.nan
        else:
            element = data[index]

        return element

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Count Process
########################################################################################################################

@process
def eo_count():
    return eoCount()


class eoCount(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, expression=None):
        """
        Returns the first element of the given array along a specific dimension. If np.nan values are ignored, the first
        valid element is taken.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        expression: obj, optional
            Specifies how the data should be counted.
                - It can be a string, which is then translated to a function leading to an nary boolean output
                (True's are counted).
                - It can be a boolean and true, then all elements are counted.
                - It can be a callable function leading to an nary boolean output (True's are counted).
                - For anything else only the valid elements in 'data' are counted.

        Returns
        -------
        count: int
            Count of the data.

        """
        if expression == True:  # explicit check needed
            count = data.shape[dimension]
        elif isinstance(expression, str):
            count = np.sum(eval(expression), axis=dimension)
        elif callable(expression):
            data = expression(data)
            count = np.sum(data, axis=dimension)
        else:
            count = np.sum(eo_is_valid(data, unary=False), axis=dimension)

        return count

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# First Process
########################################################################################################################

@process
def eo_first():
    return EOFirst()


class EOFirst(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        """
        Returns the first element of the given array along a specific dimension. If np.nan values are ignored, the first
        valid element is taken.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).

        Returns
        -------
        first_elem: object

        """
        first_elem = np.nan  # the default value of the first element
        if not eo_is_empty(data):
            dims = len(data.shape)
            if ignore_nodata:  # skip np.nan values
                nan_mask = ~pd.isnull(data)  # create mask for valid values (not np.nan)
                first_elem_idx = np.argmax(nan_mask, axis=dimension)  # along the chosen dimension, the index of the first valid element will be returned
                string_select = build_multi_dim_index("first_elem_idx", data.shape, dimension)  # create index string
                first_elem = eval("data[{}]".format(string_select))  # select the data according to the index string
            else:  # take the first element, no matter np.nan values are in the array
                strings_select = [":"] * dims
                strings_select[dimension] = "0"  # the index string has a "0" at the place of the chosen dimension
                first_elem = eval("data[{}]".format(",".join(strings_select)))  # select the data according to the index string

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
    return EOLast()


class EOLast(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        """
        Returns the last element of the given array along a specific dimension. If np.nan values are ignored, the last
        valid element is taken.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).

        Returns
        -------
        last_elem: object

        Notes
        -----
        - after flipping, eo first could be called. But this is not done, since some operations will then be executed twice
        """
        last_elem = np.nan  # the default value of the last element
        if not eo_is_empty(data):
            if ignore_nodata:  # skip np.nan values
                data = np.flip(data, axis=dimension)  # flip array so that one can search for the first element
                nan_mask = ~pd.isnull(data)  # create mask for valid values (not np.nan)
                first_elem_idx = np.argmax(nan_mask, axis=dimension)  # along the chosen dimension, the index of the first valid element will be returned
                string_select = build_multi_dim_index("first_elem_idx", data.shape, dimension)  # create index string
                last_elem = eval("data[{}]".format(string_select))  # select the data according to the index string
            else:  # take the last element, no matter np.nan values are in the array
                dims = len(data.shape)
                strings_select = [":"] * dims
                strings_select[dimension] = "-1"  # the index string has a "-1" at the place of the chosen dimension
                last_elem = eval("data[{}]".format(",".join(strings_select)))  # select the data according to the index string

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
    return EOOrder()


# TODO: can nodata=False algorithm be simplified?
class EOOrder(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, asc=True, nodata=None):
        """
        Returns the ascending (default) or descending order of the given data.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        asc: bool, optional
            If true, the order will be ascending, if false the order will be descending (True is default).
        nodata: obj, optional
            Specifies if np.nan values should be removed (nodata=None), put first (nodata=True) or put last
            (nodata=False) in the order.

        Returns
        -------
        permutation_idxs: np.array
            Order of the data.

        Notes
        -----
        - the case with nodata=False is complicated, since a simple nan masking destroys the structure of the array
        - due to the flipping the order of the np.nan values is wrong, but this is ignored, since this order should not be relevant
        """
        if asc:
            permutation_idxs = np.argsort(data, kind='mergesort', axis=dimension)
        else:  # [::-1] not possible
            permutation_idxs = np.argsort(-data, kind='mergesort', axis=dimension)  # to get the indizes in descending order, the sign of the data is changed

        if nodata is None:  # ignore np.nan values
            string_select = build_multi_dim_index("permutation_idxs", data.shape, dimension)
            data_sorted = eval("data[{}]".format(string_select))  # sort the data according to the order computed before to get the np.nan values at the same position
            return permutation_idxs[~np.isnan(data_sorted)]  # mask the np.nan values given in the sorted data in the order
        elif nodata == False:  # put location/index of np.nan values first
            # get np.nan mask from sorted data
            string_select = build_multi_dim_index("permutation_idxs", data.shape, dimension)
            data_sorted = eval("data[{}]".format(string_select))
            nan_idxs = pd.isnull(data_sorted)

            # get np.nan mask from sorted and flipped data
            string_select_flip = build_multi_dim_index("permutation_idxs_flip", data.shape, dimension)
            permutation_idxs_flip = np.flip(permutation_idxs, axis=dimension)
            data_sorted_flip = eval("data[{}]".format(string_select_flip))
            nan_idxs_flip = pd.isnull(data_sorted_flip)

            # flip causes the nan.values to be first, however the order of all other values is also flipped
            # therefore the non np.nan values (i.e. the wrong flipped order) is replaced by the right order given by the original permutation values
            permutation_idxs_flip[~nan_idxs_flip] = permutation_idxs[~nan_idxs]

            return permutation_idxs_flip
        elif nodata == True:  # default argsort behaviour, np.nan values are put last
            return permutation_idxs
        else:
            raise Exception("Status '{}' of argument 'nodata' unknown".format(nodata))

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
    return EORearrange()


class EORearrange(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, order, dimension=0):
        """
        Returns the ascending (default) or descending order of the given data.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        order: np.array
            Order in which 'data' should be sorted.
        dimension: int, optional
            Dimension/axis of interest (0 is default).

        Returns
        -------
        np.array
            Rearranged input data.

        """
        string_select = build_multi_dim_index("order", data.shape, dimension)
        return eval("data[{}]".format(string_select))

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Sort Process
########################################################################################################################

@process
def eo_sort():
    return EOSort()


# TODO: can nodata=False algorithm be simplified?
class EOSort(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, asc=True, nodata=None):
        """
        Returns the data in ascending (default) or descending order.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        asc: bool, optional
            If true, the order will be ascending, if false the order will be descending (True is default).
        nodata: obj, optional
            Specifies if np.nan values should be removed (nodata=None), put first (nodata=True) or put last
            (nodata=False) in the sorted data.

        Returns
        -------
        np.array
            Sorted input data.

        Notes
        -----
        - eo_rearrange(data, eo_order(data, dimension=dimension, asc=asc, nodata=nodata)) could be used, but is
        probably slower than sorting the array directly

        """
        if asc:
            data_sorted = np.sort(data, axis=dimension)
        else:  # [::-1] not possible
            data_sorted = -np.sort(-data, axis=dimension)  # to get the indizes in descending order, the sign of the data is changed

        if nodata is None:  # ignore np.nan values
            nan_idxs = pd.isnull(data_sorted)
            return data_sorted[~nan_idxs]
        elif nodata == False:  # put np.nan values first
            nan_idxs = pd.isnull(data_sorted)
            data_sorted_flip = np.flip(data_sorted, axis=dimension)
            nan_idxs_flip = pd.isnull(data_sorted_flip)
            data_sorted_flip[~nan_idxs_flip] = data_sorted[~nan_idxs]
            return data_sorted_flip
        elif nodata == True:  # default sort behaviour, np.nan values are put last
            return data_sorted
        else:
            raise Exception("Status '{}' of argument 'nodata' unknown".format(nodata))

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Clip Process
########################################################################################################################

@process
def eo_clip():
    return EOClip()


class EOClip(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, min, max):
        """
        Returns the clipped input data, i.e. all values below 'min' are set to 'min' and all values above 'max' are set
        to 'max'.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        min: float, int
            Minimum value allowed to be in the array.
        max: float, int
            Maximum value allowed to be in the array.

        Returns
        -------
        np.array
            Clipped input data.
        """
        data = np.where(data < min, min, data)
        data = np.where(data > max, max, data)

        return data

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass
