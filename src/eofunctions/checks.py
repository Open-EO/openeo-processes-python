import xarray
import dask

import numpy as np
import pandas as pd

from eofunctions.utils import process


# TODO: test if this works for different data types
def is_empty(data):
    """
    Checks if an object is empty (its length is zero) or not.

    Parameters
    ----------
    data : object
        Any Class instance, which has a __length__ class method.

    Returns
    -------
    bool :
        True if object is emtpy, False if not.

    """
    if len(data) == 0:
        return True
    else:
        return False


def eo_is_nodata(x):
    """Checks for None or np.nan values."""
    return pd.isnull(x)


########################################################################################################################
# Is NaN Process
########################################################################################################################

@process
def eo_is_nan():
    return EOIsNan()


# TODO: add string and other non-numerical data type handling for numpy arrays.
class EOIsNan(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num(data):
        """
        Checks whether the specified value x is not a number (often abbreviated as NaN).
        All non-numeric data types also return True.

        Parameters
        ----------
        data: float, int, str

        Returns
        -------
        bool
        """
        if isinstance(data, (int, float)):
            return pd.isnull(data)
        else:
            return True

    @staticmethod
    def exec_np(data):
        """
        Checks whether the specified value x is not a number (often abbreviated as NaN).
        All non-numeric data types also return True.

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.

        Returns
        -------
        bool
        """
        return pd.isnull(data)

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Is Valid Process
########################################################################################################################

@process
def is_valid():
    return IsValid()


class IsValid(object):


    @staticmethod
    def exec_num(data):
        """
        Checks whether the specified value 'data' is valid. A value is considered valid if it is not a no-data value
        (None, np.nan) and a finite number (only if 'data' is a number).

        Parameters
        ----------
        data: float, int, str

        Returns
        -------
        bool

        Notes
        -----
        The definition of finite and infinite numbers follows the IEEE Standard 754.
        """
        if data not in [np.nan, np.inf, None]:
            return True
        else:
            return False

    @staticmethod
    def exec_np(data, reduce=True):
        """
        Checks whether the specified array 'data' is valid. An array value is considered valid if it is not a no-data value
        (None, np.nan) and a finite number (only if the array value is a number).

        Parameters
        ----------
        data: np.array
            Input data as a numpy array.
        reduce: bool
            If True a boolean value will be returned, if False an array of boolean values.

        Returns
        -------
        bool, np.array
        """
        if is_empty(data):
            return False
        else:
            is_valid = (~pd.isnull(data) & (data != np.inf))
            if reduce:  # reduce all values, i.e. is every boolean value True?
                return is_valid.all()
            else:
                return is_valid

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


