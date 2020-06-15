import numpy as np
from openeo_processes.utils import process
from openeo_processes.comparison import is_empty


########################################################################################################################
# And Process
########################################################################################################################

@process
def and_():
    """
    Returns class instance of `And`.
    For more details, please have a look at the implementations inside `And`.

    Returns
    -------
    And :
        Class instance implementing all 'and' processes.

    """
    return And()


class And:
    """
    Class implementing all 'and' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Checks if both values are true.
        Evaluates parameter `x` before `y` and stops once the outcome is unambiguous.
        If any argument is None, the result will be None if the outcome is ambiguous.

        Parameters
        ----------
        x : bool
            A boolean value.
        y : bool
            A boolean value.

        Returns
        -------
        bool :
            Boolean result of the logical AND.

        """
        return x and y if None not in [x, y] else None

    @staticmethod
    def exec_np(x, y):
        """
        Checks if both arrays are true.
        Evaluates parameter `x` before `y` and stops once the outcome is unambiguous.
        If any argument is np.nan, the result will be np.nan if the outcome is ambiguous.

        Parameters
        ----------
        x : np.array or bool
            A boolean value.
        y : np.array or bool
            A boolean value.

        Returns
        -------
        np.array :
            Boolean result of the logical AND.

        """
        return x & y

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Or Process
########################################################################################################################

@process
def or_():
    """
    Returns class instance of `Or`.
    For more details, please have a look at the implementations inside `Or`.

    Returns
    -------
    Or :
        Class instance implementing all 'or' processes.

    """
    return Or()


class Or:
    """
    Class implementing all 'or' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Checks if at least one of the values is True. Evaluates parameter `x` before `y` and stops once the outcome
        is unambiguous. If a component is None, the result will be None if the outcome is ambiguous.

        Parameters
        ----------
        x : bool
            A boolean value.
        y : bool
            A boolean value.

        Returns
        -------
        bool :
            Boolean result of the logical OR.

        """

        return None if None in [x, y] and False in [x, y] else x or y

    @staticmethod
    def exec_np(x, y):
        """
        Checks if at least one of the array values is True. Evaluates parameter `x` before `y` and stops once the
        outcome is unambiguous. If a component is np.nan, the result will be np.nan if the outcome is ambiguous.

        Parameters
        ----------
        x : bool
            A boolean value.
        y : bool
            A boolean value.

        Returns
        -------
        np.array :
            Boolean result of the logical OR.

        """
        return x | y

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Xor Process
########################################################################################################################

@process
def xor():
    """
    Returns class instance of `Xor`.
    For more details, please have a look at the implementations inside `Xor`.

    Returns
    -------
    Xor :
        Class instance implementing all 'xor' processes.

    """
    return Xor()


class Xor:
    """
    Class implementing all 'xor' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Checks if exactly one of the values is true. If a component is None, the result will be None if the outcome
        is ambiguous.

        Parameters
        ----------
        x : bool
            A boolean value.
        y : bool
            A boolean value.

        Returns
        -------
        bool :
            Boolean result of the logical XOR.

        """
        return sum([x, y]) == 1 if None not in [x, y] else None

    @staticmethod
    def exec_np(x, y):
        """
        Checks if exactly one of the array values is true. If a component is np.nan, the result will be np.nan if the
        outcome is ambiguous.

        Parameters
        ----------
        x : bool
            A boolean value.
        y : bool
            A boolean value.

        Returns
        -------
        np.array :
            Boolean result of the logical XOR.

        """
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            return np.nan
        else:
            return (x + y) == 1

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Not Process
########################################################################################################################

@process
def not_():
    """
    Returns class instance of `Not`.
    For more details, please have a look at the implementations inside `Not`.

    Returns
    -------
    Not
        Class instance implementing all 'not' processes.

    """
    return Not()


class Not:
    """
    Class implementing all 'not' processes.
    """

    @staticmethod
    def exec_num(x):
        """
        Inverts a boolean so that True gets False and False gets True.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : bool
            Boolean value to invert.

        Returns
        -------
        bool :
            Inverted boolean value.

        """
        return not x if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Inverts booleans so that True/1 gets False/0 and False/0 gets True/1.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Boolean values to invert.

        Returns
        -------
        np.array :
            Inverted boolean values.

        """
        return ~x

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# If Process
########################################################################################################################

@process
def if_():
    """
    Returns class instance of `If`.
    For more details, please have a look at the implementations inside `If`.

    Returns
    -------
    If :
        Class instance implementing all 'if' processes.

    """
    return If()


class If:
    """
    Class implementing all 'if' processes.
    """

    @staticmethod
    def exec_num(value, accept, reject=None):
        """
        If the value passed is True, returns the value of the `accept` parameter,
        otherwise returns the value of the `reject` parameter.

        Parameters
        ----------
        value : bool
            A boolean value.
        accept : object
            A value that is returned if the boolean value is True.
        reject : object, optional
            A value that is returned if the boolean value is not True. Defaults to None.

        Returns
        -------
        object :
            Either the `accept` or `reject` argument depending on the given boolean value.

        """
        return accept if value else reject

    @staticmethod
    def exec_np(value, accept, reject=np.nan):
        """
        If the array value passed is True, returns the value of the `accept` parameter,
        otherwise returns the value of the `reject` parameter.

        Parameters
        ----------
        value : np.array
            A boolean array.
        accept : object
            A value that is returned if the boolean value is True.
        reject : object, optional
            A value that is returned if the boolean value is not True. Defaults to None.

        Returns
        -------
        np.array :
            Either the `accept` or `reject` argument depending on the given boolean value.

        """

        return np.where(value, accept, reject)

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Any Process
########################################################################################################################

@process
def any_():
    """
    Returns class instance of `Any`.
    For more details, please have a look at the implementations inside `Any`.

    Returns
    -------
    Any :
        Class instance implementing all 'any' processes.

    """
    return Any()


class Any:
    """
    Class implementing all 'any' processes.
    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Checks if any (i.e. at least one) value is True. Evaluates all values from the first to the last element and
        stops once the outcome is unambiguous. If only one value is given, the process evaluates to the given value.
        If no value is given (i.e. the array is empty) the process returns None.
        By default all NaN values are ignored so that the process returns np.nan if all values are NaN,
        True if at least one of the other values is True and False otherwise.
        Setting the `ignore_nodata` flag to False considers NaN values so that np.nan is a valid logical object.
        If a component is np.nan, the result will be np.nan if the outcome is ambiguous.

        Parameters
        ----------
        data : np.array
            A boolean array. An empty array resolves always with None.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to evaluate 'any' along (default is 0).

        Returns
        -------
        np.array :
            Boolean result of the logical operation.

        """
        if is_empty(data):
            return np.nan

        if len(data.shape) == 1:  # exand data if it has only one dimension
            data = data[:, None]

        nan_ar = np.isnan(data)
        if ignore_nodata:
            nan_mask = np.all(nan_ar, axis=dimension)
            data[nan_ar] = False
        else:
            nan_mask = np.any(nan_ar, axis=dimension)

        data_any = np.any(data, axis=dimension)
        data_any = data_any.astype(np.float32)  # convert to float to store NaN values
        data_any[nan_mask] = np.nan
        return data_any

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# All Process
########################################################################################################################

@process
def all_():
    """
    Returns class instance of `All`.
    For more details, please have a look at the implementations inside `All`.

    Returns
    -------
    All :
        Class instance implementing all 'all' processes.

    """
    return All()


class All:
    """
    Class implementing all 'all' processes.
    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Checks if all of the values are True. Evaluates all values from the first to the last element and stops once
        the outcome is unambiguous. If only one value is given, the process evaluates to the given value. If no value
        is given (i.e. the array is empty) the process returns None. By default all no-data values are ignored so
        that the process returns np.nan if all values are no-data, True if all other values are True and False
        otherwise. Setting the `ignore_nodata` flag to False considers no-data values so that np.nan is a valid
        logical object. If a component is np.nan, the result will be np.nan if the outcome is ambiguous.

        Parameters
        ----------
        data : np.array
            A boolean array. An empty array resolves always with None.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to evaluate 'all' along (default is 0).

        Returns
        -------
        np.array :
            Boolean result of the logical operation.

        """
        if is_empty(data):
            return np.nan

        if len(data.shape) == 1:  # exand data if it has only one dimension
            data = data[:, None]

        nan_ar = np.isnan(data)
        if ignore_nodata:
            nan_mask = np.all(nan_ar, axis=dimension)
            data_all = np.all(data, axis=dimension)
        else:
            nan_mask = np.any(nan_ar, axis=dimension)  # flag elements with at least one NaN value along the dimension
            data_all = np.all(data, axis=dimension)
            nan_mask = nan_mask & data_all  # reset nan mask to only mask trues and NaN values

        data_all = data_all.astype(np.float32)  # convert to float to store NaN values
        data_all[nan_mask] = np.nan
        return data_all

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass