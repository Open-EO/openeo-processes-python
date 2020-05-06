"""
Code that can be used for filtering the data. Returns a boolean array where the values that satisfied the
given filter criteria are represented by True and those that did not by False. Should correspond to some of the codes in
https://open-eo.github.io/openeo-api/v/0.4.0/processreference/ - in comparison the gt, gte, lt, lte and between functions
"""

import numpy as np
import pandas as pd

from eofunctions.utils import str2time
from eofunctions.checks import eo_is_empty
from eofunctions.checks import eo_is_valid

# TODO: refactor all functions to class functions


def eo_and(expressions, ignore_nodata=True):
    """Checks if all of the array values are True."""
    if eo_is_empty(expressions):
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
    """Checks if at least one of the array values is True."""
    if eo_is_empty(expressions):
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
    """Checks if exactly one of the array values is True."""
    if eo_is_empty(expressions):
        return np.nan
    if not ignore_nodata:
        if np.any(np.isnan(expressions)):
            return np.nan
        else:
            return np.nansum(expressions) == 1
    else:
        return np.nansum(expressions) == 1


def eo_eq(x, y, delta=None, case_sensitive=True):
    if not eo_is_valid(x) or not eo_is_valid(y):
        return np.nan
    if (type(x) in [float, int, np.ndarray]) and (type(y) in [float, int, np.ndarray]):
        if type(delta) in [float, int]:
            return np.isclose(x, y, atol=delta)
        else:
            return x == y
    elif (type(x) == str) and (type(y) == str):
        x_time = str2time(x)
        y_time = str2time(y)
        if x_time is None or y_time is None:
            if case_sensitive:
                return x == y
            else:
                return x.lower() == y.lower()
        else:
            return x_time == y_time
    else:
        return False


def eo_neq(x, y, delta=None, case_sensitive=True):
    eq_res = eo_eq(x, y, delta=delta, case_sensitive=case_sensitive)
    if np.isnan(eq_res):
        return np.nan
    else:
        return not eq_res


def eo_gt(x, y):
    if not eo_is_valid(x) or not eo_is_valid(y):
        return np.nan
    elif (type(x) == str) and (type(y) == str):
        x_time = str2time(x)
        y_time = str2time(y)
        if x_time is None or y_time is None:
            return False
        else:
            if type(x_time) == tuple and type(y_time) == tuple:
                return x_time[0] > y_time[0]
            elif (type(x_time) == tuple and type(y_time) != tuple) or (type(y_time) == tuple and type(x_time) != tuple):
                return False
            else:
                return x_time > y_time
    else:
        return x > y


def eo_gte(x, y):
    if not eo_is_valid(x) or not eo_is_valid(y):
        return np.nan
    elif (type(x) == str) and (type(y) == str):
        x_time = str2time(x)
        y_time = str2time(y)
        if x_time is None or y_time is None:
            return False
        else:
            if type(x_time) == tuple and type(y_time) == tuple:
                return x_time[0] >= y_time[0]
            elif (type(x_time) == tuple and type(y_time) != tuple) or (type(y_time) == tuple and type(x_time) != tuple):
                return False
            else:
                return x_time >= y_time
    else:
        return x >= y


def eo_lt(x, y):
    if not eo_is_valid(x) or not eo_is_valid(y):
        return np.nan
    elif (type(x) == str) and (type(y) == str):
        x_time = str2time(x)
        y_time = str2time(y)
        if x_time is None or y_time is None:
            return False
        else:
            if type(x_time) == tuple and type(y_time) == tuple:
                return x_time[0] < y_time[0]
            elif (type(x_time) == tuple and type(y_time) != tuple) or (type(y_time) == tuple and type(x_time) != tuple):
                return False
            else:
                return x_time < y_time
    else:
        return x < y


def eo_lte(x, y):
    if not eo_is_valid(x) or not eo_is_valid(y):
        return np.nan
    elif (type(x) == str) and (type(y) == str):
        x_time = str2time(x)
        y_time = str2time(y)
        if x_time is None or y_time is None:
            return False
        else:
            if type(x_time) == tuple and type(y_time) == tuple:
                return x_time[0] <= y_time[0]
            elif (type(x_time) == tuple and type(y_time) != tuple) or (type(y_time) == tuple and type(x_time) != tuple):
                return False
            else:
                return x_time <= y_time
    else:
        return x <= y


def eo_between(x, min, max, exclude_max=False):
    if not eo_is_valid(x) or not eo_is_valid(min) or not eo_is_valid(max):
        return np.nan

    if type(min) == str:
        min = str2time(min)
    if type(max) == str:
        max = str2time(max)
    if type(x) == str:
        x = str2time(x)

    min = np.min(np.array(min))  # cast to np.array because of datetime objects
    if exclude_max:
        max = np.min(np.array(max))  # cast to np.array because of datetime objects
    else:
        max = np.max(np.array(max))  # cast to np.array because of datetime objects

    if eo_lt(max, min):
        return False

    if not hasattr(x, '__iter__'):
        x = list([x])
    else:
        x = list(x)

    is_between = True
    for elem in x:  # both boundaries of x have to be inside [min;max]
        if exclude_max:
            is_between &= eo_gte(elem, min) & eo_lt(elem, max)
        else:
            is_between &= eo_gte(elem, min) & eo_lte(elem, max)

    return is_between


def eo_if(expression, accept=True, reject=False):
    if pd.isnull(expression):
        return np.nan
    else:
        return accept if expression else reject
