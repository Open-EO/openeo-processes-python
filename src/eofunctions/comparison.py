"""
Code that can be used for filtering the data. Returns a boolean array where the values that satisfied the
given filter criteria are represented by True and those that did not by False. Should correspond to some of the codes in
https://open-eo.github.io/openeo-api/v/0.4.0/processreference/ - in comparison the gt, gte, lt, lte and between functions
"""


import sys
import numpy as np
import pandas as pd
from eofunctions.math import is_valid
from eofunctions.texts import str2time


def comparison(in_array, filter_type, limits):
    """
    Filters the raster of data returning a boolean array with results.
    Filter types available: 'lt', 'lte', 'gt', 'gte' and 'between'
    Limits: value (fot lt, lte, gt and gte) or list of values (for between)
    """

    # Convert to 2d numpy array if input comes from gdal pixel function
    if (isinstance(in_array, (list, tuple))) and (len(in_array) == 1):
        in_array = in_array[0]

    # Sanity check
    if filter_type in ['lt', 'lte', 'gt', 'gte']:
        if not isinstance(limits, int) and not isinstance(limits, float):
            err_message = 'For filter type lt, lte, gt and gte, the limits have to be int or float'
            sys.exit(err_message)
    if filter_type in ['between']:
        if not isinstance(limits, list) or not len(limits) == 2:
            err_message = 'For filter type between, the limits have to be a list of 2 numbers (int or float)'
            sys.exit(err_message)

    # creating the output array for various filter types
    if filter_type == 'lt':
        out_array = in_array < limits
    elif filter_type == 'lte':
        out_array = in_array <= limits
    elif filter_type == 'gt':
        out_array = in_array > limits
    elif filter_type == 'gte':
        out_array = in_array >= limits
    elif filter_type == 'between':
        out_array = (in_array >= limits[0]) & (in_array <= limits[1])
    return out_array

def eq(x, y, delta=None, case_sensitive=True):
    if not is_valid(x) or not is_valid(y):
        return np.nan
    if (type(x) in [float, int]) and (type(y) in [float, int]):
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


def neq(x, y, delta=None, case_sensitive=True):
    eq_res = eq(x, y, delta=delta, case_sensitive=case_sensitive)
    if np.isnan(eq_res):
        return np.nan
    else:
        return not eq_res


def gt(x, y):
    if not is_valid(x) or not is_valid(y):
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


def gte(x, y):
    if not is_valid(x) or not is_valid(y):
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


def lt(x, y):
    if not is_valid(x) or not is_valid(y):
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


def lte(x, y):
    if not is_valid(x) or not is_valid(y):
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


def between(x, min, max, exclude_max=False):
    if not is_valid(x) or not is_valid(min) or not is_valid(max):
        return np.nan

    if type(min) == str:
        min = str2time(min)
    if type(max) == str:
        max = str2time(max)
    if type(x) == str:
        x = str2time(x)

    min = np.min(np.array(min))
    if exclude_max:
        max = np.min(np.array(max))
    else:
        max = np.max(np.array(max))

    if lt(max, min):
        return False

    if not hasattr(x, '__iter__'):
        x = list([x])
    else:
        x = list(x)

    is_between = True
    for elem in x:  # both boundaries of x have to be inside [min;max]
        if exclude_max:
            is_between &= gte(elem, min) & lt(elem, max)
        else:
            is_between &= gte(elem, min) & lte(elem, max)

    return is_between

def if_(expression, accept=True, reject=False):
    if pd.isnull(expression):
        return np.nan
    else:
        return accept if expression else reject
