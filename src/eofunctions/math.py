import sys
import numpy as np
from eofunctions.texts import str2time
from datetime import timedelta
import pandas as pd


# TODO: test if operations are faster using lists or numpy arrays
# TODO: discuss when one should convert None values to np.nan
# TODO: quantiles with nans are not working properly/really slow -> own implementation (e.g. like in SGRT)?
def list2nparray(x):
    x_tmp = np.array(x)
    if x_tmp.dtype.kind in ['U', 'S']:
        x = np.array(x, dtype=object)
    else:
        x = x_tmp

    return x


def e():
    return np.e


def pi():
    return np.pi


def is_nan(x):
    if isinstance(x, list):
        x = list2nparray(x)

    if isinstance(x, (int, float, np.ndarray)):
        return pd.isnull(x)
    else:
        return True


def is_nodata(x):
    if isinstance(x, list):
        x = list2nparray(x)

    if isinstance(x, np.ndarray):
        return pd.isnull(x)
    else:
        if x in [np.nan, None]:
            return True
        else:
            return False


def is_valid(x):
    if isinstance(x, list):
        x = list2nparray(x)

    if isinstance(x, np.ndarray):
        return ~pd.isnull(x) & (x != np.inf)
    else:
        if x not in [np.nan, np.inf, None]:
            return True
        else:
            return False


def is_empty(data):
    if len(data) == 0:
        return True
    else:
        return False


def int_(x):
    if isinstance(x, list):
        x = list2nparray(x)

    if isinstance(x, np.ndarray):
        return x.astype(int)
    else:
        return int(x)


def floor(x):
    return np.floor(x)


def ceil(x):
    return np.ceil(x)


def round_(x, p=0):
    if isinstance(x, list):
        x = list2nparray(x)

    if isinstance(x, np.ndarray):
        return np.around(x, p)
    else:
        return round(x, p)


def min_(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.min(data, axis=axis)
    else:
        return np.nanmin(data, axis=axis)


def max_(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.max(data, axis=axis)
    else:
        return np.nanmax(data, axis=axis)


def mean(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.mean(data, axis=axis)
    else:
        return np.nanmean(data, axis=axis)


def median(data, axis=0, ignore_nodata=True):
    return np.squeeze(quantiles(data, axis=axis, probabilities=[0.5], ignore_nodata=ignore_nodata), axis=0)


def sd(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.std(data, axis=axis, ddof=1)
    else:
        return np.nanstd(data, axis=axis, ddof=1)


def variance(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.var(data, axis=axis, ddof=1)
    else:
        return np.nanvar(data, axis=axis, ddof=1)


def quantiles(data, axis=0, probabilities=None, q=None, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if isinstance(data, list):
        data = list2nparray(data)

    if (probabilities is not None) and (q is not None):
        err_message = "The process 'quantiles' only allows that either the 'probabilities' or the 'q' parameter is set."
        sys.exit(err_message)

    if probabilities is not None:
        if is_empty(data):
            return [np.nan] * len(probabilities)
        if not ignore_nodata:
            return np.percentile(data, list(np.array(probabilities)*100.), axis=axis)
        else:
            return np.nanpercentile(data, list(np.array(probabilities)*100.), axis=axis)
    elif q is not None:
        probabilities = list(np.arange(0, 100, 100./q))[1:]
        if is_empty(data):
            return [np.nan] * len(probabilities)
        if not ignore_nodata:
            return np.percentile(data, probabilities, axis=axis)
        else:
            return np.nanpercentile(data, probabilities, axis=axis)
    else:
        err_message = "The process 'quantiles' requires either the 'probabilities' or 'q' parameter to be set."
        sys.exit(err_message)


def mod(x, y):
    if not is_valid(x) or not is_valid(y):
        return np.nan
    else:
        return x % y


def absolute(x):
    return np.abs(x)


def power(base, p):
    if not np.all(is_valid(base)) or not is_valid(p):
        return np.nan
    else:
        return np.power(base, float(p))


def sgn(x):
    return np.sign(x)


def sqrt(x):
    return np.sqrt(x)


def exp(x):
    return np.exp(x)


def ln(x):
    return np.log(x)


def log(x, base):
    return np.log(x)/np.log(base)


def cos(x):
    return np.cos(x)


def arccos(x):
    return np.arccos(x)


def cosh(x):
    return np.cosh(x)


def arcosh(x):
    return np.arccosh(x)


def sin(x):
    return np.sin(x)


def arcsin(x):
    return np.arcsin(x)


def sinh(x):
    return np.sinh(x)


def arsinh(x):
    return np.arcsinh(x)


def tan(x):
    return np.tan(x)


def arctan(x):
    return np.arctan(x)


def tanh(x):
    return np.tanh(x)


def artanh(x):
    return np.arctanh(x)


def arctan2(y, x):
    return np.arctan2(y, x)


def cummax(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.maximum.accumulate(data, axis=axis)
    else:
        data = np.array(data)
        nan_idxs = np.isnan(data)
        data[nan_idxs] = np.nanmin(data)
        data_cummax = np.maximum.accumulate(data, axis=axis).astype(float)
        data_cummax[nan_idxs] = np.nan
        return data_cummax


def cummin(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.minimum.accumulate(data, axis=axis)
    else:
        data = np.array(data)
        nan_idxs = np.isnan(data)
        data[nan_idxs] = np.nanmax(data)
        data_cummin = np.minimum.accumulate(data, axis=axis).astype(float)
        data_cummin[nan_idxs] = np.nan
        return data_cummin


def cumproduct(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.cumprod(data, axis=axis)
    else:
        data = np.array(data)
        nan_idxs = np.isnan(data)
        data_cumprod = np.nancumprod(data, axis=axis).astype(float)
        data_cumprod[nan_idxs] = np.nan
        return data_cumprod


def cumsum(data, axis=0, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.cumsum(data, axis=axis)
    else:
        data = np.array(data)
        nan_idxs = np.isnan(data)
        data_cumsum = np.nancumsum(data, axis=axis).astype(float)
        data_cumsum[nan_idxs] = np.nan
        return data_cumsum


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


def linear_scale_range(x, input_min, input_max, output_min=0, output_max=1):
    if isinstance(x, list):
        x = list2nparray(x)

    return ((x - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min


def apply_factor(in_array, factor=1):
    """
    in_array is a tuple of 2d numpy arrays if the function is called from a pixel function in gdal
    """

    # Convert to 2d numpy array if input comes from gdal pixel function
    if (isinstance(in_array, (list, tuple))) and (len(in_array) == 1):
        in_array = in_array[0]

    out_array = in_array * factor

    return out_array


def eval_(x, expression=None):
    if expression is not None:
        return eval(expression)
    else:
        return None