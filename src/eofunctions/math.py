import sys
import numpy as np


def e():
    return np.e


def pi():
    return np.pi


def is_nan(x):
    if isinstance(x, (int, float, np.ndarray)):
        return np.isnan(x)
    else:
        return True


def is_nodata(x):
    if x in [np.nan, None]:
        return True
    else:
        return False


def is_valid(x):
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
    return np.int(x)


def floor(x):
    return np.floor(x)


def ceil(x):
    return np.ceil(x)


def round_(x, p=0):
    return round(x, p)


def min_(data, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.min(data)
    else:
        return np.nanmin(data)


def max_(data, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.max(data)
    else:
        return np.nanmax(data)


def mean(data, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.mean(data)
    else:
        return np.nanmean(data)


def median(data, ignore_nodata=True):
    return quantiles(data, probabilities=[0.5], ignore_nodata=ignore_nodata)[0]


def sd(data, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.std(data, ddof=1)
    else:
        return np.nanstd(data, ddof=1)


def variance(data, ignore_nodata=True):
    if is_empty(data):
        return np.nan

    if not ignore_nodata:
        return np.var(data, ddof=1)
    else:
        return np.nanvar(data, ddof=1)


def quantiles(data, probabilities=None, q=None, ignore_nodata=True):
    if (probabilities is not None) and (q is not None):
        err_message = "The process 'quantiles' only allows that either the 'probabilities' or the 'q' parameter is set."
        sys.exit(err_message)

    if probabilities is not None:
        if is_empty(data):
            return [np.nan] * len(probabilities)
        if not ignore_nodata:
            return np.percentile(data, (np.array(probabilities)*100.).tolist()).tolist()
        else:
            return np.nanpercentile(data, (np.array(probabilities)*100.).tolist()).tolist()
    elif q is not None:
        probabilities = list(np.arange(0, 100, 100./q))[1:]
        if is_empty(data):
            return [np.nan] * len(probabilities)
        if not ignore_nodata:
            return np.percentile(data, probabilities).tolist()
        else:
            return np.nanpercentile(data, probabilities).tolist()
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
    if not is_valid(base) or not is_valid(p):
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

def cummax(data, ignore_nodata=True):
    if is_empty(data):
        return None

    if not ignore_nodata:
        return np.maximum.accumulate(data)
    else:
        nan_idxs = np.isnan(data)
        data[nan_idxs] = np.min(data)
        data_cummax = np.maximum.accumulate(data)
        data_cummax[nan_idxs] = np.nan
        return data_cummax

def cummin(data, ignore_nodata=True):
    if is_empty(data):
        return None

    if not ignore_nodata:
        return np.minimum.accumulate(data)
    else:
        nan_idxs = np.isnan(data)
        data[nan_idxs] = np.max(data)
        data_cummin = np.minimum.accumulate(data)
        data_cummin[nan_idxs] = np.nan
        return data_cummin


def cumproduct(data, ignore_nodata=True):
    if is_empty(data):
        return None

    if not ignore_nodata:
        return np.cumprod(data)
    else:
        return np.nancumprod(data)


def cumsum(data, ignore_nodata=True):
    if is_empty(data):
        return None

    if not ignore_nodata:
        return np.cumsum(data)
    else:
        return np.nancumsum(data)


def eq(x, y, delta=None, case_sensitive=True):
    if not is_valid(x) or not is_valid(y):
        return None
    if (type(x) in [float, int]) and (type(y) in [float, int]):
        if type(delta) in [float, int]:
            return np.isclose(x, y, atol=delta)
        else:
            return x == y
    elif (type(x) == str) and (type(y) == str):
        if case_sensitive:
            return x == y
        else:
            return x.lower() == y.lower()

def neq(x, y, delta=None, case_sensitive=True):
    return ~eq(x, y, delta=delta, case_sensitive=case_sensitive)

def gt(x, y):
    if not is_valid(x) or not is_valid(y):
        return None
    return x > y

def gte(x, y):
    if not is_valid(x) or not is_valid(y):
        return None
    return x >= y

def lt(x, y):
    if not is_valid(x) or not is_valid(y):
        return None
    return x < y

def lte(x, y):
    if not is_valid(x) or not is_valid(y):
        return None
    return x <= y


def between(x, min, max):
    if lt(max, min):
        return False

    return gte(x, min) & lte(x, max)


def linear_scale_range(x, input_min, input_max, output_min, output_max):
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