import sys
import numpy as np
import pandas as pd
import functools
import operator
import xarray
import xarray_extras as xar_addons
import dask
from eofunctions.utils import is_empty, process
from eofunctions.errors import *


# TODO: add input argument validation decorator to each class
########################################################################################################################
# General Functions
########################################################################################################################

def eo_e():
    return np.e


def eo_pi():
    return np.pi


def eo_not(x):
    return not x

# TODO check the no data value functions if they do the job correctly
def eo_is_nan(x):

    if isinstance(x, (int, float, np.ndarray, xarray.DataArray, dask.array)):
        return pd.isnull(x)
    else:
        return True


def eo_is_nodata(x):

    if isinstance(x, (int, float, np.ndarray, xarray.DataArray, dask.array)):
        return pd.isnull(x)
    else:
        # if x in [np.nan, None]:
        #    return True
        # else:
        return False


def eo_is_valid(x):

    if isinstance(x, (np.ndarray, xarray.DataArray, dask.array)):
        return ~pd.isnull(x) & (x != np.inf)
    else:
        if x not in [np.nan, np.inf, None]:
            return True
        else:
            return False


def eo_floor(x):
    return np.floor(x)


def eo_ceil(x):
    return np.ceil(x)


def eo_exp(x):
    return np.exp(x)


def eo_ln(x):
    return np.log(x)


def eo_log(x, base):
    return np.log(x)/np.log(base)


def eo_mod(x, y):
    if not eo_is_valid(x) or not eo_is_valid(y):
        return np.nan
    else:
        return x % y


def eo_cos(x):
    return np.cos(x)


def eo_arccos(x):
    return np.arccos(x)


def eo_cosh(x):
    return np.cosh(x)


def eo_arcosh(x):
    return np.arccosh(x)


def eo_sin(x):
    return np.sin(x)


def eo_arcsin(x):
    return np.arcsin(x)


def eo_sinh(x):
    return np.sinh(x)


def eo_arsinh(x):
    return np.arcsinh(x)


def eo_tan(x):
    return np.tan(x)


def eo_arctan(x):
    return np.arctan(x)


def eo_tanh(x):
    return np.tanh(x)


def eo_artanh(x):
    return np.arctanh(x)


def eo_arctan2(y, x):
    return np.arctan2(y, x)


def linear_scale_range(x, input_min, input_max, output_min=0, output_max=1):
    return ((x - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min


def eo_eval(x, expression=None):
    return eval(expression)


def apply_factor(in_array, factor=1):
    return in_array * factor


# TODO: This function does not work fully yet as specified in
# https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#count
# link to a function understanding a process graph is needed
def eo_count(data, axis=0, expression=None):

    if expression == True:  # explicit check needed
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
            num_of_elems = np.sum(eo_is_valid(data), axis=axis)
        else:
            num_of_elems = np.sum(eo_is_valid(data))

    return num_of_elems


########################################################################################################################
# Int Process
########################################################################################################################

@process
def eo_int():
    return eoInt()


class eoInt(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num(data):
        return int(data)

    @staticmethod
    def exec_np(data):
        return data.astype(int)

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Round Process
########################################################################################################################

@process
def eo_round():
    return eoRound()


class eoRound(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num(data, p=0):
        return round(data, p)

    @staticmethod
    def exec_np(data, p=0):
        return np.around(data, p)

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Absolute Process
########################################################################################################################

@process
def eo_absolute():
    return eoAbsolute()


class eoAbsolute(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num(data):
        return abs(data)

    @staticmethod
    def exec_np(data):
        return np.abs(data)

    @staticmethod
    def exec_xar(data):
        return data.abs()

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Signum Process
########################################################################################################################

@process
def eo_sign():
    return eoSign()


class eoSign(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num(data):
        return np.sign(data)

    @staticmethod
    def exec_np(data):
        return np.sign(data)

    @staticmethod
    def exec_xar(data):
        return data.sign()

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Sqrt Process
########################################################################################################################

@process
def eo_sqrt():
    return eoSqrt()


class eoSqrt(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num(data):
        return np.sqrt(data)

    @staticmethod
    def exec_np(data):
        return np.sqrt(data)

    @staticmethod
    def exec_xar(data):
        return data.sqrt()

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Power Process
########################################################################################################################

@process
def eo_power():
    return eoPower()


class eoPower(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_num(base, p):
        if not eo_is_valid(base) or not eo_is_valid(p):
            return np.nan
        else:
            return np.power(base, float(p))

    @staticmethod
    def exec_np(base, p):
        if not np.all(eo_is_valid(base)) or not eo_is_valid(p):
            return np.nan
        else:
            return np.power(base, float(p))

    @staticmethod
    def exec_xar(base, p):
        if not np.all(eo_is_valid(base)) or not eo_is_valid(p):
            return np.nan
        else:
            return np.power(base, float(p))

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Mean Process
########################################################################################################################

@process
def eo_mean():
    return eoMean()


class eoMean(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.mean(data, axis=dimension)
        else:
            return np.nanmean(data, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.mean(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Min Process
########################################################################################################################

@process
def eo_min():
    return eoMin()


class eoMin(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.min(data, axis=dimension)
        else:
            return np.nanmin(data, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.min(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Max Process
########################################################################################################################

@process
def eo_max():
    return eoMax()


class eoMax(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.max(data, axis=dimension)
        else:
            return np.nanmax(data, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.max(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Median Process
########################################################################################################################

@process
def eo_median():
    return eoMedian()


class eoMedian(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.median(data, axis=dimension)
        else:
            return np.nanmedian(data, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.median(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Standard Deviation Process
########################################################################################################################

@process
def eo_sd():
    return eoSd()


class eoSd(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.std(data, axis=dimension, ddof=1)
        else:
            return np.nanstd(data, axis=dimension, ddof=1)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.std(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Variance Process
########################################################################################################################

@process
def eo_variance():
    return eoVariance()


class eoVariance(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.var(data, axis=dimension, ddof=1)
        else:
            return np.nanvar(data, axis=dimension, ddof=1)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.var(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


# TODO: quantiles with nans are not working properly/really slow -> own implementation (e.g. like in SGRT)?
########################################################################################################################
# Quantiles Process
########################################################################################################################

@process
def eo_quantiles():
    return eoQuantiles()


class eoQuantiles(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0, probabilities=None, q=None):
        if (probabilities is not None) and (q is not None):
            raise QuantilesParameterConflict()

        if probabilities is not None:
            if is_empty(data):
                return [np.nan] * len(probabilities)
            probabilities = list(np.array(probabilities) * 100.)
        elif q is not None:
            probabilities = list(np.arange(0, 100, 100. / q))[1:]
            if is_empty(data):
                return [np.nan] * len(probabilities)
        else:
            raise QuantilesParameterMissing()

        if not ignore_nodata:
            return np.percentile(data, probabilities, axis=dimension)
        else:
            return np.nanpercentile(data, probabilities, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0, probabilities=None, q=None):
        if (probabilities is not None) and (q is not None):
            raise QuantilesParameterConflict()

        if probabilities is not None:
            if is_empty(data):
                return [np.nan] * len(probabilities)
        elif q is not None:
            probabilities = list(np.arange(0, 1, 1./q))[1:]
            if is_empty(data):
                return [np.nan] * len(probabilities)
        else:
            raise QuantilesParameterMissing()

        return data.quantile(np.array(probabilities), dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Cummax Process
########################################################################################################################

@process
def eo_cummax():
    return eoCummax()


class eoCummax(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.maximum.accumulate(data, axis=dimension)
        else:
            data = np.array(data)
            nan_idxs = np.isnan(data)
            data[nan_idxs] = np.nanmin(data)
            data_cummax = np.maximum.accumulate(data, axis=dimension).astype(float)
            data_cummax[nan_idxs] = np.nan

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.maximum.accumulate(data, axis=dimension)
        else:
            data = np.array(data)
            nan_idxs = np.isnan(data)
            data[nan_idxs] = np.nanmin(data)
            data_cummax = np.maximum.accumulate(data, axis=dimension).astype(float)
            data_cummax[nan_idxs] = np.nan

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Cummax Process
########################################################################################################################

@process
def eo_cummin():
    return eoCummin()


class eoCummin(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.minimum.accumulate(data, axis=dimension)
        else:
            data = np.array(data)
            nan_idxs = np.isnan(data)
            data[nan_idxs] = np.nanmax(data)
            data_cummin = np.minimum.accumulate(data, axis=dimension).astype(float)
            data_cummin[nan_idxs] = np.nan
            return data_cummin

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.minimum.accumulate(data, axis=dimension)
        else:
            data = np.array(data)
            nan_idxs = np.isnan(data)
            data[nan_idxs] = np.nanmax(data)
            data_cummin = np.minimum.accumulate(data, axis=dimension).astype(float)
            data_cummin[nan_idxs] = np.nan
            return data_cummin

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Cumproduct Process
########################################################################################################################

@process
def eo_cumproduct():
    return eoCumproduct()


class eoCumproduct(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.cumprod(data, axis=dimension)
        else:
            data = np.array(data)
            nan_idxs = np.isnan(data)
            data_cumprod = np.nancumprod(data, axis=dimension).astype(float)
            data_cumprod[nan_idxs] = np.nan
            return data_cumprod

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        return xar_addons.cumulatives.compound_prod(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Cumsum Process
########################################################################################################################

@process
def eo_cumsum():
    return eoCumsum()


class eoCumsum(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.cumsum(data, axis=dimension)
        else:
            data = np.array(data)
            nan_idxs = np.isnan(data)
            data_cumsum = np.nancumsum(data, axis=dimension).astype(float)
            data_cumsum[nan_idxs] = np.nan
            return data_cumsum

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if is_empty(data):
            return np.nan

        return xar_addons.cumulatives.compound_sum(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Sum Process
########################################################################################################################

@process
def eo_sum():
    return eoSum()


class eoSum(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        number_elems = eo_count(data, axis=dimension, expression=True)
        if number_elems < 2:
            raise SummandMissing
        if not ignore_nodata:
            return np.sum(data, axis=dimension)
        else:
            return np.nansum(data, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.sum(data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Subtract Process
########################################################################################################################

@process
def eo_subtract():
    return eoSubtract()


class eoSubtract(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        number_elems = eo_count(data, axis=dimension, expression=True)
        if number_elems < 2:
            raise SubtrahendMissing
        if not ignore_nodata:
            return np.sum(-data, axis=dimension)
        else:
            return np.nansum(-data, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):

        return data.sum(-data, dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Multiply Process
########################################################################################################################

@process
def eo_multiply():
    return eoMultiply()

# TODO: better implementation?
class eoMultiply(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        number_elems = eo_count(data, axis=dimension, expression=True)
        if number_elems < 2:
            raise MultiplicandMissing
        nan_idxs = pd.isnull(data)
        if not ignore_nodata:
            if np.any(nan_idxs):
                return np.nan
            else:
                data = data[~nan_idxs]
                return np.apply_along_axis(lambda data: functools.reduce(operator.mul, data, 1), dimension, data)
        else:
            data = data[~nan_idxs]
            return np.apply_along_axis(lambda data: functools.reduce(operator.mul, data, 1), dimension, data)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        pass

    @staticmethod
    def exec_da(self):
        pass


def eo_product(data, axis=0, ignore_nodata=True):
    return eo_multiply(data, axis=axis, ignore_nodata=ignore_nodata)


########################################################################################################################
# Divide Process
########################################################################################################################

@process
def eo_divide():
    return eoDivide()


# TODO: better implementation?
class eoDivide(object):
    def __init__(self):
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        number_elems = eo_count(data, axis=dimension, expression=True)
        if number_elems < 2:
            raise DivisorMissing
        nan_idxs = pd.isnull(data)
        if not ignore_nodata:
            if np.any(nan_idxs):
                return np.nan
            else:
                data = data[~nan_idxs]
                return np.apply_along_axis(lambda data: functools.reduce(operator.truediv, data), dimension, data)
        else:
            data = data[~nan_idxs]
            return np.apply_along_axis(lambda data: functools.reduce(operator.truediv, data), dimension, data)

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


if __name__ == '__main__':
    a = 0