import operator
import functools

import numpy as np
import pandas as pd
import xarray_extras as xar_addons


from eofunctions.eo_utils import process
from eofunctions.eo_utils import build_multi_dim_index
from eofunctions.arrays import eo_count
from eofunctions.checks import eo_is_empty
from eofunctions.checks import eo_is_valid

from eofunctions.errors import QuantilesParameterConflict
from eofunctions.errors import QuantilesParameterMissing
from eofunctions.errors import SummandMissing
from eofunctions.errors import SubtrahendMissing
from eofunctions.errors import MultiplicandMissing
from eofunctions.errors import DivisorMissing


# TODO: what should we do with numbers in case of reducers?
########################################################################################################################
# General Functions (Data type/array independent functions)
########################################################################################################################
def eo_e():
    return np.e


def eo_pi():
    return np.pi


def eo_not(x):
    return not x


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


def eo_linear_scale_range(x, input_min, input_max, output_min=0, output_max=1):
    return ((x - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min


def eo_eval(x, expression=None):
    return eval(expression)


def eo_apply_factor(in_array, factor=1):
    return in_array * factor


########################################################################################################################
# Mod Process
########################################################################################################################

@process
def eo_mod():
    return EOMod()


class EOMod(object):

    @staticmethod
    def exec_num(x, y):
        """
        Remainder after division of x by y. It returns np.nan if one of both values is not valid.

        Parameters
        ----------
        x: int, float
            A number to be used as dividend.
        y: int, float
            A number to be used as divisor.

        Returns
        -------
        float

        """
        if not eo_is_valid(x) or not eo_is_valid(y):
            return np.nan
        else:
            return x % y

    @staticmethod
    def exec_np(x, y):
        """
        Remainder after division of 'x' by 'y'. It returns np.nan if one of both arrays is not valid.

        Parameters
        ----------
        x: np.array
            An array to be used as dividend.
        y: np.array
            An array to be used as divisor.

        Returns
        -------
        np.array

        """
        if not eo_is_valid(x) or not eo_is_valid(y):
            return np.nan
        else:
            return np.mod(x, y)

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Int Process
########################################################################################################################

@process
def eo_int():
    return EOInt()


class EOInt(object):

    @staticmethod
    def exec_num(x):
        """
        The integer part of the real number 'x'.

        Parameters
        ----------
        x: int, float
            A number.

        Returns
        -------
        int
            Integer part of the number.
        """
        return int(x)

    @staticmethod
    def exec_np(x):
        """
        The integer part of the array 'x'.

        Parameters
        ----------
        x: np.array
            An array.

        Returns
        -------
        np.array
            Integer part of the numbers in the array.
        """
        return x.astype(int)

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
    return EORound()


class EORound(object):

    @staticmethod
    def exec_num(x, p=0):
        """
        Rounds a real number 'x' to specified precision 'p'.

        Parameters
        ----------
        x: int, float
            A number.
        p: int, optional
            A positive number specifies the number of digits after the decimal point to round to.
            A negative number means rounding to a power of ten, so for example -2 rounds to the nearest hundred.
            Defaults to 0.
        Returns
        -------
        int, float
            Rounded number.

        Notes
        -----
        If the fractional part of x is halfway between two integers, one of which is even and the other odd, then the
        even number is returned. This behaviour follows IEEE Standard 754. This kind of rounding is also called
        "rounding to nearest" or "banker's rounding". It minimizes rounding errors that result from consistently
        rounding a midpoint value in a single direction.
        """
        return round(x, p)

    @staticmethod
    def exec_np(x, p=0):
        """
        Rounds an array 'x' to specified precision 'p'.

        Parameters
        ----------
        x: np.array
            An array.
        p: int, optional
            A positive number specifies the number of digits after the decimal point to round to.
            A negative number means rounding to a power of ten, so for example -2 rounds to the nearest hundred.
            Defaults to 0.
        Returns
        -------
        np.array
            Rounded array.

        Notes
        -----
        If the fractional part of x is halfway between two integers, one of which is even and the other odd, then the
        even number is returned. This behaviour follows IEEE Standard 754. This kind of rounding is also called
        "rounding to nearest" or "banker's rounding". It minimizes rounding errors that result from consistently
        rounding a midpoint value in a single direction.
        """
        return np.around(x, p)

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
    return EOAbsolute()


class EOAbsolute(object):

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
def eo_sgn():
    return EOSgn()


class EOSgn(object):

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
    return EOSqrt()


class EOSqrt(object):

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
    return EOPower()


class EOPower(object):

    @staticmethod
    def exec_num(base, p):
        """
        Computes the exponentiation for the base 'base' raised to the power of 'p'.
        np.nan is returned if any of the arguments is not valid.

        Parameters
        ----------
        base: int, float
            The numerical base.
        p: int, float
            The numerical exponent.
        Returns
        -------
        int, float
            The computed value for 'base' raised to the power of 'p'.
        """
        if not eo_is_valid(base) or not eo_is_valid(p):
            return np.nan
        else:
            return np.power(base, float(p))

    @staticmethod
    def exec_np(base, p):
        """
        Computes the exponentiation for the base 'base' raised to the power of 'p'.
        np.nan is returned if any of the arguments is not valid.

        Parameters
        ----------
        base: np.array
            A base array.
        p: int, float
            The numerical exponent.
        Returns
        -------
        np.array
            The computed array for 'base' raised to the power of 'p'.
        """
        if not np.all(eo_is_valid(base)) or not eo_is_valid(p):
            return np.nan
        else:
            return np.power(base, float(p))

    @staticmethod
    def exec_xar(self):
        pass

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Mean Process
########################################################################################################################

@process
def eo_mean():
    return EOMean()


class EOMean(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOMin()


class EOMin(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOMax()


class EOMax(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOMedian()


class EOMedian(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOSd()


class EOSd(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOVariance()


class EOVariance(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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


########################################################################################################################
# Extrema Process
########################################################################################################################

@process
def eo_extrema():
    return EOExtrema()


class EOExtrema(object):

    @staticmethod
    def exec_num(data, dimension=0, ignore_nodata=True):
        return [data, data]

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        """
        Returns the extrema of the input data, i.e. the minimum and the maximum value.

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
        list of numpy arrays
            Minimum and maximum value.
        """
        min_val = eo_min(data, dimension=dimension, ignore_nodata=ignore_nodata)
        max_val = eo_max(data, dimension=dimension, ignore_nodata=ignore_nodata)

        return [min_val, max_val]

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


# TODO: quantiles with nans are not working properly/really slow -> own implementation (e.g. like in SGRT)?
########################################################################################################################
# Quantiles Process
########################################################################################################################

@process
def eo_quantiles():
    return EOQuantiles()


class EOQuantiles(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0, probabilities=None, q=None):
        EOQuantiles()._check_input(probabilities=probabilities, q=q)

        if probabilities is not None:
            probabilities = list(np.array(probabilities) * 100.)
        elif q is not None:
            probabilities = list(np.arange(0, 100, 100. / q))[1:]

        return [data]*len(probabilities)

    @staticmethod
    def exec_np(data, dimension=0, probabilities=None, q=None, ignore_nodata=True):
        """
        Calculates quantiles, which are cut points dividing the range of a probability distribution into either
        intervals corresponding to the given probabilities 'probabilities' or (nearly) equal-sized intervals
        (q-quantiles based on the parameter 'q'). Either the parameter 'probabilities' or 'q' must be specified,
        otherwise the 'QuantilesParameterMissing' exception is thrown. If both parameters are set the
        'QuantilesParameterConflict' exception is thrown.


        Parameters
        ----------
        data: np.array
            An array of numbers.
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        probabilities: list, optional
            A list of probabilities to calculate quantiles for. The probabilities must be between 0 and 1.
        q: int, optional
            A number of intervals to calculate quantiles for. Calculates q-quantiles with (nearly) equal-sized intervals.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).

        Returns
        -------
        list of numpy arrays
            An array with the computed quantiles. The list has either as many elements as the given list of
            probabilities had or 'q'-1 elements. If the input array is empty the resulting array is filled with as
            many np.nan values as required according to the list above.

        Raises
        ------
        QuantilesParameterMissing
            If both parameters 'probabilities' and 'q' are None.
        QuantilesParameterConflict
            If both parameters 'probabilities' and 'q' are set.
        """
        EOQuantiles()._check_input(probabilities=probabilities, q=q)

        if probabilities is not None:
            if eo_is_empty(data):
                return [np.nan] * len(probabilities)
            probabilities = list(np.array(probabilities) * 100.)
        elif q is not None:
            probabilities = list(np.arange(0, 100, 100. / q))[1:]
            if eo_is_empty(data):
                return [np.nan] * len(probabilities)

        if not ignore_nodata:
            return np.percentile(data, probabilities, axis=dimension)
        else:
            return np.nanpercentile(data, probabilities, axis=dimension)

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0, probabilities=None, q=None):
        EOQuantiles()._check_input(probabilities=probabilities, q=q)

        if probabilities is not None:
            if eo_is_empty(data):
                return [np.nan] * len(probabilities)
        elif q is not None:
            probabilities = list(np.arange(0, 1, 1./q))[1:]
            if eo_is_empty(data):
                return [np.nan] * len(probabilities)

        return data.quantile(np.array(probabilities), dim=dimension, skipna=~ignore_nodata)

    @staticmethod
    def exec_da(self):
        pass

    @staticmethod
    def _check_input(probabilities=None, q=None):
        if (probabilities is not None) and (q is not None):
            raise QuantilesParameterConflict()

        if probabilities is None and q is None:
            raise QuantilesParameterMissing()


########################################################################################################################
# Cummax Process
########################################################################################################################

@process
def eo_cummax():
    return EOCummax()


class EOCummax(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Finds cumulative maxima of an array of numbers. Every computed element is equal to the bigger one between
        current element and the previously computed element. The returned array and the input array have always the
        same length. By default, no-data values are skipped, but stay in the result. Setting the 'ignore_nodata' flag
        to True makes that once a np.nan value is reached all following elements are set to np.nan in the result.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).

        Returns
        -------
        np.array
            An array with the computed cumulative maxima.
        """
        if eo_is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.maximum.accumulate(data, axis=dimension)
        else:
            nan_idxs = np.isnan(data)
            data[nan_idxs] = np.nanmin(data)
            data_cummax = np.maximum.accumulate(data, axis=dimension).astype(float)
            data_cummax[nan_idxs] = np.nan  # fill in the old np.nan values again
            return data_cummax

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.maximum.accumulate(data, axis=dimension)
        else:
            data = np.array(data)
            nan_idxs = np.isnan(data)
            data[nan_idxs] = np.nanmin(data)
            data_cummax = np.maximum.accumulate(data, axis=dimension).astype(float)
            data_cummax[nan_idxs] = np.nan
            return data_cummax

    @staticmethod
    def exec_da(self):
        pass


########################################################################################################################
# Cummax Process
########################################################################################################################

@process
def eo_cummin():
    return EOCummin()


class EOCummin(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Finds cumulative minima of an array of numbers. Every computed element is equal to the smaller one between
        current element and the previously computed element. The returned array and the input array have always the
        same length. By default, no-data values are skipped, but stay in the result. Setting the 'ignore_nodata' flag
        to True makes that once a np.nan value is reached all following elements are set to np.nan in the result.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).

        Returns
        -------
        np.array
            An array with the computed cumulative minima.
        """
        if eo_is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.minimum.accumulate(data, axis=dimension)
        else:
            nan_idxs = np.isnan(data)
            data[nan_idxs] = np.nanmax(data)
            data_cummin = np.minimum.accumulate(data, axis=dimension).astype(float)
            data_cummin[nan_idxs] = np.nan  # fill in the old np.nan values again
            return data_cummin

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOCumproduct()


class EOCumproduct(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Computes cumulative products of an array of numbers. Every computed element is equal to the product of
        current element and the previously computed element. The returned array and the input array have always the
        same length. By default, no-data values are skipped, but stay in the result. Setting the 'ignore_nodata' flag
        to True makes that once a np.nan value is reached all following elements are set to np.nan in the result.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).

        Returns
        -------
        np.array
            An array with the computed cumulative products.
        """
        if eo_is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.cumprod(data, axis=dimension)
        else:
            nan_idxs = np.isnan(data)
            data_cumprod = np.nancumprod(data, axis=dimension).astype(float)
            data_cumprod[nan_idxs] = np.nan  # fill in the old np.nan values again
            return data_cumprod

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOCumsum()


class EOCumsum(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Computes cumulative sums of an array of numbers. Every computed element is equal to the sum of
        current element and the previously computed element. The returned array and the input array have always the
        same length. By default, no-data values are skipped, but stay in the result. Setting the 'ignore_nodata' flag
        to True makes that once a np.nan value is reached all following elements are set to np.nan in the result.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).

        Returns
        -------
        np.array
            An array with the computed cumulative sums.
        """
        if eo_is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.cumsum(data, axis=dimension)
        else:
            nan_idxs = np.isnan(data)
            data_cumsum = np.nancumsum(data, axis=dimension).astype(float)
            data_cumsum[nan_idxs] = np.nan  # fill in the old np.nan values again
            return data_cumsum

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=0):
        if eo_is_empty(data):
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
    return EOSum()


class EOSum(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0, extra_values=None):
        """
        Sums up all elements in a sequential array of numbers and returns the computed sum.
        By default no-data values are ignored. Setting 'ignore_nodata' to False considers no-data values so that np.nan
        is returned if any element is such a value.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        extra_values: list, optional
            Offers to add additional elements to the computed sum.

        Returns
        -------
        np.array
            The computed sum of the sequence of numbers.

        Raises
        ------
        SummandMissing
            Is thrown when less than two values are given.
        """
        extra_values = extra_values if extra_values is not None else []
        n_extra = len(extra_values)
        n = eo_count(data, dimension=dimension, expression=True) + n_extra

        if n < 2:
            raise SummandMissing

        if not ignore_nodata:
            summand = np.sum(extra_values)
            return np.sum(data, axis=dimension) + summand
        else:
            summand = np.nansum(extra_values)
            return np.nansum(data, axis=dimension) + summand

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
    return EOSubtract()


#TODO: replace no data values which appear first
class EOSubtract(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0, extra_values=None, extra_idxs=None):
        """
        Takes the first element of a sequential array of numbers and subtracts all other elements from it.
        By default no-data values are ignored. Setting 'ignore_nodata' to False considers no-data values so that np.nan
        is returned if any element is such a value.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        extra_values: list, optional
            Offers to subtract additional elements to the computed subtraction.

        Returns
        -------
        np.array
            The computed subtraction of the sequence of numbers.

        Raises
        ------
        SubtrahendMissing
            Is thrown when less than two values are given.
        """
        n_extra = len(extra_values) if extra_values is not None else 0
        n = eo_count(data, dimension=dimension, expression=True) + n_extra
        if n < 2:
            raise SubtrahendMissing

        binary_fun = lambda a, b: np.subtract(a, b)
        nodata = None
        if ignore_nodata:
            nodata = 0.

        data = binary_iterator(data, binary_fun, extra_values=extra_values, extra_idxs=extra_idxs,
                               dimension=dimension, nodata=nodata)
        
        return data

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
    return EOMultiply()


# TODO: better implementation?
class EOMultiply(object):

    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0, extra_values=None):
        """
        Multiplies all elements in a sequential array of numbers and returns the computed product.
        By default no-data values are ignored. Setting 'ignore_nodata' to False considers no-data values so that np.nan
        is returned if any element is such a value.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        extra_values: list, optional
            Offers to subtract additional elements to the computed subtraction.

        Returns
        -------
        np.array
            The computed product of the sequence of numbers.

        Raises
        ------
        MultiplicandMissing
            Is thrown when less than two values are given.
        """
        extra_values = extra_values if extra_values is not None else []
        n_extra = len(extra_values)
        n = eo_count(data, dimension=dimension, expression=True) + n_extra
        if n < 2:
            raise MultiplicandMissing

        if ignore_nodata:
            data[np.isnan(data)] = 1.
            
        if len(extra_values) > 0:
            extra_values_tot = np.prod(extra_values, axis=0)
        else:
            extra_values_tot = 1.
        
        data = np.prod(data, axis=dimension, initial=extra_values_tot)
            
        return data

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
    return EODivide()


#TODO: replace no data values which appear first
class EODivide(object):
    @staticmethod
    def exec_num(data, ignore_nodata=True, dimension=0):
        return data

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0, extra_values=None, extra_idxs=None):
        """
        Divides the first element in a sequential array of numbers by all other elements.
        By default no-data values are ignored. Setting 'ignore_nodata' to False considers no-data values so that np.nan
        is returned if any element is such a value.

        Parameters
        ----------
        data: np.array
            An array of numbers.
        ignore_nodata: bool, optional
            Specifies if np.nan values are ignored or not (True is default).
        dimension: int, optional
            Dimension/axis of interest (0 is default).
        extra_values: list, optional
            Offers to subtract additional elements to the computed subtraction.

        Returns
        -------
        np.array
            The computed ratio of the sequence of numbers.

        Raises
        ------
        DivisorMissing
            Is thrown when less than two values are given.
        """
        n_extra = len(extra_values) if extra_values is not None else 0
        n = eo_count(data, dimension=dimension, expression=True) + n_extra
        if n < 2:
            raise DivisorMissing

        binary_fun = lambda a, b: np.divide(a, b)
        nodata = None
        if ignore_nodata:
            nodata = 1.

        data = binary_iterator(data, binary_fun, extra_values=extra_values, extra_idxs=extra_idxs,
                               dimension=dimension, nodata=nodata)
            
        return data

    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


def binary_iterator(arr, binary_fun, extra_values=None, extra_idxs=None, dimension=0, nodata=None):
    n_arr = arr.shape[dimension]
    if extra_idxs is None and extra_values is not None:  # create default indizes according to the given length of 'extra_values'
        extra_idxs = list(range(n_arr, n_arr + len(extra_values)))
        n_extra = len(extra_idxs)
    elif extra_idxs is not None and extra_values is not None:
        n_extra = len(extra_idxs)
    elif extra_idxs is None and extra_values is None:
        n_extra = 0
    else:
        raise Exception("Only 'extra_idxs' is given. Please specify 'extra_values' in addition.")

    n = n_arr + n_extra
    data, arr_idx = index_arr_and_values(0, arr, 0, extra_values=extra_values, extra_idxs=extra_idxs,
                                         dimension=dimension)
    if nodata is not None:
        data = replace_nodata_arr_or_value(data, nodata)
    for i in range(1, n):
        curr_data, arr_idx = index_arr_and_values(i, arr, arr_idx, extra_values=extra_values, extra_idxs=extra_idxs,
                                                  dimension=dimension)
        if nodata is not None:
            curr_data = replace_nodata_arr_or_value(curr_data, nodata)

        data = binary_fun(data, curr_data)

    return data


def index_arr_and_values(idx, arr, arr_idx, extra_values=None, extra_idxs=None, dimension=0):
    if extra_idxs is not None and extra_values is not None:
        if idx in extra_idxs:
            return extra_values[extra_idxs.index(idx)], arr_idx
        else:
            arr_select = build_multi_dim_index(arr_idx, arr.shape, dimension)  # create index string
            arr_idx += 1
            return eval("arr[{}]".format(arr_select)), arr_idx
    else:
        arr_select = build_multi_dim_index(arr_idx, arr.shape, dimension)  # create index string
        arr_idx += 1
        return eval("arr[{}]".format(arr_select)), arr_idx


def replace_nodata_arr_or_value(data, nodata):
    if isinstance(data, np.ndarray):
        data[np.isnan(data)] = nodata
    else:
        if np.isnan(data):
            data = nodata

    return data


if __name__ == '__main__':
    A = np.ones((10, 10000, 10000))
    B = eo_subtract(A, extra_values=[2, 3])
