import xarray
import dask
import datetime
import re
import numpy as np
from datetime import timezone
from datetime import timedelta
from datetime import datetime


def eval_datatype(data):
    """
    Returns a data type tag depending on the data type of `data`.
    This can be:
        - "np": `np.ndarray` or list (list are directly converted to numpy arrays)
        - "xar": `xarray.DataArray`
        - "dar": `dask.array.core.Array`
        - "num": `np.integer`, `np.float`, int, float, or str
        - "dt": `datetime.datetime`
        - "dict": dict
        - "fun": callable object
        - "none": None

    Parameters
    ----------
    data : object
        Data to get the data type from.

    Returns
    -------
    str :
        Data type tag.

    """
    is_list = isinstance(data, list)
    is_np = isinstance(data, np.ndarray) | is_list
    is_xar = isinstance(data, xarray.DataArray)
    is_dar = isinstance(data, dask.array.core.Array)
    is_num = isinstance(data, (int, float, str, np.integer, np.float))
    is_datetime = isinstance(data, datetime)
    is_dict = isinstance(data, dict)
    is_function = callable(data)
    is_none = data is None

    if is_np:
        datatype = "np"
    elif is_xar:
        datatype = "xar"
    elif is_dar:
        datatype = "dar"
    elif is_num:
        datatype = "num"
    elif is_none:
        datatype = "none"
    elif is_datetime:
        datatype = "dt"
    elif is_dict:
        datatype = "dict"
    elif is_function:
        datatype = "fun"
    else:
        err_msg = "Data type '{}' is unknown.".format(type(data))
        raise ValueError(err_msg)

    return datatype


def process(processor):
    """
    This function serves as a decorator for empty openEO process definitions, which call a class `processor` defining
    the process implementations for different data types.

    Parameters
    ----------
    processor : class
        Class implementing an openEO process containing the methods `exec_num`, `exec_np`, `exec_xar`, or `exec_dar`.

    Returns
    -------
    object :
        Process/function wrapper returning the result of the process.

    """
    def fun_wrapper(*args, **kwargs):
        cls = processor()

        datatypes = []
        # retrieve data types of input arguments and convert lists to numpy arrays
        args = list(args)
        for i, arg in enumerate(args):
            datatypes.append(eval_datatype(arg))
            if isinstance(arg, list):
                args[i] = list2nparray(arg)
        args = tuple(args)

        # retrieve data types of input keyword arguments and convert lists to numpy arrays
        for key, kwarg in kwargs.items():
            datatypes.append(eval_datatype(kwarg))
            if isinstance(kwarg, list):
                kwargs[key] = list2nparray(kwarg)

        datatypes = np.array(datatypes)
        if (datatypes == "np").any():
            cls_fun = getattr(cls, "exec_np")
        elif (datatypes == "xar").any():
            cls_fun = getattr(cls, "exec_xar")
        elif (datatypes == "dar").any():
            cls_fun = getattr(cls, "exec_dar")
        elif ((datatypes == "num") | (datatypes == "none") | (datatypes == "dt")).all():
            cls_fun = getattr(cls, "exec_num")
        else:
            raise Exception('Datatype unknown.')

        return cls_fun(*args, **kwargs)

    return fun_wrapper


def list2nparray(x):
    """
    Converts a list in a nump

    Parameters
    ----------
    x : list or np.ndarray
        List to convert.

    Returns
    -------
    np.ndarray

    """
    x_tmp = np.array(x)
    if x_tmp.dtype.kind in ['U', 'S']:
        x = np.array(x, dtype=object)
    else:
        x = x_tmp

    return x


def create_slices(index, axis=0, n_axes=1):
    """
    Creates a multidimensional slice index.

    Parameters
    ----------
    index : int
        The zero-based index of the element to retrieve (default is 0).
    axis : int, optional
        Axis of the given index (default is 0).
    n_axes : int, optional
        Number of axes (default is 1).

    Returns
    -------
    tuple of slice:
        Tuple of index slices.

    """

    slices = [slice(None)] * n_axes
    slices[axis] = index

    return tuple(slices)


def str2time(string, allow_24h=False):
    """
    Converts time strings in various formats to a datetime object.
    The datetime formats follow the RFC3339 convention.

    Parameters
    ----------
    string : str
        String representation of time or date.
    allow_24h : bool, optional
        If True, `string` is allowed to contain '24' as hour value.

    Returns
    -------
    datetime.datetime :
        Parsed datetime object.

    """

    # handle timezone formatting and replace possibly occuring ":" in time zone string
    # handle timezone formatting for +
    if "+" in string:
        string_parts = string.split('+')
        string_parts[-1] = string_parts[-1].replace(':', '')
        string = "+".join(string_parts)

    # handle timezone formatting for -
    if "t" in string.lower():  # a full datetime string is given
        time_string = string[10:]
        if "-" in time_string:
            string_parts = time_string.split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = string[:10] + "-".join(string_parts)
    else:  # a time string is given
        if "-" in string:
            string_parts = string.split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = "-".join(string_parts)

    # searches for 24 in hour value
    pattern = re.compile("24:\d{2}:\d{2}")
    pattern_match = re.search(pattern, string)
    if pattern_match:
        if allow_24h:  # if the user allows 24 as an hour value, replace 24 by 23 and add a timedelta of one hour later
            old_sub_string = pattern_match.group()
            new_sub_string = "23" + old_sub_string[2:]
            string = string.replace(old_sub_string, new_sub_string)
        else:
            err_msg = "24 is not allowed as an hour value. Hours are only allowed to be given in the range 0 - 23. " \
                      "Set 'allow_24h' to 'True' if you want to translate 24 as a an hour."
            raise ValueError(err_msg)

    rfc3339_time_formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f",
                            "%Y-%m-%dT%H:%M:%Sz", "%Y-%m-%dt%H:%M:%SZ", "%Y-%m-%dt%H:%M:%Sz", "%Y-%m-%dT%H:%M:%S%z",
                            "%Y-%m-%dt%H:%M:%S%z", "%H:%M:%SZ", "%H:%M:%S%z"]
    date_time = None
    # loops through each format and takes the one for which the translation succeeded first
    for i, used_time_format in enumerate(rfc3339_time_formats):
        try:
            date_time = datetime.strptime(string, used_time_format)
            if date_time.tzinfo is None:
                date_time = date_time.replace(tzinfo=timezone.utc)
            break
        except:
            continue

    # add a timedelta of one hour if 24 is allowed as an hour value
    if date_time and allow_24h:
        date_time += timedelta(hours=1)

    return date_time

if __name__ == '__main__':
    pass
