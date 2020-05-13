import xarray
import dask
import copy
import datetime
import re
from datetime import timezone
from datetime import timedelta
from datetime import time
from datetime import datetime

import numpy as np


def eval_datatype(data):
    is_list = isinstance(data, list)
    is_np = isinstance(data, np.ndarray) | is_list
    is_xar = isinstance(data, xarray.DataArray)
    is_dar = isinstance(data, dask.array.core.Array)
    is_num = isinstance(data, (int, float, str, np.integer, np.float))
    is_datetime = isinstance(data, datetime)
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
    else:
        datatype = None

    return datatype


def process(processor, data_key="data"):
    def fun_wrapper(*args, **kwargs):
        cls = processor()

        if not args:
            datatype = eval_datatype(kwargs[data_key])
        else:
            datatype = eval_datatype(args[0])

        if datatype == "np":
            cls_fun = getattr(cls, "exec_np")
            if not args:
                if isinstance(kwargs[data_key], list):
                    kwargs[data_key] = list2nparray(kwargs[data_key])
            else:
                if isinstance(args[0], list):
                    args = list(args)
                    args[0] = list2nparray(args[0])
                    args = tuple(args)

        elif datatype == "xar":
            cls_fun = getattr(cls, "exec_xar")
        elif datatype == "dar":
            cls_fun = getattr(cls, "exec_dar")
        elif datatype in ["num", "none", "dt"]:
            cls_fun = getattr(cls, "exec_num")
        else:
            raise Exception('Datatype unknown.')

        return cls_fun(*args, **kwargs)

    return fun_wrapper


def list2nparray(x):
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


def str2time(string, allow_24=False):
    # handle timezone formatting
    if "+" in string:
        string_parts = string.split('+')
        string_parts[-1] = string_parts[-1].replace(':', '')
        string = "+".join(string_parts)

    if "t" in string.lower():  # special handling due to - sign in date string
        if "-" in string[10:]:
            string_parts = string[10:].split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = string[:10] + "-".join(string_parts)
    else:
        if "-" in string:
            string_parts = string.split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = "-".join(string_parts)

    if allow_24:
        pattern = re.compile("24:\d{2}:\d{2}")
        pattern_match = re.search(pattern, string)
        if pattern_match:
            old_sub_string = pattern_match.group()
            new_sub_string = "23" + old_sub_string[2:]
            string = string.replace(old_sub_string, new_sub_string)

    rfc3339_time_formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f",
                            "%Y-%m-%dT%H:%M:%Sz", "%Y-%m-%dt%H:%M:%SZ", "%Y-%m-%dt%H:%M:%Sz", "%Y-%m-%dT%H:%M:%S%z",
                            "%Y-%m-%dt%H:%M:%S%z", "%H:%M:%SZ", "%H:%M:%S%z"]
    date_time = None
    for i, used_time_format in enumerate(rfc3339_time_formats):
        try:
            date_time = datetime.strptime(string, used_time_format)
            if date_time.tzinfo is None:
                date_time = date_time.replace(tzinfo=timezone.utc)
            if i == 0:
                date_time_max = datetime.combine(date_time.date(), time()) + timedelta(hours=24) \
                                - timedelta(seconds=1)
                date_time = (datetime.combine(date_time.date(), time()).replace(tzinfo=timezone.utc),
                             date_time_max.replace(tzinfo=timezone.utc))
            break
        except:
            continue

    if date_time and allow_24:
        date_time += timedelta(hours=1)

    return date_time

if __name__ == '__main__':
    create_slices(0, (3, 4, 5), 0)
