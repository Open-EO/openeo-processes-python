import xarray
import dask
import copy
import datetime

import numpy as np


def eval_datatype(data):
    is_list = isinstance(data, list)
    is_np = isinstance(data, np.ndarray) | is_list
    is_xar = isinstance(data, xarray.DataArray)
    is_dar = isinstance(data, dask.array.core.Array)
    is_num = isinstance(data, (int, float, str, np.integer, np.float))
    is_datetime = isinstance(data, datetime.datetime)
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


def build_multi_dim_index(index_name, shape, axis):
    dims = len(shape)
    index_name = str(index_name)
    expand_dim_exprs = [["None"] * (dims - 1)] * (dims - 1)
    for i, elem in enumerate(expand_dim_exprs):
        elem_cp = copy.deepcopy(elem)
        elem_cp[i] = ":"
        expand_dim_exprs[i] = ",".join(elem_cp)
    expand_dim_exprs.insert(axis, None)
    strings_select = []
    for i, n in enumerate(shape):
        if i == axis:
            strings_select.append(index_name)
        else:
            strings_select.append("np.arange({})[{}]".format(n, expand_dim_exprs[i]))

    return ",".join(strings_select)
