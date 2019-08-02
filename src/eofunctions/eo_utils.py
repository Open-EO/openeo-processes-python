import xarray
import dask
import copy

import numpy as np
import pandas as pd


def eval_datatype(data):
    is_list = isinstance(data, list)
    is_np = isinstance(data, np.ndarray) | is_list
    is_xar = isinstance(data, xarray.DataArray)
    is_dar = isinstance(data, dask.array.core.Array)
    is_num = isinstance(data, (int, float))

    if is_np:
        datatype = "np"
    elif is_xar:
        datatype = "xar"
    elif is_dar:
        datatype = "dar"
    elif is_num:
        datatype = "num"
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
        elif datatype == "num":
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


def eo_is_empty(data):
    if len(data) == 0:
        return True
    else:
        return False

# TODO check the no data value functions if they do the job correctly
def eo_is_nan(x):
    if isinstance(x, (int, float, np.ndarray, xarray.DataArray, dask.array.core.Array)):
        return pd.isnull(x)
    else:
        return True


def eo_is_nodata(x):

    if isinstance(x, (int, float, np.ndarray, xarray.DataArray, dask.array.core.Array)):
        return pd.isnull(x)
    else:
        # if x in [np.nan, None]:
        #    return True
        # else:
        return False


def eo_is_valid(x, unary=True):

    if isinstance(x, (np.ndarray, xarray.DataArray, dask.array.core.Array)):
        if eo_is_empty(x):
            return False
        else:
            is_valid = (~pd.isnull(x) & (x != np.inf))
            if unary:
                return is_valid.all()
            else:
                return is_valid
    else:
        if x not in [np.nan, np.inf, None]:
            return True
        else:
            return False
