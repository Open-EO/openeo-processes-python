import numpy as np
import xarray
import dask
import copy

def eval_datatype(data):
    is_list = isinstance(data, list)
    is_np = isinstance(data, np.ndarray) | (is_list and isinstance(data[0], np.ndarray))
    is_xar = isinstance(data, xarray.DataArray)
    is_dar = isinstance(data, dask.array.core.Array)
    is_num = isinstance(data, (int, float))

    if is_np:
        datatype = "numpy"
    elif is_xar:
        datatype = "xar"
    elif is_dar:
        datatype = "dar"
    elif is_num:
        datatype = "num"
    else:
        datatype = None

    return datatype


def process(processor):
    def fun_wrapper(*args, **kwargs):
        cls = processor()
        datatype = eval_datatype(args[0])
        if datatype == "numpy":
            cls_fun = getattr(cls, "exec_np")
            if isinstance(args[0], list):
                args[0] = list2nparray(args[0])
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


def is_empty(data):
    if len(data) == 0:
        return True
    else:
        return False


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