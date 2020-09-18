import datetime

import dask.array.core
import numpy as np
import pytest
import xarray
from openeo_processes.utils import eval_datatype, get_process, has_process


@pytest.mark.parametrize(["data", "expected"], [
    (None, "NoneType"),
    (True, "bool"),
    (False, "bool"),
    ("123", "str"),
    (123, "int"),
    (123.456, "float"),
    ([1, 2, 3], "list"),
    ((1, 2, 3), "tuple"),
    ({1, 2, 3}, "set"),
    ({1: 2, 3: 4}, "dict"),
    (lambda x, y: x + y, "function"),
    (datetime.datetime.now(), "datetime"),
    (np.array([1, 2, 3]), "numpy"),
    (xarray.DataArray([1, 2, 3]), "xarray"),
    (dask.array.core.from_array([1, 2, 3]), "dask"),
])
def test_eval_datatype(data, expected):
    assert eval_datatype(data) == expected


def test_has_process():
    assert has_process("add")
    assert has_process("multiply")
    assert not has_process("foobar")


def test_get_process():
    fun = get_process("add")
    assert fun(2, 3) == 5
    fun = get_process("multiply")
    assert fun(2, 3) == 6
    fun = get_process("sum")
    assert fun([1, 2, 3, 4, 5, 6]) == 21
