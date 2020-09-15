import datetime

import dask.array.core
import numpy as np
import pytest
import xarray
from openeo_processes.utils import eval_datatype


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
