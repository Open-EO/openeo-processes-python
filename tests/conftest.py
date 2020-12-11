from datetime import datetime
import numpy as np
import pytest
import xarray as xr


@pytest.fixture()
def np_array():
    return get_numpy()


@pytest.fixture()
def xr_array():
    """
    Returns a fixed xarray DataArray array with 4 labelled dimensions
    with coordinates.
    """

    data = get_numpy()
    coords = get_coords()
    xr_data = xr.DataArray(data=data, dims=coords.keys(), coords=coords)

    return xr_data


@pytest.fixture()
def xr_array_B08():
    """
    Returns a fixed xarray DataArray array with 3 labelled dimensions
    with coordinates.
    """

    data = get_numpy()
    data = data[0, :]
    coords = get_coords()
    _ = coords.pop('s')

    xr_data = xr.DataArray(data=data, dims=coords.keys(), coords=coords)

    return xr_data


def get_numpy():
    """
    Returns a fixed numpy array with 4 dimensions.
    """

    steps = get_steps()
    size_x = steps['x']
    size_y = steps['y']

    data = np.ones((3, 2, size_y, size_x))
    data[0, :] *= 8  # identify band 8 by its value
    data[1, :] *= 4  # identify band 4 by its value
    data[2, :] *= 2  # identify band 2 by its value

    data[:, 1, :] *= 10  # second time step of each band multiplied by 10

    return data


def get_coords():

    steps = get_steps()

    coords = {
        's': ['B08', 'B04', 'B02'],
        't': [datetime(2019, 12, 1), datetime(2019, 12, 5)],
        'y': np.arange(55.3, 55.3 + steps['y']),
        'x': np.arange(118.9, 118.9 + steps['x'])
    }
    return coords


def get_steps():
    steps = {
        'y': 5,
        'x': 3
    }
    return steps
