from datetime import datetime
import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="class")
def test_data(request):
    class TestDataDriver:
        def __init__(self):
            self.steps = {'y': 5, 'x': 3}
            self.coords_4d = {
                's': ['B08', 'B04', 'B02'],
                't': [datetime(2019, 12, 1), datetime(2019, 12, 5)],
                'y': np.arange(55.3, 55.3 + self.steps['y']),
                'x': np.arange(118.9, 118.9 + self.steps['x'])
            }
            self.coords_3d = {
                't': [datetime(2019, 12, 1), datetime(2019, 12, 5)],
                'y': np.arange(55.3, 55.3 + self.steps['y']),
                'x': np.arange(118.9, 118.9 + self.steps['x'])
            }
            self._get_numpy()
            self._get_xarray()

        def _get_numpy(self):
            """
            Returns a fixed numpy array with 4 dimensions.
            """

            data = np.ones((3, 2, self.steps['y'], self.steps['x']))
            data[0, :] *= 8  # identify band 8 by its value
            data[1, :] *= 4  # identify band 4 by its value
            data[2, :] *= 2  # identify band 2 by its value

            data[:, 1, :] *= 10  # second t-step of each band multiplied by 10

            self.np_data_4d = data
            self.np_data_3d = data[0, :]

        def _get_xarray(self):
            """
            Returns a fixed xarray DataArray array with 3 labelled dimensions
            with coordinates.
            """

            self.xr_data_4d = xr.DataArray(data=self.np_data_4d,
                                           dims=self.coords_4d.keys(),
                                           coords=self.coords_4d)
            self.xr_data_4d.attrs['crs'] = 'EPSG:4326'
            self.xr_data_3d = xr.DataArray(data=self.np_data_3d,
                                           dims=self.coords_3d.keys(),
                                           coords=self.coords_3d)
            self.xr_data_3d.attrs['crs'] = 'EPSG:4326'

    request.cls.test_data = TestDataDriver()
