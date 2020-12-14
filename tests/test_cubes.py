"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import unittest
import pytest
import openeo_processes as oeop


@pytest.mark.usefixtures("test_data")
class CubesTester(unittest.TestCase):
    """ Tests all cubes functions. """

    def test_reduce_dimension(self):
        """ Tests `reduce_dimension` function. """

        # xarray tests
        # Reduce spectral dimension using the process `sum`
        # Take sum over 's' dimension in a 4d array
        reduced = oeop.reduce_dimension(self.test_data.xr_data_4d, reducer=oeop.sum, dimension='s')
        self.assertListEqual(
            list(reduced[:, 0, 0].data),
            [14, 140]
            )

    def test_save_result(self):
        """ Tests `reduce_dimension` function. """

        # xarray tests
        oeop.save_result(self.test_data.xr_data_3d, "out.tif")


if __name__ == "__main__":
    unittest.main()
