"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import unittest
import numpy as np
from copy import deepcopy
import eofunctions as eof


class MathTester(unittest.TestCase):
    """ Tests all math functions. """

    def test_e(self):
        """ Tests `e` function. """
        assert eof.e() == np.e

    def test_pi(self):
        """ Tests `pi` function. """
        assert eof.pi() == np.pi

    def test_floor(self):
        """ Tests `floor` function. """
        assert eof.floor(0) == 0
        assert eof.floor(3.5) == 3
        assert eof.floor(-0.4) == -1
        assert eof.floor(-3.5) == -4

    def test_ceil(self):
        """ Tests `ceil` function. """
        assert eof.ceil(0) == 0
        assert eof.ceil(3.5) == 4
        assert eof.ceil(-0.4) == 0
        assert eof.ceil(-3.5) == -3

    def test_int(self):
        """ Tests `int` function. """
        assert eof.int(0) == 0
        assert eof.int(3.5) == 3
        assert eof.int(-0.4) == 0
        assert eof.int(-3.5) == -3

    def test_round(self):
        """ Tests `round` function. """
        assert eof.round(0) == 0
        assert eof.round(3.56, p=1) == 3.6
        assert eof.round(-0.4444444, p=2) == -0.44
        assert eof.round(-2.5) == -2
        assert eof.round(-3.5) == -4
        assert eof.round(1234.5, p=-2) == 1200

    def test_exp(self):
        """ Tests `exp` function. """
        assert eof.exp(0) == 1
        assert np.isnan(eof.exp(np.nan))

    def test_log(self):
        """ Tests `log` function. """
        assert eof.log(10, 10) == 1
        assert eof.log(2, 2) == 1
        assert eof.log(4, 2) == 2
        assert eof.log(1, 16) == 0

    def test_ln(self):
        """ Tests `ln` function. """
        assert eof.ln(eof.e()) == 1
        assert eof.ln(1) == 0

    def test_cos(self):
        """ Tests `cos` function. """
        assert eof.cos(0) == 1

    def test_arccos(self):
        """ Tests `arccos` function. """
        assert eof.arccos(1) == 0

    def test_cosh(self):
        """ Tests `cosh` function. """
        assert eof.cosh(0) == 1

    def test_arcosh(self):
        """ Tests `arcosh` function. """
        assert eof.arcosh(1) == 0

    def test_sin(self):
        """ Tests `sin` function. """
        assert eof.sin(0) == 0

    def test_arcsin(self):
        """ Tests `arcsin` function. """
        assert eof.arcsin(0) == 0

    def test_sinh(self):
        """ Tests `sinh` function. """
        assert eof.sinh(0) == 0

    def test_arsinh(self):
        """ Tests `arsinh` function. """
        assert eof.arsinh(0) == 0

    def test_tan(self):
        """ Tests `tan` function. """
        assert eof.tan(0) == 0

    def test_arctan(self):
        """ Tests `arctan` function. """
        assert eof.arctan(0) == 0

    def test_tanh(self):
        """ Tests `tanh` function. """
        assert eof.tanh(0) == 0

    def test_artanh(self):
        """ Tests `artanh` function. """
        assert eof.artanh(0) == 0

    def test_arctan2(self):
        """ Tests `arctan2` function. """
        assert eof.arctan2(0, 0) == 0
        assert np.isnan(eof.arctan2(np.nan, 1.5))

    def test_linear_scale_range(self):
        """ Tests `linear_scale_range` function. """
        assert eof.linear_scale_range(0.3, input_min=-1, input_max=1, output_min=0, output_max=255) == 165.75
        assert eof.linear_scale_range(25.5, input_min=0, input_max=255) == 0.1
        assert np.isnan(eof.linear_scale_range(np.nan, input_min=0, input_max=100))

    def test_scale(self):
        """ Tests `scale` function. """
        arr = np.random.randn(10)
        assert np.all(eof.scale(arr) == arr)

    def test_mod(self):
        """ Tests `mod` function. """
        assert eof.mod(27, 5) == 2
        assert eof.mod(-27, 5) == 3
        assert eof.mod(27, -5) == -3
        assert eof.mod(-27, -5) == -2
        assert eof.mod(27, 5) == 2
        assert np.isnan(eof.mod(27, np.nan))
        assert np.isnan(eof.mod(np.nan, 5))

    def test_absolute(self):
        """ Tests `absolute` function. """
        assert eof.absolute(0) == 0
        assert eof.absolute(3.5) == 3.5
        assert eof.absolute(-0.4) == 0.4
        assert eof.absolute(-3.5) == 3.5

    def test_sgn(self):
        """ Tests `sgn` function. """
        assert eof.sgn(-2) == -1
        assert eof.sgn(3.5) == 1
        assert eof.sgn(0) == 0
        assert np.isnan(eof.sgn(np.nan))

    def test_sqrt(self):
        """ Tests `sqrt` function. """
        assert eof.sqrt(0) == 0
        assert eof.sqrt(1) == 1
        assert eof.sqrt(9) == 3
        assert np.isnan(eof.sqrt(np.nan))

    def test_power(self):
        """ Tests `power` function. """
        assert eof.power(0, 2) == 0
        assert eof.power(2.5, 0) == 1
        assert eof.power(3, 3) == 27
        assert eof.round(eof.power(5, -1), 1) == 0.2
        assert eof.power(1, 0.5) == 1
        assert eof.power(1, None) is None
        assert eof.power(None, 2) is None

    def test_mean(self):
        """ Tests `mean` function. """
        assert eof.mean([1, 0, 3, 2]) == 1.5
        assert eof.mean([9, 2.5, np.nan, -2.5]) == 3
        assert np.isnan(eof.mean([1, np.nan], ignore_nodata=False))
        assert np.isnan(eof.mean([]))

    def test_min(self):
        """ Tests `min` function. """
        assert eof.min([1, 0, 3, 2]) == 0
        assert eof.min([5, 2.5, np.nan, -0.7]) == -0.7
        assert np.isnan(eof.min([1, 0, 3, np.nan, 2], ignore_nodata=False))
        assert np.isnan(eof.min([np.nan, np.nan]))

    def test_max(self):
        """ Tests `max` function. """
        assert eof.max([1, 0, 3, 2]) == 3
        assert eof.max([5, 2.5, np.nan, -0.7]) == 5
        assert np.isnan(eof.max([1, 0, 3, np.nan, 2], ignore_nodata=False))
        assert np.isnan(eof.max([np.nan, np.nan]))

    def test_median(self):
        """ Tests `median` function. """
        assert eof.median([1, 3, 3, 6, 7, 8, 9]) == 6
        assert eof.median([1, 2, 3, 4, 5, 6, 8, 9]) == 4.5
        assert eof.median([-1, -0.5, np.nan, 1]) == -0.5
        assert np.isnan(eof.median([-1, 0, np.nan, 1], ignore_nodata=False))
        assert np.isnan(eof.median([]))

    def test_sd(self):
        """ Tests `sd` function. """
        assert eof.sd([-1, 1, 3, np.nan]) == 2
        assert np.isnan(eof.sd([-1, 1, 3, np.nan], ignore_nodata=False))
        assert np.isnan(eof.sd([]))

    def test_variance(self):
        """ Tests `variance` function. """
        assert eof.variance([-1, 1, 3]) == 4
        assert eof.variance([2, 3, 3, np.nan, 4, 4, 5]) == 1.1
        assert np.isnan(eof.variance([-1, 1, np.nan, 3], ignore_nodata=False))
        assert np.isnan(eof.variance([]))

    def test_extrema(self):
        """ Tests `extrema` function. """
        self.assertListEqual(eof.extrema([1, 0, 3, 2]), [0, 3])
        self.assertListEqual(eof.extrema([5, 2.5, np.nan, -0.7]), [-0.7, 5])
        assert np.isclose(eof.extrema([1, 0, 3, np.nan, 2], ignore_nodata=False), [np.nan, np.nan],
                          equal_nan=True).all()
        assert np.isclose(eof.extrema([]), [np.nan, np.nan], equal_nan=True).all()

    def test_clip(self):
        """ Tests `clip` function. """
        assert eof.clip(-5, min_x=-1, max_x=1) == -1
        assert eof.clip(10.001, min_x=1, max_x=10) == 10
        assert eof.clip(0.000001, min_x=0, max_x=0.02) == 0.000001
        assert eof.clip(None, min_x=0, max_x=1) is None

        # test array clipping
        assert np.isclose(eof.clip([-2, -1, 0, 1, 2], min_x=-1, max_x=1), [-1, -1, 0, 1, 1], equal_nan=True).all()
        assert np.isclose(eof.clip([-0.1, -0.001, np.nan, 0, 0.25, 0.75, 1.001, np.nan], min_x=0, max_x=1),
                          [0, 0, np.nan, 0, 0.25, 0.75, 1, np.nan], equal_nan=True).all()

    def test_quantiles(self):
        """ Tests `quantiles` function. """
        quantiles_1 = eof.quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], probabilities=[0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
        quantiles_1 = [eof.round(quantile, p=2) for quantile in quantiles_1]
        assert quantiles_1 == [2.07, 2.14, 2.28, 2.7, 3.4, 4.5]
        quantiles_2 = eof.quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], q=4)
        quantiles_2 = [eof.round(quantile, p=2) for quantile in quantiles_2]
        assert quantiles_2 == [4, 4.5, 5.5]
        quantiles_3 = eof.quantiles(data=[-1, -0.5, np.nan, 1], q=2)
        quantiles_3 = [eof.round(quantile, p=2) for quantile in quantiles_3]
        assert quantiles_3 == [-0.5]
        quantiles_4 = eof.quantiles(data=[-1, -0.5, np.nan, 1], q=4, ignore_nodata=False)
        assert np.all([np.isnan(quantile) for quantile in quantiles_4]) and len(quantiles_4) == 3
        quantiles_5 = eof.quantiles(data=[], probabilities=[0.1, 0.5])
        assert np.all([np.isnan(quantile) for quantile in quantiles_5]) and len(quantiles_5) == 2

    def test_cummin(self):
        """ Tests `cummin` function. """
        self.assertListEqual(eof.cummin([5, 3, 1, 3, 5]).tolist(), [5, 3, 1, 1, 1])
        assert np.isclose(eof.cummin([5, 3, np.nan, 1, 5]), [5, 3, np.nan, 1, 1], equal_nan=True).all()
        assert np.isclose(eof.cummin([5, 3, np.nan, 1, 5], ignore_nodata=False),
                          [5, 3, np.nan, np.nan, np.nan], equal_nan=True).all()

    def test_cummax(self):
        """ Tests `cummax` function. """
        self.assertListEqual(eof.cummax([1, 3, 5, 3, 1]).tolist(), [1, 3, 5, 5, 5])
        assert np.isclose(eof.cummax([1, 3, np.nan, 5, 1]), [1, 3, np.nan, 5, 5], equal_nan=True).all()
        assert np.isclose(eof.cummax([1, 3, np.nan, 5, 1], ignore_nodata=False),
                          [1, 3, np.nan, np.nan, np.nan], equal_nan=True).all()

    def test_cumproduct(self):
        """ Tests `cumproduct` function. """
        self.assertListEqual(eof.cumproduct([1, 3, 5, 3, 1]).tolist(), [1, 3, 15, 45, 45])
        assert np.isclose(eof.cumproduct([1, 2, 3, np.nan, 3, 1]), [1, 2, 6, np.nan, 18, 18], equal_nan=True).all()
        assert np.isclose(eof.cumproduct([1, 2, 3, np.nan, 3, 1], ignore_nodata=False),
                          [1, 2, 6, np.nan, np.nan, np.nan], equal_nan=True).all()

    def test_cumsum(self):
        """ Tests `cumsum` function. """
        self.assertListEqual(eof.cumsum([1, 3, 5, 3, 1]).tolist(), [1, 4, 9, 12, 13])
        assert np.isclose(eof.cumsum([1, 3, np.nan, 3, 1]), [1, 4, np.nan, 7, 8], equal_nan=True).all()
        assert np.isclose(eof.cumsum([1, 3, np.nan, 3, 1], ignore_nodata=False),
                          [1, 4, np.nan, np.nan, np.nan], equal_nan=True).all()

    def test_sum(self):
        """ Tests `sum` function. """
        assert eof.sum([5, 1]) == 6
        assert eof.sum([-2, 4, 2.5]) == 4.5
        assert np.isnan(eof.sum([1, np.nan], ignore_nodata=False))

    def test_product(self):
        """ Tests `product` function. """
        assert eof.product([5, 0]) == 0
        assert eof.product([-2, 4, 2.5]) == -20
        assert np.isnan(eof.product([1, np.nan], ignore_nodata=False))
        assert eof.product([-1]) == -1
        assert np.isnan(eof.product([np.nan], ignore_nodata=False))
        assert np.isnan(eof.product([]))

        C = np.ones((2, 5, 5)) * 100
        assert np.sum(eof.product(C) - np.ones((5, 5)) * 10000) == 0
        assert np.sum(eof.product(deepcopy(C), extra_values=[2]) - np.ones((5, 5)) * 20000) == 0
        assert np.sum(eof.product(deepcopy(C), extra_values=[2, 3]) - np.ones((5, 5)) * 60000) == 0

    def test_add(self):
        """ Tests `add` function. """
        assert eof.add(5, 2.5) == 7.5
        assert eof.add(-2, -4) == -6
        assert eof.add(1, None) is None

    def test_subtract(self):
        """ Tests `subtract` function. """
        assert eof.subtract(5, 2.5) == 2.5
        assert eof.subtract(-2, 4) == -6
        assert eof.subtract(1, None) is None

    def test_multiply(self):
        """ Tests `multiply` function. """
        assert eof.multiply(5, 2.5) == 12.5
        assert eof.multiply(-2, -4) == 8
        assert eof.multiply(1, None) is None

    def test_divide(self):
        """ Tests `divide` function. """
        assert eof.divide(5, 2.5) == 2.
        assert eof.divide(-2, 4) == -0.5
        assert eof.divide(1, None) is None

    # TODO: add test
    def test_normalized_difference(self):
        """ Tests `normalized_difference` function. """
        pass

if __name__ == "__main__":
    unittest.main()
