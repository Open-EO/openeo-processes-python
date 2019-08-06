import sys
import os
sys.path.append(os.path.dirname(__file__))
import eofunctions as eof
import numpy as np
from utils_test import assert_list_items


def test_e():
    assert eof.eo_e() == np.e


def test_pi():
    assert eof.eo_pi() == np.pi


def test_int():
    assert eof.eo_int(0) == 0
    assert eof.eo_int(3.5) == 3
    assert eof.eo_int(-0.4) == 0
    assert eof.eo_int(-3.5) == -3


def test_floor():
    assert eof.eo_floor(0) == 0
    assert eof.eo_floor(3.5) == 3
    assert eof.eo_floor(-0.4) == -1
    assert eof.eo_floor(-3.5) == -4


def test_ceil():
    assert eof.eo_ceil(0) == 0
    assert eof.eo_ceil(3.5) == 4
    assert eof.eo_ceil(-0.4) == 0
    assert eof.eo_ceil(-3.5) == -3


def test_round():
    assert eof.eo_round(0) == 0
    assert eof.eo_round(3.56, p=1) == 3.6
    assert eof.eo_round(-0.4444444, p=2) == -0.44
    assert eof.eo_round(-2.5) == -2
    assert eof.eo_round(-3.5) == -4
    assert eof.eo_round(1234.5, p=-2) == 1200


def test_min():
    assert eof.eo_min([1, 0, 3, 2]) == 0
    assert eof.eo_min([5, 2.5, np.nan, -0.7]) == -0.7
    assert np.isnan(eof.eo_min([1, 0, 3, np.nan, 2], ignore_nodata=False))
    assert np.isnan(eof.eo_min([np.nan, np.nan]))


def test_max():
    assert eof.eo_max([1, 0, 3, 2]) == 3
    assert eof.eo_max([5, 2.5, np.nan, -0.7]) == 5
    assert np.isnan(eof.eo_max([1, 0, 3, np.nan, 2], ignore_nodata=False))
    assert np.isnan(eof.eo_max([np.nan, np.nan]))


def test_mean():
    assert eof.eo_mean([1, 0, 3, 2]) == 1.5
    assert eof.eo_mean([9, 2.5, np.nan, -2.5]) == 3
    assert np.isnan(eof.eo_mean([1, np.nan], ignore_nodata=False))
    assert np.isnan(eof.eo_mean([]))


def test_median():
    assert eof.eo_median([1, 3, 3, 6, 7, 8, 9]) == 6
    assert eof.eo_median([1, 2, 3, 4, 5, 6, 8, 9]) == 4.5
    assert eof.eo_median([-1, -0.5, np.nan, 1]) == -0.5
    assert np.isnan(eof.eo_median([-1, 0, np.nan, 1], ignore_nodata=False))
    assert np.isnan(eof.eo_median([]))


def test_sd():
    assert eof.eo_sd([-1, 1, 3, np.nan]) == 2
    assert np.isnan(eof.eo_sd([-1, 1, 3, np.nan], ignore_nodata=False))
    assert np.isnan(eof.eo_sd([]))


def test_variance():
    assert eof.eo_variance([-1, 1, 3]) == 4
    assert eof.eo_variance([2, 3, 3, np.nan, 4, 4, 5]) == 1.1
    assert np.isnan(eof.eo_variance([-1, 1, np.nan, 3], ignore_nodata=False))
    assert np.isnan(eof.eo_variance([]))


def test_quantiles():
    quantiles_1 = eof.eo_quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], probabilities=[0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
    quantiles_1 = [eof.eo_round(quantile, p=2) for quantile in quantiles_1]
    assert quantiles_1 == [2.07, 2.14, 2.28, 2.7, 3.4, 4.5]
    quantiles_2 = eof.eo_quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], q=4)
    quantiles_2 = [eof.eo_round(quantile, p=2) for quantile in quantiles_2]
    assert quantiles_2 == [4, 4.5, 5.5]
    quantiles_3 = eof.eo_quantiles(data=[-1, -0.5, np.nan, 1], q=2)
    quantiles_3 = [eof.eo_round(quantile, p=2) for quantile in quantiles_3]
    assert quantiles_3 == [-0.5]
    quantiles_4 = eof.eo_quantiles(data=[-1, -0.5, np.nan, 1], q=4, ignore_nodata=False)
    assert np.all([np.isnan(quantile) for quantile in quantiles_4]) and len(quantiles_4) == 3
    quantiles_5 = eof.eo_quantiles(data=[], probabilities=[0.1, 0.5])
    assert np.all([np.isnan(quantile) for quantile in quantiles_5]) and len(quantiles_5) == 2


def test_mod():
    assert eof.eo_mod(27, 5) == 2
    assert eof.eo_mod(-27, 5) == 3
    assert eof.eo_mod(27, -5) == -3
    assert eof.eo_mod(-27, -5) == -2
    assert eof.eo_mod(27, 5) == 2
    assert np.isnan(eof.eo_mod(27, np.nan))
    assert np.isnan(eof.eo_mod(np.nan, 5))


def test_absolute():
    assert eof.eo_absolute(0) == 0
    assert eof.eo_absolute(3.5) == 3.5
    assert eof.eo_absolute(-0.4) == 0.4
    assert eof.eo_absolute(-3.5) == 3.5


def test_power():
    assert eof.eo_power(0, 2) == 0
    assert eof.eo_power(2.5, 0) == 1
    assert eof.eo_power(3, 3) == 27
    assert eof.eo_round(eof.eo_power(5, -1), 1) == 0.2
    assert eof.eo_power(1, 0.5) == 1
    assert np.isnan(eof.eo_power(1, np.nan))
    assert np.isnan(eof.eo_power(np.nan, 2))


def test_sgn():
    assert eof.eo_sgn(-2) == -1
    assert eof.eo_sgn(3.5) == 1
    assert eof.eo_sgn(0) == 0
    assert np.isnan(eof.eo_sgn(np.nan))


def test_sqrt():
    assert eof.eo_sqrt(0) == 0
    assert eof.eo_sqrt(1) == 1
    assert eof.eo_sqrt(9) == 3
    assert np.isnan(eof.eo_sqrt(np.nan))


def test_exp():
    assert eof.eo_exp(0) == 1
    assert np.isnan(eof.eo_exp(np.nan))


def test_ln():
    assert eof.eo_ln(eof.eo_e()) == 1
    assert eof.eo_ln(1) == 0


def test_log():
    assert eof.eo_log(10, 10) == 1
    assert eof.eo_log(2, 2) == 1
    assert eof.eo_log(4, 2) == 2
    assert eof.eo_log(1, 16) == 0


def test_cos():
    assert eof.eo_cos(0) == 1


def test_arccos():
    assert eof.eo_arccos(1) == 0


def test_cosh():
    assert eof.eo_cosh(0) == 1


def test_arcosh():
    assert eof.eo_arcosh(1) == 0


def test_sin():
    assert eof.eo_sin(0) == 0


def test_arcsin():
    assert eof.eo_arcsin(0) == 0


def test_sinh():
    assert eof.eo_sinh(0) == 0


def test_arsinh():
    assert eof.eo_arsinh(0) == 0


def test_tan():
    assert eof.eo_tan(0) == 0


def test_arctan():
    assert eof.eo_arctan(0) == 0


def test_tanh():
    assert eof.eo_tanh(0) == 0


def test_artanh():
    assert eof.eo_artanh(0) == 0


def test_arctan2():
    assert eof.eo_arctan2(0, 0) == 0
    assert np.isnan(eof.eo_arctan2(np.nan, 1.5))


def test_cummax():
    assert_list_items(eof.eo_cummax([1, 3, 5, 3, 1]).tolist(), [1, 3, 5, 5, 5])
    assert_list_items(eof.eo_cummax([1, 3, np.nan, 5, 1]).tolist(), [1, 3, np.nan, 5, 5])
    assert_list_items(eof.eo_cummax([1, 3, np.nan, 5, 1], ignore_nodata=False).tolist(),
                      [1, 3, np.nan, np.nan, np.nan])


def test_cummin():
    assert_list_items(eof.eo_cummin([5, 3, 1, 3, 5]).tolist(), [5, 3, 1, 1, 1])
    assert_list_items(eof.eo_cummin([5, 3, np.nan, 1, 5]).tolist(), [5, 3, np.nan, 1, 1])
    assert_list_items(eof.eo_cummin([5, 3, np.nan, 1, 5], ignore_nodata=False).tolist(),
                      [5, 3, np.nan, np.nan, np.nan])


def test_cumproduct():
    assert_list_items(eof.eo_cumproduct([1, 3, 5, 3, 1]).tolist(), [1, 3, 15, 45, 45])
    assert_list_items(eof.eo_cumproduct([1, 2, 3, np.nan, 3, 1]).tolist(), [1, 2, 6, np.nan, 18, 18])
    assert_list_items(eof.eo_cumproduct([1, 2, 3, np.nan, 3, 1], ignore_nodata=False).tolist(),
                      [1, 2, 6, np.nan, np.nan, np.nan])


def test_cumsum():
    assert_list_items(eof.eo_cumsum([1, 3, 5, 3, 1]).tolist(), [1, 4, 9, 12, 13])
    assert_list_items(eof.eo_cumsum([1, 3, np.nan, 3, 1]).tolist(), [1, 4, np.nan, 7, 8])
    assert_list_items(eof.eo_cumsum([1, 3, np.nan, 3, 1], ignore_nodata=False).tolist(),
                      [1, 4, np.nan, np.nan, np.nan])


def test_linear_scale_range():
    assert eof.eo_linear_scale_range(0.3, input_min=-1, input_max=1, output_min=0, output_max=255) == 165.75
    assert eof.eo_linear_scale_range(25.5, input_min=0, input_max=255) == 0.1
    assert np.isnan(eof.eo_linear_scale_range(np.nan, input_min=0, input_max=100))


def test_apply_factor():
    arr = np.random.randn(10)
    assert np.any(eof.eo_apply_factor(arr) == arr)


def test_extrema():
    assert_list_items(eof.eo_extrema([1, 0, 3, 2]), [0, 3])
    assert_list_items(eof.eo_extrema([5, 2.5, np.nan, -0.7]), [-0.7, 5])
    assert_list_items(eof.eo_extrema([1, 0, 3, np.nan, 2], ignore_nodata=False), [np.nan, np.nan])
    assert_list_items(eof.eo_extrema([]), [np.nan, np.nan])


def test_sum():
    assert eof.eo_sum([5, 1]) == 6
    assert eof.eo_sum([-2, 4, 2.5]) == 4.5
    assert np.isnan(eof.eo_sum([1, np.nan], ignore_nodata=False))


def test_subtract():
    assert eof.eo_subtract([5, 10]) == -5
    assert eof.eo_subtract([-2, 4, -2]) == -4
    assert np.isnan(eof.eo_subtract([1, np.nan], ignore_nodata=False))


def test_multiply():
    assert eof.eo_multiply([5, 0]) == 0
    assert eof.eo_multiply([-2, 4, 2.5]) == -20
    assert np.isnan(eof.eo_multiply([1, np.nan], ignore_nodata=False))


def test_divide():
    assert eof.eo_divide([15, 5]) == 3
    assert eof.eo_divide([-2, 4, 2.5]) == -0.2
    assert np.isnan(eof.eo_divide([1, np.nan], ignore_nodata=False))


if __name__ == "__main__":
    test_e()
    #test_pi()
    #test_int()
    #test_floor()
    #test_ceil()
    #test_round()
    #test_min()
    #test_max()
    #test_mean()
    #test_median()
    #test_sd()
    #test_variance()
    #test_quantiles()
    #test_mod()
    #test_absolute()
    #test_power()
    #test_sgn()
    #test_sqrt()
    #test_exp()
    #test_ln()
    #test_log()
    #test_cos()
    #test_arccos()
    #test_cosh()
    #test_arcosh()
    #test_sin()
    #test_arcsin()
    #test_sinh()
    #test_arsinh()
    #test_tan()
    #test_arctan()
    #test_tanh()
    #test_artanh()
    #test_arctan2()
    #test_cummax()
    #test_cummin()
    #test_cumproduct()
    #test_cumsum()
    #test_linear_scale_range()
    #test_apply_factor()
    #test_extrema()
    #test_sum()
    #test_subtract()
    #test_multiply()
    #test_divide()
