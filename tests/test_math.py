import sys
import os
sys.path.append(os.path.dirname(__file__))
import eofunctions as eof
import numpy as np
from utils_test import assert_list_items

# TODO: test exceptions


def test_e():
    assert eof.e() == np.e


def test_pi():
    assert eof.pi() == np.pi


def test_is_nan():
    assert eof.is_nan(x=1) == False
    assert eof.is_nan(x='Test') == True


def test_is_nodata():
    assert eof.is_nodata(x=1) == False
    assert eof.is_nodata(x='Test') == False
    assert eof.is_nodata(x=None) == True


def test_is_valid():
    assert eof.is_valid(x=1) == True
    assert eof.is_valid(x='Test') == True
    assert eof.is_valid(x=None) == False


def test_int():
    assert eof.int_(x=0) == 0
    assert eof.int_(x=3.5) == 3
    assert eof.int_(x=-0.4) == 0
    assert eof.int_(x=-3.5) == -3


def test_floor():
    assert eof.floor(x=0) == 0
    assert eof.floor(x=3.5) == 3
    assert eof.floor(x=-0.4) == -1
    assert eof.floor(x=-3.5) == -4


def test_ceil():
    assert eof.ceil(x=0) == 0
    assert eof.ceil(x=3.5) == 4
    assert eof.ceil(x=-0.4) == 0
    assert eof.ceil(x=-3.5) == -3


def test_round():
    assert eof.round_(x=0) == 0
    assert eof.round_(x=3.56, p=1) == 3.6
    assert eof.round_(x=-0.4444444, p=2) == -0.44
    assert eof.round_(x=-2.5) == -2
    assert eof.round_(x=-3.5) == -4
    assert eof.round_(x=1234.5, p=-2) == 1200


def test_min():
    assert eof.min_(data=[1, 0, 3, 2]) == 0
    assert eof.min_(data=[5, 2.5, np.nan, -0.7]) == -0.7
    assert np.isnan(eof.min_(data=[1, 0, 3, np.nan, 2], ignore_nodata=False))
    assert np.isnan(eof.min_(data=[np.nan, np.nan]))


def test_max():
    assert eof.max_(data=[1, 0, 3, 2]) == 3
    assert eof.max_(data=[5, 2.5, np.nan, -0.7]) == 5
    assert np.isnan(eof.max_(data=[1, 0, 3, np.nan, 2], ignore_nodata=False))
    assert np.isnan(eof.max_(data=[np.nan, np.nan]))


def test_mean():
    assert eof.mean(data=[1, 0, 3, 2]) == 1.5
    assert eof.mean(data=[9, 2.5, np.nan, -2.5]) == 3
    assert np.isnan(eof.mean(data=[1, np.nan], ignore_nodata=False))
    assert np.isnan(eof.mean(data=[]))


def test_median():
    assert eof.median(data=[1, 3, 3, 6, 7, 8, 9]) == 6
    assert eof.median(data=[1, 2, 3, 4, 5, 6, 8, 9]) == 4.5
    assert eof.median(data=[-1, -0.5, np.nan, 1]) == -0.5
    assert np.isnan(eof.median(data=[-1, 0, np.nan, 1], ignore_nodata=False))
    assert np.isnan(eof.median(data=[]))


def test_sd():
    assert eof.sd(data=[-1, 1, 3, np.nan]) == 2
    assert np.isnan(eof.sd(data=[-1, 1, 3, np.nan], ignore_nodata=False))
    assert np.isnan(eof.sd(data=[]))


def test_variance():
    assert eof.variance(data=[-1, 1, 3]) == 4
    assert eof.variance(data=[2, 3, 3, np.nan, 4, 4, 5]) == 1.1
    assert np.isnan(eof.variance(data=[-1, 1, np.nan, 3], ignore_nodata=False))
    assert np.isnan(eof.variance(data=[]))


def test_quantiles():
    quantiles_1 = eof.quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], probabilities=[0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
    quantiles_1 = [eof.round_(quantile, p=2) for quantile in quantiles_1]
    assert quantiles_1 == [2.07, 2.14, 2.28, 2.7, 3.4, 4.5]
    quantiles_2 = eof.quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], q=4)
    quantiles_2 = [eof.round_(quantile, p=2) for quantile in quantiles_2]
    assert quantiles_2 == [4, 4.5, 5.5]
    quantiles_3 = eof.quantiles(data=[-1, -0.5, np.nan, 1], q=2)
    quantiles_3 = [eof.round_(quantile, p=2) for quantile in quantiles_3]
    assert quantiles_3 == [-0.5]
    quantiles_4 = eof.quantiles(data=[-1, -0.5, np.nan, 1], q=4, ignore_nodata=False)
    assert np.all([np.isnan(quantile) for quantile in quantiles_4]) and len(quantiles_4) == 3
    quantiles_5 = eof.quantiles(data=[], probabilities=[0.1, 0.5])
    assert np.all([np.isnan(quantile) for quantile in quantiles_5]) and len(quantiles_5) == 2


def test_mod():
    assert eof.mod(x=27, y=5) == 2
    assert eof.mod(x=-27, y=5) == 3
    assert eof.mod(x=27, y=-5) == -3
    assert eof.mod(x=-27, y=-5) == -2
    assert eof.mod(x=27, y=5) == 2
    assert np.isnan(eof.mod(x=27, y=np.nan))
    assert np.isnan(eof.mod(x=np.nan, y=5))


def test_absolute():
    assert eof.absolute(x=0) == 0
    assert eof.absolute(x=3.5) == 3.5
    assert eof.absolute(x=-0.4) == 0.4
    assert eof.absolute(x=-3.5) == 3.5


def test_power():
    assert eof.power(base=0, p=2) == 0
    assert eof.power(base=2.5, p=0) == 1
    assert eof.power(base=3, p=3) == 27
    assert eof.round_(eof.power(base=5, p=-1), 1) == 0.2
    assert eof.power(base=1, p=0.5) == 1
    assert np.isnan(eof.power(base=1, p=np.nan))
    assert np.isnan(eof.power(base=np.nan, p=2))


def test_sgn():
    assert eof.sgn(x=-2) == -1
    assert eof.sgn(x=3.5) == 1
    assert eof.sgn(x=0) == 0
    assert np.isnan(eof.sgn(x=np.nan))


def test_sqrt():
    assert eof.sqrt(x=0) == 0
    assert eof.sqrt(x=1) == 1
    assert eof.sqrt(x=9) == 3
    assert np.isnan(eof.sqrt(x=np.nan))


def test_exp():
    assert eof.exp(x=0) == 1
    assert np.isnan(eof.exp(x=np.nan))


def test_ln():
    assert eof.ln(x=eof.e()) == 1
    assert eof.ln(x=1) == 0


def test_log():
    assert eof.log(x=10, base=10) == 1
    assert eof.log(x=2, base=2) == 1
    assert eof.log(x=4, base=2) == 2
    assert eof.log(x=1, base=16) == 0


def test_cos():
    assert eof.cos(x=0) == 1


def test_arccos():
    assert eof.arccos(x=1) == 0


def test_cosh():
    assert eof.cosh(x=0) == 1


def test_arcosh():
    assert eof.arcosh(x=1) == 0


def test_sin():
    assert eof.sin(x=0) == 0


def test_arcsin():
    assert eof.arcsin(x=0) == 0


def test_sinh():
    assert eof.sinh(x=0) == 0


def test_arsinh():
    assert eof.arsinh(x=0) == 0


def test_tan():
    assert eof.tan(x=0) == 0


def test_arctan():
    assert eof.arctan(x=0) == 0


def test_tanh():
    assert eof.tanh(x=0) == 0


def test_artanh():
    assert eof.artanh(x=0) == 0


def test_arctan2():
    assert eof.arctan2(y=0, x=0) == 0
    assert np.isnan(eof.arctan2(y=np.nan, x=1.5))


def test_cummax():
    assert_list_items(eof.cummax(data=[1, 3, 5, 3, 1]).tolist(), [1, 3, 5, 5, 5])
    assert_list_items(eof.cummax(data=[1, 3, np.nan, 5, 1]).tolist(), [1, 3, np.nan, 5, 5])
    assert_list_items(eof.cummax(data=[1, 3, np.nan, 5, 1], ignore_nodata=False).tolist(),
                      [1, 3, np.nan, np.nan, np.nan])


def test_cummin():
    assert_list_items(eof.cummin(data=[5, 3, 1, 3, 5]).tolist(), [5, 3, 1, 1, 1])
    assert_list_items(eof.cummin(data=[5, 3, np.nan, 1, 5]).tolist(), [5, 3, np.nan, 1, 1])
    assert_list_items(eof.cummin(data=[5, 3, np.nan, 1, 5], ignore_nodata=False).tolist(),
                      [5, 3, np.nan, np.nan, np.nan])


def test_cumproduct():
    assert_list_items(eof.cumproduct(data=[1, 3, 5, 3, 1]).tolist(), [1, 3, 15, 45, 45])
    assert_list_items(eof.cumproduct(data=[1, 2, 3, np.nan, 3, 1]).tolist(), [1, 2, 6, np.nan, 18, 18])
    assert_list_items(eof.cumproduct(data=[1, 2, 3, np.nan, 3, 1], ignore_nodata=False).tolist(),
                      [1, 2, 6, np.nan, np.nan, np.nan])


def test_cumsum():
    assert_list_items(eof.cumsum(data=[1, 3, 5, 3, 1]).tolist(), [1, 4, 9, 12, 13])
    assert_list_items(eof.cumsum(data=[1, 3, np.nan, 3, 1]).tolist(), [1, 4, np.nan, 7, 8])
    assert_list_items(eof.cumsum(data=[1, 3, np.nan, 3, 1], ignore_nodata=False).tolist(),
                      [1, 4, np.nan, np.nan, np.nan])


def test_eq():
    assert np.isnan(eof.eq(x=1, y=np.nan))
    assert eof.eq(x=1, y=1) == True
    assert eof.eq(x=1, y="1") == False
    assert eof.eq(x=1.02, y=1, delta=0.01) == False
    assert eof.eq(x=-1, y=-1.001, delta=0.01) == True
    assert eof.eq(x=115, y=110, delta=10) == True
    assert eof.eq(x="Test", y="test") == False
    assert eof.eq(x="Test", y="test", case_sensitive=False) == True
    assert eof.eq(x="Ä", y="ä", case_sensitive=False) == True
    assert eof.eq(x="00:00:00+00:00", y="00:00:00Z") == True
    assert eof.eq(x="2018-01-01T12:00:00Z", y="2018-01-01T12:00:00") == False
    assert eof.eq(x="2018-01-01T00:00:00Z", y="2018-01-01T01:00:00+01:00") == True


def test_neq():
    assert np.isnan(eof.neq(x=1, y=np.nan))
    assert eof.neq(x=1, y=1) == False
    assert eof.neq(x=1, y="1") == True
    assert eof.neq(x=1.02, y=1, delta=0.01) == True
    assert eof.neq(x=-1, y=-1.001, delta=0.01) == False
    assert eof.neq(x=115, y=110, delta=10) == False
    assert eof.neq(x="Test", y="test") == True
    assert eof.neq(x="Test", y="test", case_sensitive=False) == False
    assert eof.neq(x="Ä", y="ä", case_sensitive=False) == False
    assert eof.neq(x="00:00:00+00:00", y="00:00:00Z") == False
    assert eof.neq(x="2018-01-01T12:00:00Z", y="2018-01-01T12:00:00") == True
    assert eof.neq(x="2018-01-01T00:00:00Z", y="2018-01-01T01:00:00+01:00") == False


def test_gt():
    assert np.isnan(eof.gt(x=1, y=np.nan))
    assert eof.gt(x=0, y=0) == False
    assert eof.gt(x=2, y=1) == True
    assert eof.gt(x=-0.5, y=-0.6) == True
    assert eof.gt(x="00:00:00Z", y="00:00:00+01:00") == True
    assert eof.gt(x="1950-01-01T00:00:00Z", y="2018-01-01T12:00:00Z") == False
    assert eof.gt(x="2018-01-01T12:00:00+00:00", y="2018-01-01T12:00:00Z") == False


def test_gte():
    assert np.isnan(eof.gte(x=1, y=np.nan))
    assert eof.gte(x=0, y=0) == True
    assert eof.gte(x=1, y=2) == False
    assert eof.gte(x=-0.5, y=-0.6) == True
    assert eof.gte(x="00:00:00Z", y="00:00:00+01:00") == True
    assert eof.gte(x="1950-01-01T00:00:00Z", y="2018-01-01T12:00:00Z") == False
    assert eof.gte(x="2018-01-01T12:00:00+00:00", y="2018-01-01T12:00:00Z") == True


def test_lt():
    assert np.isnan(eof.lt(x=1, y=np.nan))
    assert eof.lt(x=0, y=0) == False
    assert eof.lt(x=1, y=2) == True
    assert eof.lt(x=-0.5, y=-0.6) == False
    assert eof.lt(x="00:00:00+01:00", y="00:00:00Z") == True
    assert eof.lt(x="1950-01-01T00:00:00Z", y="2018-01-01T12:00:00Z") == True
    assert eof.lt(x="2018-01-01T12:00:00+00:00", y="2018-01-01T12:00:00Z") == False


def test_lte():
    assert np.isnan(eof.lte(x=1, y=np.nan))
    assert eof.lte(x=0, y=0) == True
    assert eof.lte(x=1, y=2) == True
    assert eof.lte(x=-0.5, y=-0.6) == False
    assert eof.lte(x="00:00:00+01:00", y="00:00:00Z") == True
    assert eof.lte(x="1950-01-01T00:00:00Z", y="2018-01-01T12:00:00Z") == True
    assert eof.lte(x="2018-01-01T12:00:00+00:00", y="2018-01-01T12:00:00Z") == True


def test_between():
    assert np.isnan(eof.between(x=np.nan, min=0, max=1))
    assert eof.between(x=0.5, min=1, max=0) == False
    assert eof.between(x=-0.5, min=0, max=-1) == False
    assert eof.between(x="00:59:59Z", min="01:00:00+01:00", max="01:00:00Z") == True
    assert eof.between(x="2018-07-23T17:22:45Z", min="2018-01-01T00:00:00Z", max="2018-12-31T23:59:59Z") == True
    assert eof.between(x="2000-01-01", min="2018-01-01", max="2020-01-01") == False
    assert eof.between(x="2018-12-31T17:22:45Z", min="2018-01-01", max="2018-12-31", exclude_max=True) == False


def test_linear_scale_range():
    assert eof.linear_scale_range(x=0.3, input_min=-1, input_max=1, output_min=0, output_max=255) == 165.75
    assert eof.linear_scale_range(x=25.5, input_min=0, input_max=255) == 0.1
    assert np.isnan(eof.linear_scale_range(x=np.nan, input_min=0, input_max=100))


def test_apply_factor():
    arr = np.random.randn(10)
    assert np.any(eof.apply_factor(arr) == arr)


def test_extrema():
    assert_list_items(eof.eo_extrema([1, 0, 3, 2]), [0, 3])
    assert_list_items(eof.eo_extrema([5, 2.5, np.nan, -0.7]), [-0.7, 5])
    assert_list_items(eof.eo_extrema([1, 0, 3, np.nan, 2], ignore_nodata=False), [np.nan, np.nan])
    assert_list_items(eof.eo_extrema([]), [np.nan, np.nan])

def test_sum():
    assert eof.eo_sum_([5, 1]) == 6
    assert eof.eo_sum_([-2, 4, 2.5]) == 4.5
    assert np.isnan(eof.eo_sum_([1, np.nan], ignore_nodata=False))


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
