import eofunctions as eof
import numpy as np

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
    assert eof.arccos(x=0) == eof.pi()/2


def test_cosh():
    assert eof.cosh(x=0) == 1


def test_arcosh():
    assert np.isnan(eof.arcosh(x=0))


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

