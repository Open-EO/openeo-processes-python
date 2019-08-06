import numpy as np
import eofunctions as eof

from utils_test import assert_list_items


def test_and():
    assert eof.eo_and([False, np.nan]) == False
    assert eof.eo_and([True, np.nan]) == True
    assert eof.eo_and([False, np.nan], ignore_nodata=False) == False
    assert np.isnan(eof.eo_and([True, np.nan], ignore_nodata=False))
    assert eof.eo_and([True, False, True, False]) == False
    assert eof.eo_and([True, False]) == False
    assert eof.eo_and([True, True]) == True
    assert eof.eo_and([True]) == True
    assert np.isnan(eof.eo_and([]))


def test_or():
    assert eof.eo_or([False, np.nan]) == False
    assert eof.eo_or([True, np.nan]) == True
    assert np.isnan(eof.eo_or([False, np.nan], ignore_nodata=False))
    assert eof.eo_or([True, np.nan], ignore_nodata=False) == True
    assert eof.eo_or([True, False, True, False]) == True
    assert eof.eo_or([True, False]) == True
    assert eof.eo_or([False, False]) == False
    assert eof.eo_or([True]) == True
    assert np.isnan(eof.eo_or([]))


def test_xor():
    assert eof.eo_xor([False, np.nan]) == False
    assert eof.eo_xor([True, np.nan]) == True
    assert np.isnan(eof.eo_xor([False, np.nan], ignore_nodata=False))
    assert np.isnan(eof.eo_xor([True, np.nan], ignore_nodata=False))
    assert eof.eo_xor([True, False, True, False]) == False
    assert eof.eo_xor([True, False]) == True
    assert eof.eo_xor([True, True]) == False
    assert eof.eo_xor([True]) == True
    assert np.isnan(eof.eo_xor([]))


def test_if():
    assert eof.eo_if(True) == True
    assert np.isnan(eof.eo_if(np.nan))
    assert eof.eo_if(False) == False
    assert eof.eo_if(True, accept="A") == "A"
    assert_list_items(eof.eo_if(False, accept=[1, 2, 3], reject=[4, 5, 6]), [4, 5, 6])


def test_eq():
    assert np.isnan(eof.eo_eq(1, np.nan))
    assert eof.eo_eq(1, 1) == True
    assert eof.eo_eq(1, "1") == False
    assert eof.eo_eq(1.02, 1, delta=0.01) == False
    assert eof.eo_eq(-1, -1.001, delta=0.01) == True
    assert eof.eo_eq(115, 110, delta=10) == True
    assert eof.eo_eq("Test", "test") == False
    assert eof.eo_eq("Test", "test", case_sensitive=False) == True
    assert eof.eo_eq("Ä", "ä", case_sensitive=False) == True
    assert eof.eo_eq("00:00:00+00:00", "00:00:00Z") == True
    assert eof.eo_eq("2018-01-01T12:00:00Z", "2018-01-01T12:00:00") == False
    assert eof.eo_eq("2018-01-01T00:00:00Z", "2018-01-01T01:00:00+01:00") == True


def test_neq():
    assert np.isnan(eof.eo_neq(1, np.nan))
    assert eof.eo_neq(1, 1) == False
    assert eof.eo_neq(1, "1") == True
    assert eof.eo_neq(1.02, 1, delta=0.01) == True
    assert eof.eo_neq(-1, -1.001, delta=0.01) == False
    assert eof.eo_neq(115, 110, delta=10) == False
    assert eof.eo_neq("Test", "test") == True
    assert eof.eo_neq("Test", "test", case_sensitive=False) == False
    assert eof.eo_neq("Ä", "ä", case_sensitive=False) == False
    assert eof.eo_neq("00:00:00+00:00", "00:00:00Z") == False
    assert eof.eo_neq("2018-01-01T12:00:00Z", "2018-01-01T12:00:00") == True
    assert eof.eo_neq("2018-01-01T00:00:00Z", "2018-01-01T01:00:00+01:00") == False


def test_gt():
    assert np.isnan(eof.eo_gt(1, np.nan))
    assert eof.eo_gt(0, 0) == False
    assert eof.eo_gt(2, 1) == True
    assert eof.eo_gt(-0.5, -0.6) == True
    assert eof.eo_gt("00:00:00Z", "00:00:00+01:00") == True
    assert eof.eo_gt("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z") == False
    assert eof.eo_gt("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z") == False


def test_gte():
    assert np.isnan(eof.eo_gte(1, np.nan))
    assert eof.eo_gte(0, 0) == True
    assert eof.eo_gte(1, 2) == False
    assert eof.eo_gte(-0.5, -0.6) == True
    assert eof.eo_gte("00:00:00Z", "00:00:00+01:00") == True
    assert eof.eo_gte("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z") == False
    assert eof.eo_gte("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z") == True


def test_lt():
    assert np.isnan(eof.eo_lt(1, np.nan))
    assert eof.eo_lt(0, 0) == False
    assert eof.eo_lt(1, 2) == True
    assert eof.eo_lt(-0.5, -0.6) == False
    assert eof.eo_lt("00:00:00+01:00", "00:00:00Z") == True
    assert eof.eo_lt("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z") == True
    assert eof.eo_lt("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z") == False


def test_lte():
    assert np.isnan(eof.eo_lte(1, np.nan))
    assert eof.eo_lte(0, 0) == True
    assert eof.eo_lte(1, 2) == True
    assert eof.eo_lte(-0.5, -0.6) == False
    assert eof.eo_lte("00:00:00+01:00", "00:00:00Z") == True
    assert eof.eo_lte("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z") == True
    assert eof.eo_lte("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z") == True


def test_between():
    assert np.isnan(eof.eo_between(np.nan, min=0, max=1))
    assert eof.eo_between(0.5, min=1, max=0) == False
    assert eof.eo_between(-0.5, min=0, max=-1) == False
    assert eof.eo_between("00:59:59Z", min="01:00:00+01:00", max="01:00:00Z") == True
    assert eof.eo_between("2018-07-23T17:22:45Z", min="2018-01-01T00:00:00Z", max="2018-12-31T23:59:59Z") == True
    assert eof.eo_between("2000-01-01", min="2018-01-01", max="2020-01-01") == False
    assert eof.eo_between("2018-12-31T17:22:45Z", min="2018-01-01", max="2018-12-31", exclude_max=True) == False

if __name__ == '__main__':
    #test_and()
    #test_or()
    #test_xor()
    #test_if()
    #test_eq()
    #test_neq()
    #test_gt()
    #test_gte()
    #test_lt()
    #test_lte()
    test_between()
