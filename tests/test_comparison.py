import unittest
import openeo_processes as eof


class ComparisonTester(unittest.TestCase):
    """ Tests all comparison functions. """

    def test_is_nan(self):
        """ Tests `is_nan` function. """
        assert not eof.is_nan(1)
        assert eof.is_nan('Test')

    def test_is_nodata(self):
        """ Tests `is_nodata` function. """
        assert not eof.is_nodata(1)
        assert not eof.is_nodata('Test')
        assert eof.is_nodata(None)

    def test_is_valid(self):
        """ Tests `is_valid` function. """
        assert eof.is_valid(1)
        assert eof.is_valid('Test')
        assert not eof.is_valid(None)

    def test_eq(self):
        """ Tests `eq` function. """
        assert eof.eq(1, None) is None
        assert eof.eq(1, 1)
        assert not eof.eq(1, "1")
        assert not eof.eq(1.02, 1, delta=0.01)
        assert eof.eq(-1, -1.001, delta=0.01)
        assert eof.eq(115, 110, delta=10)
        assert not eof.eq("Test", "test")
        assert eof.eq("Test", "test", case_sensitive=False)
        assert eof.eq("Ä", "ä", case_sensitive=False)
        assert eof.eq("00:00:00+00:00", "00:00:00Z")
        assert not eof.eq("2018-01-01T12:00:00Z", "2018-01-01T12:00:00")
        assert eof.eq("2018-01-01T00:00:00Z", "2018-01-01T01:00:00+01:00")

    def test_neq(self):
        """ Tests `neq` function. """
        assert eof.neq(1, None) is None
        assert not eof.neq(1, 1)
        assert eof.neq(1, "1")
        assert eof.neq(1.02, 1, delta=0.01)
        assert not eof.neq(-1, -1.001, delta=0.01)
        assert not eof.neq(115, 110, delta=10)
        assert eof.neq("Test", "test")
        assert not eof.neq("Test", "test", case_sensitive=False)
        assert not eof.neq("Ä", "ä", case_sensitive=False)
        assert not eof.neq("00:00:00+00:00", "00:00:00Z")
        assert eof.neq("2018-01-01T12:00:00Z", "2018-01-01T12:00:00")
        assert not eof.neq("2018-01-01T00:00:00Z", "2018-01-01T01:00:00+01:00")

    def test_gt(self):
        """ Tests `gt` function. """
        assert eof.gt(1, None) is None
        assert not eof.gt(0, 0)
        assert eof.gt(2, 1)
        assert eof.gt(-0.5, -0.6)
        assert eof.gt("00:00:00Z", "00:00:00+01:00")
        assert not eof.gt("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert not eof.gt("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_gte(self):
        """ Tests `gte` function. """
        assert eof.gte(1, None) is None
        assert eof.gte(0, 0)
        assert not eof.gte(1, 2)
        assert eof.gte(-0.5, -0.6)
        assert eof.gte("00:00:00Z", "00:00:00+01:00")
        assert not eof.gte("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert eof.gte("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_lt(self):
        """ Tests `lt` function. """
        assert eof.lt(1, None) is None
        assert not eof.lt(0, 0)
        assert eof.lt(1, 2)
        assert not eof.lt(-0.5, -0.6)
        assert eof.lt("00:00:00+01:00", "00:00:00Z")
        assert eof.lt("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert not eof.lt("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_lte(self):
        """ Tests `lte` function. """
        assert eof.lte(1, None) is None
        assert eof.lte(0, 0)
        assert eof.lte(1, 2)
        assert not eof.lte(-0.5, -0.6)
        assert eof.lte("00:00:00+01:00", "00:00:00Z")
        assert eof.lte("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert eof.lte("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_between(self):
        """ Tests `between` function. """
        assert eof.between(None, min=0, max=1) is None
        assert not eof.between(0.5, min=1, max=0)
        assert not eof.between(-0.5, min=0, max=-1)
        assert eof.between("00:59:59Z", min="01:00:00+01:00", max="01:00:00Z")
        assert eof.between("2018-07-23T17:22:45Z", min="2018-01-01T00:00:00Z", max="2018-12-31T23:59:59Z")
        assert not eof.between("2000-01-01", min="2018-01-01", max="2020-01-01")
        assert not eof.between("2018-12-31T17:22:45Z", min="2018-01-01", max="2018-12-31", exclude_max=True)

if __name__ == '__main__':
    unittest.main()
