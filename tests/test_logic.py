import unittest
import numpy as np
import openeo_processes as eof


class LogicTester(unittest.TestCase):
    """ Tests all logic functions. """

    def test_not_(self):
        """ Tests `not_` function. """
        assert not eof.not_(True)
        assert eof.not_(False)
        assert eof.not_(None) is None

    def test_and_(self):
        """ Tests `and_` function. """
        assert not eof.and_(False, None)
        assert eof.and_(True, None) is None
        assert not eof.and_(False, False)
        assert not eof.and_(True, False)
        assert eof.and_(True, True)

    def test_or_(self):
        """ Tests `or_` function. """
        assert eof.or_(False, None) is None
        assert eof.or_(True, None)
        assert not eof.or_(False, False)
        assert eof.or_(True, False)
        assert eof.or_(True, True)

    def test_xor(self):
        """ Tests `xor` function. """
        assert eof.xor(False, None) is None
        assert eof.xor(True, None) is None
        assert not eof.xor(False, False)
        assert eof.xor(True, False)
        assert not eof.xor(True, True)

    def test_if_(self):
        """ Tests `if_` function. """
        assert eof.if_(True, "A", "B") == "A"
        assert eof.if_(None, "A", "B") == "B"
        assert all(eof.if_(False, [1, 2, 3], [4, 5, 6]) == [4, 5, 6])
        assert eof.if_(True, 123) == 123
        assert eof.if_(False, 1) is None

    def test_any_(self):
        """ Tests `any_` function. """
        assert not eof.any_([False, np.nan])
        assert eof.any_([True, np.nan])
        assert np.isnan(eof.any_([False, np.nan], ignore_nodata=False))
        assert eof.any_([True, np.nan], ignore_nodata=False)
        assert eof.any_([True, False, True, False])
        assert eof.any_([True, False])
        assert not eof.any_([False, False])
        assert eof.any_([True])
        assert np.isnan(eof.any_([np.nan], ignore_nodata=False))
        assert np.isnan(eof.any_([]))

    def test_all_(self):
        """ Tests `all_` function. """
        assert not eof.all_([False, np.nan])
        assert eof.all_([True, np.nan])
        assert not eof.all_([False, np.nan], ignore_nodata=False)
        assert np.isnan(eof.all_([True, np.nan], ignore_nodata=False))
        assert not eof.all_([True, False, True, False])
        assert not eof.all_([True, False])
        assert eof.any_([True, True])
        assert eof.any_([True])
        assert np.isnan(eof.any_([np.nan], ignore_nodata=False))
        assert np.isnan(eof.any_([]))

if __name__ == '__main__':
    unittest.main()
