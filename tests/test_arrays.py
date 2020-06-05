"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import unittest
import numpy as np
import eofunctions as eof


class ArrayTester(unittest.TestCase):
    """ Tests all array functions. """

    def test_array_contains(self):
        """ Tests `array_contains` function. """
        assert eof.array_contains([1, 2, 3], value=2)
        assert not eof.array_contains(["A", "B", "C"], value="b")
        assert not eof.array_contains([1, 2, 3], value="2")
        assert eof.array_contains([1, 2, 3], value=2)
        assert eof.array_contains([1, 2, np.nan], value=np.nan)
        assert eof.array_contains([[1, 2], [3, 4]], value=[1, 2])
        assert not eof.array_contains([[1, 2],[3, 4]], value=2)
        assert eof.array_contains([{"a": "b"}, {"c": "d"}], value={"a": "b"})

    def test_array_element(self):
        """ Tests `array_element` function. """
        assert eof.array_element([9, 8, 7, 6, 5], index=2) == 7
        assert eof.array_element(["A", "B", "C"], index=0) == "A"
        assert np.isnan(eof.array_element([], index=0, return_nodata=True))

        # multi-dim
        test_array = np.empty((3, 2, 2))
        test_array[0, :, :] = np.array([[1, 2], [3, 4]])
        test_array[1, :, :] = np.array([[1, 2], [3, 4]]) * 20
        test_array[2, :, :] = np.array([[1, 2], [3, 4]]) * 500
        array_i2_d0 = np.array([
                                [500, 1000],
                                [1500, 2000]
                                ])
        array_i0_d1 = np.array([
                                [1, 2],
                                [20, 40],
                                [500, 1000]
                                ])
        assert np.isclose(eof.array_element(test_array, index=2, dimension=0), array_i2_d0, equal_nan=True).all()
        assert np.isclose(eof.array_element(test_array, index=0, dimension=1), array_i0_d1, equal_nan=True).all()

    def test_count(self):
        """ Tests `count` function. """
        assert eof.count([]) == 0
        assert eof.count([1, 0, 3, 2]) == 4
        assert eof.count(["ABC", np.nan]) == 1
        assert eof.count([False, np.nan], condition=True) == 2
        assert eof.count([0, 1, 2, 3, 4, 5, np.nan], condition=eof.eo_gt, context={'y': 2})

    #TODO: add test
    def test_array_apply(self):
        """ Tests `array_apply` function. """
        pass

    # TODO: add test
    def test_array_filter(self):
        """ Tests `array_filter` function. """
        pass

    # TODO: add test
    def test_array_find(self):
        """ Tests `array_find` function. """
        pass

    # TODO: add test
    def test_array_labels(self):
        """ Tests `array_labels` function. """
        pass

    def test_first(self):
        """ Tests `first` function. """
        assert eof.first([1, 0, 3, 2]) == 1
        assert eof.first([np.nan, "A", "B"]) == "A"
        assert np.isnan(eof.first([np.nan, 2, 3], ignore_nodata=False))
        assert np.isnan(eof.first([]))

        # 2D test
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        first_elem_ref = np.array([[[3., 2.], [1., 2.]]])
        first_elem = eof.first(test_arr)
        assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        first_elem_ref = np.array([[[np.nan, 2.], [1., 2.]]])
        first_elem = eof.first(test_arr, ignore_nodata=False)
        assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()

    def test_last(self):
        """ Tests `last` function. """
        assert eof.last([1, 0, 3, 2]) == 2
        assert eof.last(["A", "B", np.nan]) == "B"
        assert np.isnan(eof.last([0, 1, np.nan], ignore_nodata=False))
        assert np.isnan(eof.last([]))

        # 2D test
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 3]], [[1, 2], [1, np.nan]]])
        last_elem_ref = np.array([[[1., 2.], [1., 3.]]])
        last_elem = eof.last(test_arr)
        assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        last_elem_ref = np.array([[[1., 2.], [1., np.nan]]])
        last_elem = eof.last(test_arr, ignore_nodata=False)
        assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()

    def test_order(self):
        """ Tests `order` function. """
        self.assertListEqual(eof.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]).tolist(),
                             [1, 2, 8, 5, 0, 4, 7, 9, 10])
        self.assertListEqual(eof.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], nodata=True).tolist(),
                             [1, 2, 8, 5, 0, 4, 7, 9, 10, 3, 6])
        self.assertListEqual(eof.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True).tolist(),
                             [9, 10, 7, 4, 0, 5, 8, 2, 1, 3, 6])
        self.assertListEqual(eof.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=False).tolist(),
                             [6, 3, 9, 10, 7, 4, 0, 5, 8, 2, 1])

    def test_rearrange(self):
        """ Tests `rearrange` function. """
        self.assertListEqual(eof.rearrange([5, 4, 3], [2, 1, 0]).tolist(), [3, 4, 5])
        self.assertListEqual(eof.rearrange([5, 4, 3, 2], [1, 3]).tolist(), [4, 2])
        self.assertListEqual(eof.rearrange([5, 4, 3, 2], [0, 2, 1, 3]).tolist(), [5, 3, 4, 2])

    def test_sort(self):
        """ Tests `sort` function. """
        self.assertListEqual(eof.sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]).tolist(),
                             [-1, 2, 3, 4, 6, 7, 8, 9, 9])
        assert np.isclose(eof.sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True),
                          [9, 9, 8, 7, 6, 4, 3, 2, -1, np.nan, np.nan], equal_nan=True).all()

if __name__ == "__main__":
    unittest.main()
