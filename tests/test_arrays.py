import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import eofunctions as eof
from utils_test import assert_list_items


def test_array_contains():
    assert eof.eo_array_contains([1, 2, 3], element=2) == True
    assert eof.eo_array_contains(["A", "B", "C"], element="b") == False
    assert eof.eo_array_contains([1, 2, 3], element="2") == False
    assert eof.eo_array_contains([1, 2, 3], element=2) == True
    assert eof.eo_array_contains([1, 2, np.nan], element=np.nan) == True
    assert eof.eo_array_contains([[1, 2], [3, 4]], element=[1, 2]) == True
    assert eof.eo_array_contains([[1, 2],[3, 4]], element=2) == False
    assert eof.eo_array_contains([{"a": "b"}, {"c": "d"}], element={"a": "b"}) == True


def test_array_element():
    assert eof.eo_array_element([9, 8, 7, 6, 5], index=2) == 7
    assert eof.eo_array_element(["A", "B", "C"], index=0) == "A"
    assert np.isnan(eof.eo_array_element([], index=0, return_nodata=True))


def test_count():
    assert eof.eo_count([]) == 0
    assert eof.eo_count([1, 0, 3, 2]) == 4
    assert eof.eo_count(["ABC", np.nan]) == 1
    assert eof.eo_count([False, np.nan], expression=True) == 2
    assert eof.eo_count([False, np.nan], expression=lambda x: eof.eo_gt(x, 2))


def test_first():
    assert eof.eo_first([1, 0, 3, 2]) == 1
    assert eof.eo_first([np.nan, "A", "B"]) == "A"
    assert np.isnan(eof.eo_first([np.nan, 2, 3], ignore_nodata=False))
    assert np.isnan(eof.eo_first([]))


def test_last():
    assert eof.eo_last([1, 0, 3, 2]) == 2
    assert eof.eo_last(["A", "B", np.nan]) == "B"
    assert np.isnan(eof.eo_last([0, 1, np.nan], ignore_nodata=False))
    assert np.isnan(eof.eo_last([]))


def test_order():
    assert_list_items(eof.eo_order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]),
                      [1, 2, 8, 5, 0, 4, 7, 9, 10])
    assert_list_items(eof.eo_order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], nodata=True),
                      [1, 2, 8, 5, 0, 4, 7, 9, 10, 3, 6])
    assert_list_items(eof.eo_order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True),
                      [9, 10, 7, 4, 0, 5, 8, 2, 1, 3, 6])
    assert_list_items(eof.eo_order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=False),
                      [6, 3, 9, 10, 7, 4, 0, 5, 8, 2, 1])


def test_rearrange():
    assert_list_items(eof.eo_rearrange([5, 4, 3], [2, 1, 0]), [3, 4, 5])
    assert_list_items(eof.eo_rearrange([5, 4, 3, 2], [1, 3]), [4, 2])
    assert_list_items(eof.eo_rearrange([5, 4, 3, 2], [0, 2, 1, 3]), [5, 3, 4, 2])


def test_sort():
    assert_list_items(eof.eo_sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]), [-1, 2, 3, 4, 6, 7, 8, 9, 9])
    assert_list_items(eof.eo_sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True),
                      [9, 9, 8, 7, 6, 4, 3, 2, -1, np.nan, np.nan])


def test_clip():
    assert_list_items(eof.eo_clip([-2, -1, 0, 1, 2], min=-1, max=1), [-1, -1, 0, 1, 1])
    assert_list_items(eof.eo_clip([-0.1, -0.001, np.nan, 0, 0.25, 0.75, 1.001, np.nan], min=0, max=1),
                      [0, 0, np.nan, 0, 0.25, 0.75, 1, np.nan])


if __name__ == "__main__":
    test_array_contains()
    #test_array_element()
    #test_count()
    #test_first()
    #test_last()
    #test_order()
    #test_rearrange()
    #test_sort()
    #test_clip()
