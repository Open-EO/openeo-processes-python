import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import eofunctions as eof
from utils_test import assert_list_items


def test_if():
    assert eof.if_(expression=True) == True
    assert np.isnan(eof.if_(expression=np.nan))
    assert eof.if_(expression=False) == False
    assert eof.if_(expression=True, accept="A") == "A"
    assert_list_items(eof.if_(expression=False, accept=[1, 2, 3], reject=[4, 5, 6]), [4, 5, 6])