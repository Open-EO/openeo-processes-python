import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import eofunctions as eof
from utils_test import assert_list_items


def test_and():
    assert eof.and_(expressions=[False, np.nan]) == False
    assert eof.and_(expressions=[True, np.nan]) == True
    assert eof.and_(expressions=[False, np.nan], ignore_nodata=False) == False
    assert np.isnan(eof.and_(expressions=[True, np.nan], ignore_nodata=False))
    assert eof.and_(expressions=[True, False, True, False]) == False
    assert eof.and_(expressions=[True, False]) == False
    assert eof.and_(expressions=[True, True]) == True
    assert eof.and_(expressions=[True]) == True
    assert np.isnan(eof.and_(expressions=[]))


def test_or():
    assert eof.or_(expressions=[False, np.nan]) == False
    assert eof.or_(expressions=[True, np.nan]) == True
    assert np.isnan(eof.or_(expressions=[False, np.nan], ignore_nodata=False))
    assert eof.or_(expressions=[True, np.nan], ignore_nodata=False) == True
    assert eof.or_(expressions=[True, False, True, False]) == True
    assert eof.or_(expressions=[True, False]) == True
    assert eof.or_(expressions=[False, False]) == False
    assert eof.or_(expressions=[True]) == True
    assert np.isnan(eof.or_(expressions=[]))


def test_xor():
    assert eof.xor_(expressions=[False, np.nan]) == False
    assert eof.xor_(expressions=[True, np.nan]) == True
    assert np.isnan(eof.xor_(expressions=[False, np.nan], ignore_nodata=False))
    assert np.isnan(eof.xor_(expressions=[True, np.nan], ignore_nodata=False))
    assert eof.xor_(expressions=[True, False, True, False]) == False
    assert eof.xor_(expressions=[True, False]) == True
    assert eof.xor_(expressions=[True, True]) == False
    assert eof.xor_(expressions=[True]) == True
    assert np.isnan(eof.xor_(expressions=[]))

def test_if():
    assert eof.if_(expression=True) == True
    assert np.isnan(eof.if_(expression=np.nan))
    assert eof.if_(expression=False) == False
    assert eof.if_(expression=True, accept="A") == "A"
    assert_list_items(eof.if_(expression=False, accept=[1, 2, 3], reject=[4, 5, 6]), [4, 5, 6])