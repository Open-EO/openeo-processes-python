import eofunctions as eof


def test_is_nan():
    assert eof.eo_is_nan(1) == False
    assert eof.eo_is_nan('Test') == True


def test_is_nodata():
    assert eof.eo_is_nodata(1) == False
    assert eof.eo_is_nodata('Test') == False
    assert eof.eo_is_nodata(None) == True


def test_is_valid():
    assert eof.eo_is_valid(1) == True
    assert eof.eo_is_valid('Test') == True
    assert eof.eo_is_valid(None) == False


if __name__ == '__main__':
    test_is_nan()
    test_is_nodata()
    test_is_valid()
