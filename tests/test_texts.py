import eofunctions as eof
import numpy as np

# TODO: test exceptions
def test_text_begins():
    assert eof.text_begins(data="Lorem ipsum dolor sit amet", pattern="amet") == False
    assert eof.text_begins(data="Lorem ipsum dolor sit amet", pattern="Lorem") == True
    assert eof.text_begins(data="Lorem ipsum dolor sit amet", pattern="lorem") == False
    assert eof.text_begins(data="Lorem ipsum dolor sit amet", pattern="lorem", case_sensitive=False) == True
    assert eof.text_begins(data="Ä", pattern="ä", case_sensitive=False) == True


def test_text_ends():
    assert eof.text_ends(data="Lorem ipsum dolor sit amet", pattern="amet") == True
    assert eof.text_ends(data="Lorem ipsum dolor sit amet", pattern="AMET") == False
    assert eof.text_ends(data="Lorem ipsum dolor sit amet", pattern="Lorem") == False
    assert eof.text_ends(data="Lorem ipsum dolor sit amet", pattern="AMET", case_sensitive=False) == True
    assert eof.text_ends(data="Ä", pattern="ä", case_sensitive=False) == True


def test_text_contains():
    assert eof.text_contains(data="Lorem ipsum dolor sit amet", pattern="openEO") == False
    assert eof.text_contains(data="Lorem ipsum dolor sit amet", pattern="ipsum dolor") == True
    assert eof.text_contains(data="Lorem ipsum dolor sit amet", pattern="Ipsum Dolor") == False
    assert eof.text_contains(data="Lorem ipsum dolor sit amet", pattern = "SIT", case_sensitive=False) == True
    assert eof.text_contains(data="ÄÖÜ", pattern="ö", case_sensitive=False) == True


def test_text_merge():
    assert eof.text_merge(data=["Hello", "World"], separator=" ") == "Hello World"
    assert eof.text_merge(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) == "1234567890"
    assert eof.text_merge(data=[np.nan, True, False, 1, -1.5, "ß"], separator="\n") == "nan\ntrue\nfalse\n1\n-1.5\nß"
    assert eof.text_merge(data=[2, 0], separator=1) == "210"
    assert eof.text_merge(data=[]) == ""
