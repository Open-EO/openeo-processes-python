import eofunctions as eof
import numpy as np


def test_text_begins():
    assert eof.text_begins("Lorem ipsum dolor sit amet", pattern="amet") == False
    assert eof.text_begins("Lorem ipsum dolor sit amet", pattern="Lorem") == True
    assert eof.text_begins("Lorem ipsum dolor sit amet", pattern="lorem") == False
    assert eof.text_begins("Lorem ipsum dolor sit amet", pattern="lorem", case_sensitive=False) == True
    assert eof.text_begins("Ä", pattern="ä", case_sensitive=False) == True


def test_text_ends():
    assert eof.text_ends("Lorem ipsum dolor sit amet", pattern="amet") == True
    assert eof.text_ends("Lorem ipsum dolor sit amet", pattern="AMET") == False
    assert eof.text_ends("Lorem ipsum dolor sit amet", pattern="Lorem") == False
    assert eof.text_ends("Lorem ipsum dolor sit amet", pattern="AMET", case_sensitive=False) == True
    assert eof.text_ends("Ä", pattern="ä", case_sensitive=False) == True


def test_text_contains():
    assert eof.text_contains("Lorem ipsum dolor sit amet", pattern="openEO") == False
    assert eof.text_contains("Lorem ipsum dolor sit amet", pattern="ipsum dolor") == True
    assert eof.text_contains("Lorem ipsum dolor sit amet", pattern="Ipsum Dolor") == False
    assert eof.text_contains("Lorem ipsum dolor sit amet", pattern = "SIT", case_sensitive=False) == True
    assert eof.text_contains("ÄÖÜ", pattern="ö", case_sensitive=False) == True


def test_text_merge():
    assert eof.text_merge(["Hello", "World"], separator=" ") == "Hello World"
    assert eof.text_merge([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) == "1234567890"
    assert eof.text_merge([np.nan, True, False, 1, -1.5, "ß"], separator="\n") == "nan\ntrue\nfalse\n1\n-1.5\nß"
    assert eof.text_merge([2, 0], separator=1) == "210"
    assert eof.text_merge([]) == ""


if __name__ == '__main__':
    test_text_begins()
    #test_text_ends()
    #test_text_contains()
    #test_text_merge()