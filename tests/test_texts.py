import unittest
import numpy as np
import eofunctions as eof


class TextTester(unittest.TestCase):
    """ Tests all math functions. """

    def test_text_begins(self):
        """ Tests `text_begins` function. """
        assert not eof.text_begins("Lorem ipsum dolor sit amet", pattern="amet")
        assert eof.text_begins("Lorem ipsum dolor sit amet", pattern="Lorem")
        assert not eof.text_begins("Lorem ipsum dolor sit amet", pattern="lorem")
        assert eof.text_begins("Lorem ipsum dolor sit amet", pattern="lorem", case_sensitive=False)
        assert eof.text_begins("Ä", pattern="ä", case_sensitive=False)
        assert eof.text_begins(None, pattern="None") is None

    def test_text_ends(self):
        """ Tests `text_ends` function. """
        assert eof.text_ends("Lorem ipsum dolor sit amet", pattern="amet")
        assert not eof.text_ends("Lorem ipsum dolor sit amet", pattern="AMET")
        assert not eof.text_ends("Lorem ipsum dolor sit amet", pattern="Lorem")
        assert eof.text_ends("Lorem ipsum dolor sit amet", pattern="AMET", case_sensitive=False)
        assert eof.text_ends("Ä", pattern="ä", case_sensitive=False)
        assert eof.text_ends(None, pattern="None") is None

    def test_text_contains(self):
        """ Tests `text_contains` function. """
        assert not eof.text_contains("Lorem ipsum dolor sit amet", pattern="openEO")
        assert eof.text_contains("Lorem ipsum dolor sit amet", pattern="ipsum dolor")
        assert not eof.text_contains("Lorem ipsum dolor sit amet", pattern="Ipsum Dolor")
        assert eof.text_contains("Lorem ipsum dolor sit amet", pattern="SIT", case_sensitive=False)
        assert eof.text_contains("ÄÖÜ", pattern="ö", case_sensitive=False)
        assert eof.text_contains(None, pattern="None") is None

    def test_text_merge(self):
        """ Tests `text_merge` function. """
        assert eof.text_merge(["Hello", "World"], separator=" ") == "Hello World"
        assert eof.text_merge([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) == "1234567890"
        assert eof.text_merge([np.nan, True, False, 1, -1.5, "ß"], separator="\n") == "nan\ntrue\nfalse\n1\n-1.5\nß"
        assert eof.text_merge([2, 0], separator=1) == "210"
        assert eof.text_merge([]) == ""


if __name__ == '__main__':
    unittest.main()
