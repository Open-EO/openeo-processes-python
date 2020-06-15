
def text_begins(text, pattern, case_sensitive=True):
    """
    Checks whether the text (also known as `text`) contains the text specified for `pattern` at the beginning.
    The no-data value None is passed through and therefore gets propagated.

    Parameters
    ----------
    text : str
        Text in which to find something at the beginning.
    pattern : str
        Text to find at the beginning of `text`.
    case_sensitive : bool, optional
        Case sensitive comparison can be disabled by setting this parameter to False (default is True).

    Returns
    -------
    bool :
        True if `text` begins with `pattern`, False otherwise.

    """
    if text is None:
        return None

    if case_sensitive:
        return text.startswith(pattern)
    else:
        return text.lower().startswith(pattern.lower())


def text_ends(text, pattern, case_sensitive=True):
    """
    Checks whether the text (also known as `text`) contains the text specified for `pattern` at the end.
    The no-data value None is passed through and therefore gets propagated.

    Parameters
    ----------
    text : str
        Text in which to find something at the end.
    pattern : str
        Text to find at the end of `text`.
    case_sensitive : bool, optional
        Case sensitive comparison can be disabled by setting this parameter to False (default is True).

    Returns
    -------
    bool :
        True if `text` ends with `pattern`, False otherwise.

    """
    if text is None:
        return None

    if case_sensitive:
        return text.endswith(pattern)
    else:
        return text.lower().endswith(pattern.lower())


def text_contains(text, pattern, case_sensitive=True):
    """
    Checks whether the text (also known as `text`) contains the text specified for `pattern`.
    The no-data value None is passed through and therefore gets propagated.

    Parameters
    ----------
    text : str
        String in which to find something in.
    pattern : str
        String to find in `text`.
    case_sensitive : bool, optional
        Case sensitive comparison can be disabled by setting this parameter to False (default is True).

    Returns
    -------
    bool :
        True if `text` contains the `pattern`, False otherwise.

    """
    if text is None:
        return None

    if case_sensitive:
        return pattern in text
    else:
        return pattern.lower() in text.lower()


def text_merge(data, separator=''):
    """
    Merges string representations of a set of elements together to a single string, with the separator
    between each element.

    Parameters
    ----------
    data : list
        A list of elements. Numbers, boolean values and None values get converted to their (lower case) string
        representation. For example: 1 (int), -1.5 (float), True / False (boolean values)
    separator : object, optional
        A separator to put between each of the individual texts. Defaults to an empty string ('').

    Returns
    -------
    str :
        Returns a string containing a string representation of all the array elements in the same order,
        with the separator between each element.

    """
    if data is None:
        return None

    data = [str(elem).lower() if type(elem) != str else elem for elem in data]
    separator = str(separator)

    return separator.join(data)
