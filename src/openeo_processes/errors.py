class QuantilesParameterMissing(Exception):
    def __init__(self):
        self.message = "The process 'quantiles' requires either the 'probabilities' or 'q' parameter to be set."

    def __str__(self):
        return self.message


class QuantilesParameterConflict(Exception):
    def __init__(self):
        self.message = "The process 'quantiles' only allows that either the 'probabilities' or the 'q' parameter is set."

    def __str__(self):
        return self.message


class ArrayElementParameterMissing(Exception):
    def __init__(self):
        self.message = "The process 'array_element' requires either the 'index' or 'labels' parameter to be set."

    def __str__(self):
        return self.message


class ArrayElementParameterConflict(Exception):
    def __init__(self):
        self.message = "The process 'array_element' only allows that either the 'index' or the 'labels' parameter is " \
                       "set."

    def __str__(self):
        return self.message


class ArrayElementNotAvailable(Exception):
    def __init__(self):
        self.message = "The array has no element with the specified index or label."

    def __str__(self):
        return self.message


class GenericError(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message
