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


class SummandMissing(Exception):
    def __init__(self):
        self.message = "Addition requires at least two numbers."

    def __str__(self):
        return self.message


class SubtrahendMissing(Exception):
    def __init__(self):
        self.message = "Subtraction requires at least two numbers (a minuend and one or more subtrahends)."

    def __str__(self):
        return self.message


class MultiplicandMissing(Exception):
    def __init__(self):
        self.message = "Multiplication requires at least two numbers."

    def __str__(self):
        return self.message


class DivisorMissing(Exception):
    def __init__(self):
        self.message = "Division requires at least two numbers (a dividend and one or more divisors)."

    def __str__(self):
        return self.message

