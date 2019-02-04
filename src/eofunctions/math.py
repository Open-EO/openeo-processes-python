import math
import numpy as np


def is_valid(x):
    if type(x) == float:
        return ~np.isnan(x) & ~np.isinf(x)
    else:
        return True
