"""
Code that can be used for filtering the data. Returns a boolean array where the values that satisfied the
given filter criteria are represented by True and those that did not by False.
"""


import sys
import numpy as np


def filter_numeric(in_array, filter_type, limits):
    """
    Filters the raster of data returning a boolean array with results.
    Filter types available: 'lt', 'lte', 'gt', 'gte' and 'between'
    Limits: value (fot lt, lte, gt and gte) or list of values (for between)
    """

    # Convert to 2d numpy array if input comes from gdal pixel function
    if (isinstance(in_array, (list, tuple))) and (len(in_array) == 1):
        in_array = in_array[0]

    # Sanity check
    if filter_type in ['lt', 'lte', 'gt', 'gte']:
        if not isinstance(limits, int) and not isinstance(limits, float):
            err_message = 'For filter type lt, lte, gt and gte, the limits have to be int or float'
            sys.exit(err_message)
    if filter_type in ['between']:
        if not isinstance(limits, list) or not len(limits) == 2:
            err_message = 'For filter type between, the limits have to be a list of 2 numbers (int or float)'
            sys.exit(err_message)

    if filter_type == 'lt':
        out_array = in_array < limits
    elif filter_type == 'lte':
        out_array = in_array <= limits
    elif filter_type == 'gt':
        out_array = in_array > limits
    elif filter_type == 'gte':
        out_array = in_array >= limits
    elif filter_type == 'between':
        out_array = (in_array >= limits[0]) & (in_array <= limits[1])
    return out_array

in_array = np.array([[1,2,3],[4,5,6],[7,8,9]])

out_array = filter_numeric(in_array, 'between', [5,8])
print(out_array)