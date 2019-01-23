"""
Code that can be used for masking the data
"""


import sys
import numpy as np


def mask(in_array, mask, replacement = np.nan):
    """
    Applies a mask to an array. It replaces all elements of the array that are non-zero (for numbers) or true
    (for boolean) in mask. These elements are replaced with the value specified for replacement, which defaults to
    np.nan value. Caution - the data type is converted to float in case, that the replacement is set to np.nan!
    """

    # Convert to 2d numpy array if input comes from gdal pixel function
    if (isinstance(in_array, (list, tuple))) and (len(in_array) == 1):
        in_array = in_array[0]

    # Sanity check
    if mask.shape != in_array.shape:
        err_message = 'The shape of the input array and mask array has to match'
        sys.exit(err_message)

    if replacement is np.nan:
        in_array = in_array.astype('float')

    if mask.dtype in ['bool']:
        in_array[np.where(mask == True)] = replacement
    elif mask.dtype in ['int16', 'int32', 'int64', 'int8', 'uint16', 'uint32', 'uint64', 'uint8', 'float64', 'float32']:
        in_array[np.where(mask != 0)] = replacement
    else:
        err_message = 'unknown dtype of the mask array'
        sys.exit(err_message)

    return in_array


#in_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
#import filter as filter

#in_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
#masked = mask(in_array, filter.filter_numeric(in_array, 'lte', 5))


#print(masked)