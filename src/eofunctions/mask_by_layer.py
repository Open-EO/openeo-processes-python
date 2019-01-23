"""
Code that can be used for masking the data
"""


import sys
import numpy as np


def mask_by_layer(rasters, replacement = np.nan):
    """
    Applies a mask to the specified layers. The mask needs to be the last layer of the specified layers. The same mask
    is applied to all layers. It replaces all elements of the array that are non-zero (for numbers) or true
    (for boolean) in mask. These elements are replaced with the value specified for replacement, which defaults to
    np.nan value. Caution - the data type is converted to float in case, that the replacement is set to np.nan!
    """

    # Convert to 3d numpy array if input comes from gdal pixel function
    if (isinstance(in_array, (list, tuple))) and (len(in_array) == 1):
        in_array = in_array[0]

    in_array = rasters[0:-1]
    mask = rasters[-1]

    # Sanity check
    if mask.shape[-2:] != in_array.shape[-2:]:
        err_message = 'The shape of each layer of the input array and mask array has to match'
        sys.exit(err_message)

    if replacement is np.nan:
        in_array = in_array.astype('float')

    if mask.dtype in ['bool']:
        for layer in in_array:
            layer[np.where(mask == True)] = replacement
    elif mask.dtype in ['int16', 'int32', 'int64', 'int8', 'uint16', 'uint32', 'uint64', 'uint8', 'float64', 'float32']:
        for layer in in_array:
            layer[np.where(mask != 0)] = replacement
    else:
        err_message = 'unknown dtype of the mask array'
        sys.exit(err_message)

    return in_array




#rasters = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[True,False,False],[False,True,False],[False,False,True]]])
#masked = mask_by_layer(rasters)
#print(masked)