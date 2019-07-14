"""
Code that can be used for masking the data either by a separate layer of by a conditional expression.
This code works on different principle than the mask code described in
https://open-eo.github.io/openeo-api/v/0.4.0/processreference/ for mask function.
"""


import sys
import numpy as np


def mask(rasters, mask_type, expression = '', replacement = np.nan):
    """
    Applies a mask to the specified layers.

    For mask_type layer: The mask needs to be the last layer of the specified layers. The same mask
    is applied to all layers. It replaces all elements of the array that are non-zero (for numbers) or true
    (for boolean) in mask. These elements are replaced with the value specified for replacement, which defaults to
    np.nan value. Caution - the data type is converted to float in case, that the replacement is set to np.nan!

    For mask_type 'expression': applies a mask to the specified layers. All values that will match the boolean
    expression (such as all values lt 0) will be replaced by a replacement value. Caution - the data type is converted
    to float in case, that the replacement is set to np.nan!

    Parameters
    --------------
    rasters: input raster to be masked
    mask_type: 'layer' or 'expression'
    expression: expression in for of string to be used for masking. The values that will be true will be masked.
        Example: 'x<5'
    replacement: value to be used for replacement. Defaults to np.nan
    """

    # check if all the layers have the same dimension
    for raster in rasters[1:]:
        if not raster.shape == rasters[0].shape:
            err_message = 'Input rasters have different dimensions.'
            sys.exit(err_message)

    # Convert to 3d numpy array if input comes from gdal pixel function
    if (isinstance(rasters, (list, tuple))):
        rasters = np.asarray(rasters)

    # masking using last layer of the input rasters
    if mask_type in ['layer']:
        in_array = rasters[:-1]
        mask_array = rasters[-1]

        # converting the data to a float type in case that the replacement value is np.nan
        if replacement is np.nan:
            in_array = in_array.astype('float')

        # Sanity check
        if mask_array.shape[-2:] != in_array.shape[-2:]:
            err_message = 'The shape of each layer of the input array and mask array has to match'
            sys.exit(err_message)

        # masking true values in case of boolean mask and non-zero values in case of numeric mask
        if mask_array.dtype in ['bool']:
            for layer in in_array:
                layer[np.where(mask_array == True)] = replacement
        else:
            for layer in in_array:
                layer[np.where(mask_array != 0)] = replacement

    # masking using boolean expression
    elif mask_type in ['expression']:
        x = rasters

        # converting the data to a float type in case that the replacement value is np.nan
        if replacement is np.nan:
            x = x.astype('float')



        # run the expression, return error if not valid
        try:
            #expression = 'mask_array = ' + expression
            mask_array=eval(expression)
            in_array = x
            in_array[np.where(mask_array == True)] = replacement
        except:
            err_message = 'The expression can not be run.'
            sys.exit(err_message)

    return in_array




