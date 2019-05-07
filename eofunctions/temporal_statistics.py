"""
Computes simple temporal statistics (mean, min and max).
"""


import sys
import numpy as np


def temporal_statistics(raster_list, stat_type, ignore_nodata = True):
    """
    Calculates the (min, max, mean) value in a sequence of rasters.
    Rasters must have the same dimensions.
    """

    # Sanity check
    for raster in raster_list[1:]:
        if not raster.shape == raster_list[0].shape:
            err_message = 'Input rasters have different dimensions.'
            sys.exit(err_message)

    if stat_type == 'min':
        if ignore_nodata == True:
            np.nanmin(raster_list, axis=0)
        elif ignore_nodata == False:
            out_value = np.ones(raster_list[0].shape) * 1e12
            for (k, _) in enumerate(np.arange(len(raster_list) - 1)):
                tmp_value = np.minimum(raster_list[k], raster_list[k+1])
                out_value = np.minimum(tmp_value, out_value)
    elif stat_type == 'max':
        if ignore_nodata == True:
            np.nanmin(raster_list, axis=0)
        elif ignore_nodata == False:
            out_value = np.ones(raster_list[0].shape) * (-1e12)
            for (k, _) in enumerate(np.arange(len(raster_list) - 1)):
                tmp_value = np.maximum(raster_list[k], raster_list[k+1])
                out_value = np.maximum(tmp_value, out_value)
    elif stat_type == 'mean':
        if ignore_nodata == True:
            out_value = np.nanmean(raster_list, axis=0)
        elif ignore_nodata == False:
            out_value = np.mean(raster_list, axis=0)

        # original code do not handle nan values in any way
        # --------------------------------------------------------
        # out_value = np.zeros(raster_list[0].shape)
        # for (k, _) in enumerate(np.arange(len(raster_list))):
        #    out_value = out_value + raster_list[k]
        # out_value = out_value / len(raster_list)
        # --------------------------------------------------------

    return out_value
