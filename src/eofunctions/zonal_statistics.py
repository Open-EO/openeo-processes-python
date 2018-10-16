"""
A module docstring.
"""


import numpy as np

def zonal_statistics(raster, stat_type):
    """
    Calculates the (min, max, mean, median) value (excluding NaNs) in a raster.
    """

    if stat_type == 'min':
        out_value = np.nanmin(raster)
    elif stat_type == 'max':
        out_value = np.nanmax(raster)
    elif stat_type == 'mean':
        out_value = np.nanmean(raster)
    elif stat_type == 'median':
        out_value = np.nanmedian(raster)

    return out_value
