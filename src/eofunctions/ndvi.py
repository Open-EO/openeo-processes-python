"""
A module doctring.
"""


import numpy as np
from osgeo import gdal


def ndvi(datasets, from_file='OFF'):
    """
    Given data from the NIR and red bands, calculate the Normalized Diffrence Vegetation Index.

    datasets -- A list of two filenames or 2d numpy arrays as follows [NIR_data, red_data]
    from_file -- If 'datasets' contains filenames, must be set to 'ON'
    """

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    if from_file == 'ON':
        ds0 = gdal.Open(datasets[0])
        nir = ds0.ReadAsAraay()
        ds1 = gdal.Open(datasets[1])
        red = ds1.ReadAsAraay()
    else:
        nir = datasets[0]
        red = datasets[1]

    numerator = (nir - red)
    denominator = (nir + red)
    denominator[denominator == 0] = np.nan
    ndvi_output = numerator / denominator

    return ndvi_output
