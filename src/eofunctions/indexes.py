"""
This module contains the ndvi function, which calculates the
Normalized Difference Vegetation Index.
"""


import numpy as np
from osgeo import gdal


def eo_ndvi(rasters):
    """
    Given data from the red and NIR bands (as 2D numpy arrays), it calculates
    the Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    rasters : list
    A list of two 2D numpy arrays as follows [red_data, NIR_data]

    Returns
    -------
    ndvi_output : 2D numpy array
    NDVI
    """

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    red = rasters[0]
    nir = rasters[1]

    numerator = (nir - red)
    denominator = (nir + red)
    ndvi_output = numerator / denominator

    return ndvi_output



"""
This module contains the NDWI function, which calculates the
Normalized Difference Water Index.
"""


def eo_ndwi(rasters):
    """
    Given data from the Green and NIR bands (as 2D numpy arrays), it calculates
    the Normalized Difference Water Index (NDWI).

    Parameters
    ----------
    rasters : list
    A list of two 2D numpy arrays as follows [Green_data, NIR_data]

    Returns
    -------
    ndwi_output : 2D numpy array
    NDWI
    """

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    green = rasters[0]
    nir = rasters[1]

    numerator = (nir - green)
    denominator = (nir + green)
    denominator[denominator == 0] = 9999
    ndwi_output = numerator / denominator

    # TODO maybe use only nir band -> mask = nir < np.max(nir) * 11 / 100

    return ndwi_output
