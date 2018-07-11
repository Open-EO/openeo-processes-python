def multiply(in_array, factor=1):
    """
    in_array is a tuple of 2d numpy arrays if the function is called from a pixel function in gdal
    """

    # Convert to 2d numpy array if input comes from gdal pixel function
    if (isinstance(in_array, (list, tuple))) and (len(in_array) == 1):
        in_array = in_array[0]

    out_array = in_array * factor

    return out_array
