
def filter_bbox(eo_obj, bbox):
    '''
    Filters the spatial extent of the given raster cube 'eo_obj'.
    :param eo_obj: eoObject,
        raster/data cube
    :param bbox: BoundingBox
        bounding box class for defining the extent and its spatial reference
    :return:
    '''
    extent = (bbox.west, bbox.south, bbox.east, bbox.north)
    eo_obj.crop(extent, crs=bbox.crs)