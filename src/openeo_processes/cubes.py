
from openeo_processes.utils import process


###############################################################################
# Load Collection Process
###############################################################################


@process
def load_collection():
    """
    Returns class instance of `LoadCollection`.
    For more details, please have a look at the implementations inside
    `LoadCollection`.

    Returns
    -------
    LoadCollection :
        Class instance implementing all 'load_collection' processes.

    """
    return LoadCollection()


class LoadCollection:
    """
    Class implementing all 'load_collection' processes.

    """

    @staticmethod
    def exec_odc(odc_cube, product: str, x: tuple, y: tuple, time: tuple,
                 dask_chunks: dict, crs: str = "EPSG:4326"):

        datacube = odc_cube.load(product=product, dask_chunks=dask_chunks,
                                 **{'x': x, 'y': y, 'time': time, 'crs': crs})
        # Convert to xr.DataArray
        # TODO: add conversion with multiple and custom dimensions
        datacube = datacube.to_array(dim='bands')

        return datacube
