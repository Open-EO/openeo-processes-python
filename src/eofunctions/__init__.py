# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'eoFunctions'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'

from eofunctions.arrays import *
from eofunctions.comparison import *
from eofunctions.mask import *
from eofunctions.math import *
from eofunctions.indexes import *
from eofunctions.texts import *