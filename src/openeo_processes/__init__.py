# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'openeo_processes'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'

from openeo_processes.logic import *
from openeo_processes.arrays import *
from openeo_processes.comparison import *
from openeo_processes.math import *
from openeo_processes.texts import *
from openeo_processes.utils import get_process, has_process
