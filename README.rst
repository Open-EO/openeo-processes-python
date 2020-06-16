===========
openeo_processes
===========


openeo_processes is a Python representation of the openEO processes.

The list supported here tries to be as complete as possible, but some processes (typically the 'cube' ones) are intrinsically connected to the back-end implementation and data model, and therefore are omitted here. Examples of missing processes are 'load_collection' or 'merge_cubes'.

Processes are currently aligned with openEO API version 1.0.

Installation
============

1. Install miniconda and clone repository:
------------------------------------------

::

  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  git clone https://github.com/Open-EO/openeo-processes-python.git
  cd openeo-processes-python

  This script adds ``$HOME/miniconda/bin`` temporarily to the ``PATH`` to do this
  permanently add ``export PATH="$HOME/miniconda/bin:$PATH"`` to your ``.bashrc``
  or ``.zshrc``

2. Create the conda environment
-------------------------------

::

  conda env create -f conda_environment.yml
  
3. Install package in the conda environment
--------------------------------------------------------

::

  source activate openeo_processes
  python setup.py install
  python setup.py test
  
Change 'install' with 'develop' if you plan to further develop the package.

4. Run tests
--------------------------------------------------------

::

  source activate openeo_processes
  python setup.py test


Note
====

This project has been set up using PyScaffold 3.0.3. For details and usage
information on PyScaffold see http://pyscaffold.org/.
