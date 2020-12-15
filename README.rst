================
openeo_processes
================


openeo_processes is a Python representation of the openEO processes.

The list supported here tries to be as complete as possible, but some processes (typically the 'cube' ones) are intrinsically connected to the back-end implementation and data model, and therefore are omitted here. Examples of missing processes are 'load_collection' or 'merge_cubes'.

Processes are currently aligned with openEO API version 1.0.

Installation
============

1. At the moment, this package is only installable from source.
   So start with cloning the repository::

        git clone https://github.com/Open-EO/openeo-processes-python.git
        cd openeo-processes-python

2. It is recommended to install this package in a virtual environment,
   e.g. by using ``venv`` (from the Python standard library), ``virtualenv``,
   a conda environment, ...
   For example, to create a *new* virtual environment using ``venv``
   (in a folder called ``.venv``) and to activate it::

        python3 -m venv .venv
        source .venv/bin/activate
        python -m pip install --user --upgrade pip

   (You might want to use a different bootstrap python executable
   instead of ``python3`` in this example.)

3.  Install the package in the virtual environment,
    preferably through ``pip`` of your virtual environment::

        pip install .

    If you plan to do development on the package itself,
    install it in "development" mode with::

        pip install -e .

    If plan to process xarray or dask arrays, you probably
    have the corresponding libraries already installed in your virtual env,
    and ``openeo_processes`` will handle them appropriately out of the box.
    You can however also explicitly pull these libraries in as "extra" dependencies
    when installing ``openeo_processes``.
    For example with one of the following install commands::

        pip install .[dask]
        pip install .[xarray]
        pip install .[dask,xarray]


4. Optionally run the tests::

        python setup.py test
  


Note
====

This project has been set up using PyScaffold 3.0.3. For details and usage
information on PyScaffold see http://pyscaffold.org/.
