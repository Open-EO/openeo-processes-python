===========
eoFunctions
===========


eoFunctions is a collection of geo-physical functions developed for Earth Observation satellite data.

Installation
============

1. Install miniconda and clone repository:
------------------------------------------

::

  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  git clone https://git.eodc.eu/eodc/eoDataReaders.git
  cd eoDataReaders

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

  source activate eofunctions
  python setup.py install
  python setup.py test
  
Change 'install' with 'develop' if you plan to further develop the package.

4. Run tests
--------------------------------------------------------

::

  source activate eofunctions
  python setup.py test


Note
====

This project has been set up using PyScaffold 3.0.3. For details and usage
information on PyScaffold see http://pyscaffold.org/.
