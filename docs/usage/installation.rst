.. _installation:

Installation
============

To install Finitewave, navigate to the root directory of the project and run:

.. code-block:: bash

    $ python -m build
    $ pip install dist/finitewave-<version>.whl


This will install Finitewave as a Python package on your system.

For development purposes, you can install the package in an editable mode,
which allows changes to be immediately reflected without reinstallation:

.. code-block:: bash

    $ pip install -e .

Requirements
------------

Finitewave requires the following dependencies:

+-----------------+---------+--------------------------------------------------+
| Dependency      | Version | Link                                             |
+=================+=========+==================================================+
| ffmpeg-python   | 0.2.0   | https://pypi.org/project/ffmpeg-python/          |
+-----------------+---------+--------------------------------------------------+
| imageio-ffmpeg  | 0.4.5   | https://pypi.org/project/imageio-ffmpeg/         |
+-----------------+---------+--------------------------------------------------+
| matplotlib      | 3.9.2   | https://pypi.org/project/matplotlib/             |
+-----------------+---------+--------------------------------------------------+
| natsort         | 8.4.0   | https://pypi.org/project/natsort/                |
+-----------------+---------+--------------------------------------------------+
| numba           | 0.60.0  | https://pypi.org/project/numba/                  |
+-----------------+---------+--------------------------------------------------+
| numpy           | 1.26.4  | https://pypi.org/project/numpy/                  |
+-----------------+---------+--------------------------------------------------+
| pandas          | 2.2.3   | https://pypi.org/project/pandas/                 |
+-----------------+---------+--------------------------------------------------+
| pyvista         | 0.44.1  | https://pypi.org/project/pyvista/                |
+-----------------+---------+--------------------------------------------------+
| scikit-image    | 0.24.0  | https://pypi.org/project/scikit-image/           |
+-----------------+---------+--------------------------------------------------+
| scipy           | 1.14.1  | https://pypi.org/project/scipy/                  |
+-----------------+---------+--------------------------------------------------+
| setuptools      | 74.1.2  | https://pypi.org/project/setuptools/             |
+-----------------+---------+--------------------------------------------------+
| tqdm            | 4.66.5  | https://pypi.org/project/tqdm/                   |
+-----------------+---------+--------------------------------------------------+


You can install all the dependencies by running:

.. code-block:: bash

    $ pip install -r requirements.txt
