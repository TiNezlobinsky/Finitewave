Advanced
========

This section covers some advanced topics that you may find useful when using
finitewave.

Parallelization
---------------

Finitewave supports parallelization using the
`numba <https://numba.pydata.org/>`_. This allows you to run simulations on
multiple cores. To enable parallelization, you need to set the number of
threads to use in the environment variable ``NUMBA_NUM_THREADS``. For example,
to use 4 threads, you can run the following command:

.. code-block:: Python

    aliev_panfilov = AlievPanfilov2D()
    aliev_panfilov.number_of_threads = 4