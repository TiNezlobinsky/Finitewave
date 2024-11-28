Advanced
========

This section covers some advanced topics that you may find useful when using
finitewave.

Parallelization
---------------

Finitewave supports parallelization using the
`numba <https://numba.pydata.org/>`_. This allows you to run simulations on
multiple cores. By default, ALL cores are used. You can control the
number of threads by setting the environment variable ``NUMBA_NUM_THREADS``
for numba or using ``num_of_threads`` parameter of the ``run`` method.

.. note::

    ``num_of_threads`` can not exceed the ``NUMBA_NUM_THREADS``.

For example, to use 4 threads, you can run the following command:

.. code-block:: Python

    aliev_panfilov = AlievPanfilov2D()
    aliev_panfilov.run(num_of_threads=4)

Stencils
--------

The ``CardiacModel`` class uses the ``Stencil`` class to calculate the
weights for the divergence kernels. The stencil is a set of points used to
approximate the derivatives of the diffusion equation.

For example in 2D simulations, you can choose between a 9-point stencil
(anisotropic) or a 5-point stencil (orthotropic or isotropic). Each stencil
class provides both the method for calculating the ``weights`` and the
``diffusion_kernel``, which will be used by ``CardiacModel`` to evaluate the
diffusion term.

By default, the ``IsotropicStencil2D`` class is used for simulations without
fibers and ``AssymetricStencil2D`` for simulations with fibers, as it handles
anisotropic diffusion. 

.. note::

    If you explicitly provide ``IsotropicStencil2D`` and ``fibers``. The fibers
    will be ignored and isotropic conduction will be used.

The ``AssymetricStencil2D`` class has two parameter ``D_al`` and ``D_ac`` that
define the diffusion coefficients in the direction of the fibers and the
perpendicular direction. By default this values are set to ``1.0`` and ``1/9``.

.. note::

    The ``CardiacModel`` has additional diffusion coefficient ``D_model`` which
    is specific for each model. The fininal diffusion coefficient will be the
    multiplication of the ``D_model * D_al * conductivity`` in each node.


Fibrosis
--------

