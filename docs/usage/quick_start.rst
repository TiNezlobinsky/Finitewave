
Quick start
===================
This example demonstrates how to set up a simple 2D Aliev-Panfilov model


Make imports

.. code-block:: Python

    import finitewave as fw
    import numpy as np
    import matplotlib.pyplot as plt


Initialize a 100x100 mesh with all nodes set to 1 (healthy cardiac tissue).
Add empty nodes (0) at the mesh edges to simulate boundaries.

.. code-block:: Python

    n = 100
    tissue = fw.CardiacTissue2D([n, n])
    tissue.mesh = np.ones([n, n])
    tissue.add_boundaries()


Set up Aliev-Panfilov model to perform simulation

.. code-block:: Python

    aliev_panfilov = fw.AlievPanfilov2D()
    aliev_panfilov.dt = 0.01  # time step
    aliev_panfilov.dr = 0.25  # space step
    aliev_panfilov.t_max = 10  # simulation time


Set up stimulation parameters (activation from a line of nodes in the mesh)

.. code-block:: Python

    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0,
                                                 volt_value=1,
                                                 x1=1,
                                                 x2=n-1,
                                                 y1=1,
                                                 y2=3))


Assign the tissue and stimulation parameters to the model

.. code-block:: Python

    aliev_panfilov.cardiac_tissue = tissue
    aliev_panfilov.stim_sequence = stim_sequence


Run the simulation and show the output

.. code-block:: Python

    aliev_panfilov.run()

    plt.imshow(aliev_panfilov.u)
    plt.title("Aliev-Panfilov 2D model")
    plt.colorbar(label='u')
    plt.show()


The output should look like this:

.. image-sg:: /usage/images/quick_start_001.png
  :alt: Aliev-Panfilov 2D model
  :srcset: /usage/images/quick_start_001.png
  :class: sphx-glr-single-img
