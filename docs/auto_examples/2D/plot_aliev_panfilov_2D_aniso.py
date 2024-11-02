
"""
AlievPanfilov2D (Aniso)
==========================

This example demonstrates how to use the Aliev-Panfilov model in 2D with
anisotropic tissue.

Anisotropy is set by specifying a fiber array (``CardiacTissue`` class) and
diffusion coefficients ``D_al``, ``D_ac`` (diffusion along and across fibers).
Always use ``AsymmetricStencil`` for weights computations in case of
anisotropic tissue.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# create a mesh of cardiomyocytes (elems = 1):
n = 400
tissue = fw.CardiacTissue2D([n, n])
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()
# add fibers orientation vectors
tissue.fibers = np.zeros([n, n, 2])
tissue.fibers[:, :, 0] = np.cos(0.25 * np.pi)
tissue.fibers[:, :, 1] = np.sin(0.25 * np.pi)
# add numeric method stencil for weights computations
tissue.stencil = fw.AsymmetricStencil2D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al/9

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, n//2 - 3, n//2 + 3,
                                             n//2 - 3, n//2 + 3))

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.figure()
plt.imshow(aliev_panfilov.u)
plt.show()