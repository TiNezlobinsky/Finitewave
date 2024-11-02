
"""
AlievPanfilov2D (Iso)
==========================

This example demonstrates how to use the Aliev-Panfilov model in 2D with
isotropic stencil.
"""

import matplotlib.pyplot as plt
import numpy as np
import finitewave as fw

# create a mesh of cardiomyocytes (elems = 1):
n = 400
tissue = fw.CardiacTissue2D([n, n])
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()

# add IsotropicStencil for weights computations (default)
tissue.stencil = fw.IsotropicStencil2D()

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

# run the model:
aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.imshow(aliev_panfilov.u)
plt.show()
