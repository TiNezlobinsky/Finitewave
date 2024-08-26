
#
# Aniosotropic tissue (fibers at 45 degrees) with the Aliev-Panfilov model.
# Anisotropy is set by specifying a fiber array (CardiacTissue class) and
# diffusion coefficients D_al, D_ac (diffusion along and across fibers).
#

from finitewave.cpuwave3D.tissue.cardiac_tissue_3d import CardiacTissue3D
from finitewave.cpuwave3D.model.aliev_panfilov_3d import AlievPanfilov3D
from finitewave.cpuwave3D.stimulation.stim_voltage_coord_3d \
    import StimVoltageCoord3D
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import AsymmetricStencil3D
from finitewave.core.stimulation.stim_sequence import StimSequence

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 100

tissue = CardiacTissue3D((n, n, n), mode='aniso')
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, n])
tissue.add_boundaries()
# add fibers orientation vectors
theta, alpha = 0.25*np.pi, 0.1*np.pi/4
tissue.fibers = np.zeros((n, n, n, 3))
tissue.fibers[:, :, :, 0] = np.cos(theta) * np.cos(alpha)
tissue.fibers[:, :, :, 1] = np.cos(theta) * np.sin(alpha)
tissue.fibers[:, :, :, 2] = np.sin(theta)
# add numeric method stencil for weights computations
tissue.stencil = AsymmetricStencil3D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al/9

# create model object:
aliev_panfilov = AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord3D(0, 1, n//2 - 5, n//2 + 5,
                                          n//2 - 5, n//2 + 5,
                                          n//2 - 5, n//2 + 5))
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# show the potential map at the end of calculations:
fig, axs = plt.subplots(1, 3)
axs[0].imshow(aliev_panfilov.u[:, :, 50])
axs[1].imshow(aliev_panfilov.u[:, 50, :])
axs[2].imshow(aliev_panfilov.u[50, :, :])
plt.show()
