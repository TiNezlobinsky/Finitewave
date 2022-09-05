#
# Aniosotropic tissue (fibers at 45 degrees) with the Aliev-Panfilov model.
# Anisotropy is set by specifying a fiber array (CardiacTissue class) and
# diffusion coefficients D_al, D_ac (diffusion along and across fibers).
#

from finitewave.cpuwave3D.tissue import CardiacTissue3D
from finitewave.cpuwave3D.model import AlievPanfilov3D
from finitewave.cpuwave3D.stimulation import StimVoltageCoord3D
from finitewave.cpuwave3D.stencil import IsotropicStencil3D
from finitewave.core.stimulation import StimSequence


import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 100

tissue = CardiacTissue3D((n, n, n))
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, n])
tissue.add_boundaries()
# add numeric method stencil for weights computations
tissue.stencil = IsotropicStencil3D()

aliev_panfilov = AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord3D(0, 1, 45, 55, 45, 55, 45, 55))
# add the tissue and the stim parameters to the model object:

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
fig, axs = plt.subplots(1, 3)
axs[0].imshow(aliev_panfilov.u[:, :, 50])
axs[1].imshow(aliev_panfilov.u[:, 50, :])
axs[2].imshow(aliev_panfilov.u[50, :, :])
plt.show()
