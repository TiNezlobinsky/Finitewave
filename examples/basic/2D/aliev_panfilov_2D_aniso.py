
#
# Aniosotropic tissue (fibers at 45 degrees) with the Aliev-Panfilov model.
# Anisotropy is set by specifying a fiber array (CardiacTissue class) and
# diffusion coefficients D_al, D_ac (diffusion along and across fibers).
#

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d \
    import StimVoltageCoord2D
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import AsymmetricStencil2D
from finitewave.core.stimulation.stim_sequence import StimSequence

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 400

tissue = CardiacTissue2D([n, n], mode='aniso')
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()
tissue.mesh[n//4 - n//10: n//4 + n//10, n//4 - n//10: n//4 + n//10] = 2
# add fibers orientation vectors
tissue.fibers = np.zeros([n, n, 2])
tissue.fibers[:, :, 0] = np.cos(0.25 * np.pi)
tissue.fibers[:, :, 1] = np.sin(0.25 * np.pi)
# add numeric method stencil for weights computations
tissue.stencil = AsymmetricStencil2D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al/9

# create model object:
aliev_panfilov = AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30
# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, n//2 - 3, n//2 + 3,
                                          n//2 - 3, n//2 + 3))
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
# plt.figure()
plt.imshow(aliev_panfilov.u)
plt.show()
