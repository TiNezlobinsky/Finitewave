
#
# Aniosotropic tissue (fibers at 45 degrees) with the Aliev-Panfilov model.
# Anisotropy is set by specifying a fiber array (CardiacTissue class) and
# diffusion coefficients D_al, D_ac (diffusion along and across fibers).
#

from finitewave.cpuwave3D.tissue import CardiacTissue3D
from finitewave.cpuwave3D.model import AlievPanfilov3D
from finitewave.cpuwave3D.stimulation import StimVoltageCoord3D
from finitewave.cpuwave3D.stencil import AsymmetricStencil3D
from finitewave.core.stimulation import StimSequence

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n_i = 200
n_j = 200
n_k = 100

# Set fibrosis density [0 - 1]
fibrosis_density = 0.4
random_mesh = np.random.random((n_i, n_j, n_k))

tissue = CardiacTissue3D((n_i, n_j, n_k), mode='aniso')
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n_i, n_j, n_k])
tissue.mesh[random_mesh < fibrosis_density] = 2
tissue.add_boundaries()

phi_k = np.linspace(- np.pi / 3, np.pi / 2, n_k - 2)
# add fibers orientation vectors
tissue.fibers = np.zeros((n_i, n_j, n_k, 3))
for k, phi in enumerate(phi_k):
    tissue.fibers[:, :, k + 1, 0] = np.cos(phi)
    tissue.fibers[:, :, k + 1, 1] = np.sin(phi)
    tissue.fibers[:, :, k + 1, 2] = 0

# add numeric method stencil for weights computations
tissue.stencil = AsymmetricStencil3D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al / 9

# create model object:
aliev_panfilov = AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30
# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord3D(0, 1, n_i // 2 - 5, n_i // 2 + 5,
                                          n_j // 2 - 5, n_j // 2 + 5,
                                          0, n_k))
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# show the potential map at the end of calculations:
fig, axs = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)

slices = np.linspace(1, n_k - 2, 9).astype(int)

for i in range(3):
    for j in range(3):
        k = 3 * i + j
        axs[i, j].imshow(aliev_panfilov.u[:, :, slices[k]], origin='lower')
        axs[i, j].set_title('Phi = {:.0f}'.format(
            np.degrees(phi_k[slices[k] - 1])))
plt.tight_layout()
plt.show()
