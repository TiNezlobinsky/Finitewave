
#
# The model is a 3D Aliev-Panfilov model with anisotropic stencil.
# The model is stimulated with a voltage pulse in the center of the tissue.
# Anisotropy is set by specifying a fiber array (CardiacTissue class) and
# diffusion coefficients D_al, D_ac (diffusion along and across fibers).
#

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100

tissue = fw.CardiacTissue3D((n, n, n))
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, n])
tissue.add_boundaries()
# add fibers orientation vectors
theta, alpha = 0. * np.pi, 0. * np.pi/4
tissue.fibers = np.zeros((3, n, n, n))
tissue.fibers[0] = np.cos(theta) * np.cos(alpha)
tissue.fibers[1] = np.cos(theta) * np.sin(alpha)
tissue.fibers[2] = np.sin(theta)
# add numeric method stencil for weights computations
tissue.stencil = fw.AsymmetricStencil3D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al/9

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 0.01
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5))
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()


# show the potential map in axial, coronal and sagittal planes:
fig, axs = plt.subplots(1, 3)
axs[0].imshow(aliev_panfilov.u[:, :, n//2])
axs[1].imshow(aliev_panfilov.u[:, n//2, :])
axs[2].imshow(aliev_panfilov.u[n//2, :, :])
axs[0].set_title('Axial')
axs[1].set_title('Coronal')
axs[2].set_title('Sagittal')
plt.show()

vis_mesh = tissue.mesh.copy()
vis_mesh[n//2:, n//2:, n//2:] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh)
grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
grid.plot(clim=[0, 1], cmap='viridis')
