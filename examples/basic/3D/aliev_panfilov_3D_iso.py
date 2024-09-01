#
# The model is a 3D Aliev-Panfilov model with isotropic stencil.
# The model is stimulated with a voltage pulse in the center of the tissue.
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
# add numeric method stencil for weights computations
tissue.stencil = fw.IsotropicStencil3D()

aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 7
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 45, 55, 45, 55, 45, 55))
# add the tissue and the stim parameters to the model object:

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

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

# visualize the potential map in 3D
vis_mesh = tissue.mesh.copy()
vis_mesh[n//2:, n//2:, n//2:] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh)
grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
grid.plot(clim=[0, 1], cmap='viridis')
