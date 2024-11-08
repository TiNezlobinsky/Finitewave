
"""
AlievPanfilov3D (Iso)
==========================

This example demonstrates how to use the Aliev-Panfilov model in 3D with
isotropic stencil.
"""

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
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1,
                                             n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5,
                                             n//2 - 5, n//2 + 5))

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# visualize the potential map in 3D
vis_mesh = tissue.mesh.copy()
vis_mesh[n//2:, n//2:, n//2:] = 0

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(vis_mesh)
grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
grid.plot(clim=[0, 1], cmap='viridis')
