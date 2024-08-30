
#
# Left ventricle simlation with the Aliev-Panfilov model.
# Mesh and fibers were taken from Niderer's data storage (https://zenodo.org/records/3890034)
# Fibers were generated with Rule-based algorithm.
# Ventricle is stimulated from the apex.

# ! Run this script from /examples/basic/3D directory to load the mesh and fibers properly.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw


path = Path(__file__).parent

mesh = np.load(path.joinpath("data", "mesh_x3.npy"))

# Due to the limitation of github file size, the fibers are divided into 5 parts.
# Load each part and concatenate them to get the full fibers array.
num_parts = 5
parts = []
# Load each part and append to the list
for i in range(num_parts):
    filename = path.joinpath("data", f"fibers_x3_{i+1}.npy")
    part = np.load(filename)
    parts.append(part)

# Concatenate all parts along the Z-axis (axis=2)
fibers = np.concatenate(parts, axis=2)

tissue = fw.CardiacTissue3D(mesh.shape, mode='aniso')
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = mesh
tissue.add_boundaries()
# add fibers orientation vectors
tissue.fibers = fibers
# add numeric method stencil for weights computations
tissue.stencil = fw.AsymmetricStencil3D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al/9

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 40
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, mesh.shape[0],
                                             0, mesh.shape[0],
                                             0, 20))
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# show the potential map at the end of calculations:
fig, axs = plt.subplots(1, 3)


# Visualize ventricle in axial, coronal, and sagittal planes:
axs[0].imshow(aliev_panfilov.u[aliev_panfilov.u.shape[0] // 2, :, :], cmap='viridis')
axs[0].imshow(mesh[aliev_panfilov.u.shape[0] // 2, :, :], cmap='Reds', alpha=0.3) 

axs[1].imshow(aliev_panfilov.u[:, aliev_panfilov.u.shape[1] // 2, :], cmap='viridis')
axs[1].imshow(mesh[:, aliev_panfilov.u.shape[1] // 2, :], cmap='Reds', alpha=0.3)  

axs[2].imshow(aliev_panfilov.u[:, :, aliev_panfilov.u.shape[2] // 2], cmap='viridis')
axs[2].imshow(mesh[:, :, aliev_panfilov.u.shape[2] // 2], cmap='Reds', alpha=0.3) 

# Set titles
axs[0].set_title('Axial')
axs[1].set_title('Coronal')
axs[2].set_title('Sagittal')

plt.show()
