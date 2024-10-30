"""
Left ventricle simlation with the Aliev-Panfilov model.
Mesh and fibers were taken from Niderer's data storage (https://zenodo.org/records/3890034)
Here we use matrix stimlation to simultaneusly stimulate ventricle from apex and base.
After the end of the simulation you will see two waves propagating from the apex and the base.
"""

from pathlib import Path

import finitewave as fw
import numpy as np

path = Path(__file__).parent

# Load mesh as cubic array
mesh = np.load(path.joinpath("data", "mesh.npy"))

tissue = fw.CardiacTissue3D(mesh.shape)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = mesh
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 15

# set up stimulation parameters:
stim_sequence = fw.StimSequence()

stim_array = np.zeros(mesh.shape)
# stim array for the apex stimulation
stim_array[:, :, :20] = 1

# stim array for the base stimulation
stim_array[:, :, -10:] = 1

# Note: you can select only existing (=1) mesh points by applying the mask
# mask = mesh == 1
# But the stimulation classes already do this for you.

stim_sequence.add_stim(fw.StimVoltageMatrix3D(0, 1, stim_array))

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
# initialize model: compute weights, add stimuls, trackers etc.
aliev_panfilov.run()

# show the potential map at the end of calculations

# visualize the ventricle in 3D
mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(tissue.mesh)
mesh_grid = mesh_builder.add_scalar(aliev_panfilov.u, "u")
mesh_grid.plot(clim=[0, 1], cmap="viridis")
