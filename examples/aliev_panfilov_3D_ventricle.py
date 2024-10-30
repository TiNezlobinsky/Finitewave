"""
Left ventricle simlation with the Aliev-Panfilov model.
Mesh and fibers were taken from Niderer's data storage (https://zenodo.org/records/3890034)
Fibers were generated with Rule-based algorithm.
Ventricle is stimulated from the apex.
"""

from pathlib import Path

import finitewave as fw
import numpy as np

path = Path(__file__).parent

# Load mesh as cubic array
mesh = np.load(path.joinpath("data", "mesh.npy"))

# Load fibers as list of 3D vectors (x, y, z)
fibers_list = np.load(path.joinpath("data", "fibers.npy"))
fibers = np.zeros(mesh.shape + (3,), dtype=float)
fibers[mesh > 0] = fibers_list

tissue = fw.CardiacTissue3D(mesh.shape)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = mesh
tissue.add_boundaries()
# add fibers orientation vectors
tissue.fibers = fibers
# add numeric method stencil for weights computations
tissue.stencil = fw.AsymmetricStencil3D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al / 9

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 40
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(
    fw.StimVoltageCoord3D(0, 1, 0, mesh.shape[0], 0, mesh.shape[0], 0, 20)
)
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
