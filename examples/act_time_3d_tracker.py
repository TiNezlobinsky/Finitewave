"""
Track the action time in a 3D mesh.
"""

import finitewave as fw
import numpy as np

# number of nodes on the side
n = 100
nj = 100
nk = 10

tissue = fw.CardiacTissue3D([n, nj, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, nj, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, nj, nk])

# add fibers (oriented along X):
tissue.fibers = np.zeros([n, nj, nk, 3])
tissue.fibers[:, :, 0] = 1.0
tissue.fibers[:, :, 1] = 0.0
tissue.fibers[:, :, 2] = 0.0

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 60

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 3, 0, nj, 0, nk))

tracker_sequence = fw.TrackerSequence()
act_time_tracker = fw.ActivationTime3DTracker()
act_time_tracker.target_model = aliev_panfilov
act_time_tracker.threshold = 0.5
tracker_sequence.add_tracker(act_time_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

mesh_builder = fw.VisMeshBuilder3D()
grid = mesh_builder.build_mesh(tissue.mesh)
grid = mesh_builder.add_scalar(act_time_tracker.act_t, name="Activation Time")
grid.plot(cmap="viridis")
