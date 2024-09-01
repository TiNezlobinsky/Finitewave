

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
nj = 100
nk = 10

tissue = fw.CardiacTissue3D([n, nj, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, nj, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add fibers (oriented along X):
tissue.fibers = np.zeros([n, nj, nk, 3])
tissue.fibers[:,:,:,0] = 1
tissue.fibers[:,:,:,1] = 0
tissue.fibers[:,:,:,2] = 0

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()

# set up numerical parameters:
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 50

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 3, 0, nj, 0, nk))

tracker_sequence = fw.TrackerSequence()
# add action potential tracker
act_pot_tracker = fw.ActionPotential3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
act_pot_tracker.cell_ind = [30, 30, 5]
tracker_sequence.add_tracker(act_pot_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

plt.plot(np.arange(len(act_pot_tracker.output)) * aliev_panfilov.dt,
         act_pot_tracker.output)
plt.show()
