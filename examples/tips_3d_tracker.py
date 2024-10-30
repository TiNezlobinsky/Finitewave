"""
Track the tip in a 3D mesh.
"""

import finitewave as fw
import matplotlib.pyplot as plt
import numpy as np

# number of nodes on the side
n = 200
nj = 200
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
tissue.fibers[:, :, 0] = 1
tissue.fibers[:, :, 1] = 0
tissue.fibers[:, :, 2] = 0

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 150

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, 100, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, 100, 0, n, 0, nk))

tracker_sequence = fw.TrackerSequence()
spiral_3d_tracker = fw.Spiral3DTracker()
tracker_sequence.add_tracker(spiral_3d_tracker)

aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

swcore = np.array(spiral_3d_tracker.swcore)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(swcore[:, 2], swcore[:, 3], swcore[:, 4])
plt.show()
