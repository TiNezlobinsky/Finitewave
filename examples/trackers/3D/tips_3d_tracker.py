
from finitewave.cpuwave3D.tissue.cardiac_tissue_3d import CardiacTissue3D
from finitewave.cpuwave3D.model.aliev_panfilov_3d import AlievPanfilov3D
from finitewave.cpuwave3D.stimulation.stim_voltage_coord_3d import StimVoltageCoord3D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave3D.tracker.spiral_3d_tracker import Spiral3DTracker

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# number of nodes on the side
n = 200
nj = 200
nk = 10

tissue = CardiacTissue3D([n, nj, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, nj, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, nj, nk])

# add fibers (oriented along X):
tissue.fibers = np.zeros([n, nj, nk, 3])
tissue.fibers[:,:,0] = 1
tissue.fibers[:,:,1] = 0
tissue.fibers[:,:,2] = 0

# create model object:
aliev_panfilov = AlievPanfilov3D()
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 150

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord3D(0, 1, 0, n, 0, 100, 0, nk))
stim_sequence.add_stim(StimVoltageCoord3D(31, 1, 0, 100, 0, n, 0, nk))

tracker_sequence = TrackerSequence()
spiral_3d_tracker = Spiral3DTracker()
tracker_sequence.add_tracker(spiral_3d_tracker)

aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

swcore = np.array(spiral_3d_tracker.swcore)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(swcore[:,2], swcore[:,3], swcore[:,4])
plt.show()
