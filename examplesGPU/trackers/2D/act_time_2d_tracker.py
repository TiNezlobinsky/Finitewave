
#
# Use the ActivationTime2DTracker to create an activation time map.
#

from finitewave.gpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.gpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.gpuwave2D.model.tp06_2d import TP062D
from finitewave.gpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.gpuwave2D.tracker.activation_time_2d_tracker import ActivationTime2DTracker

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 256

tissue = CardiacTissue2D(size_i=n, size_j=n)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, n])

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
model = TP062D()
# model = AlievPanfilov2D()

# set up numerical parameters:
model.dt    = 0.01
model.dr    = 0.25
model.t_max = 100
model.prog_bar = True
model.show_kernel_time = True

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, -20, 0, 3, 0, n))

tracker_sequence = TrackerSequence()
# add action potential tracker
act_time_tracker = ActivationTime2DTracker()
act_time_tracker.threshold = -40
tracker_sequence.add_tracker(act_time_tracker)

# add the tissue and the stim parameters to the model object:
model.cardiac_tissue   = tissue
model.stim_sequence    = stim_sequence
model.tracker_sequence = tracker_sequence

model.run()

X, Y = np.mgrid[0:n-2:1, 0:n-2:1]
levels = np.arange(0., 120, 10)

fig, ax = plt.subplots()
ax.imshow(act_time_tracker.act_t[1:-1, 1:-1])
CS = ax.contour(X, Y, np.transpose(act_time_tracker.act_t[1:-1, 1:-1]), colors='black')
ax.clabel(CS, inline=True, fontsize=10)
plt.show()
