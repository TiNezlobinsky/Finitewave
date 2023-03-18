
#
# Use the ActivationTime2DTracker to create an activation time map.
#

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave2D.tracker.multi_activation_time_2d_tracker import MultiActivationTime2DTracker

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 200

tissue = CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
# aliev_panfilov = AlievPanfilov2D()
aliev_panfilov = AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 300

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, 3, 0, n))
stim_sequence.add_stim(StimVoltageCoord2D(100, 1, 0, 3, 0, n))
stim_sequence.add_stim(StimVoltageCoord2D(200, 1, 0, 3, 0, n))

tracker_sequence = TrackerSequence()
# add action potential tracker
act_time_tracker = MultiActivationTime2DTracker()
act_time_tracker.threshold = 0.5
tracker_sequence.add_tracker(act_time_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

print (len(act_time_tracker.act_t))

#X, Y = np.mgrid[0:n-2:1, 0:n-2:1]
#levels = np.arange(0., 120, 10)

#fig, ax = plt.subplots()
#ax.imshow(act_time_tracker.act_t[1:-1, 1:-1])
#CS = ax.contour(X, Y, np.transpose(act_time_tracker.act_t[1:-1, 1:-1]), colors='black')
#ax.clabel(CS, inline=True, fontsize=10)
#plt.show()
