
#
# Here we use the ActionPotential2DTracker to plot a voltage variable graph for the cell 30, 30.
#

from finitewave.gpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.gpuwave2D.model.tp06_2d import TP062D
from finitewave.gpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.gpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.gpuwave2D.tracker.action_potential_2d_tracker import ActionPotential2DTracker

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 128

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
model = AlievPanfilov2D()
# model = TP062D()

# set up numerical parameters:
model.dt    = 0.01
model.dr    = 0.25
model.t_max = 300
model.Di = 1
model.Dj = 1
model.prog_bar = True

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, 3, 0, n))

tracker_sequence = TrackerSequence()
# add action potential tracker
act_pot_tracker = ActionPotential2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
act_pot_tracker.cell_ind = [30, 30]
tracker_sequence.add_tracker(act_pot_tracker)

# add the tissue and the stim parameters to the model object:
model.cardiac_tissue   = tissue
model.stim_sequence    = stim_sequence
model.tracker_sequence = tracker_sequence


model.run()


plt.plot(np.arange(len(act_pot_tracker.output))*model.dt, act_pot_tracker.output)
plt.show()
