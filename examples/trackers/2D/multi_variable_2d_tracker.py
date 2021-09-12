
#
# Here we use the ActionPotential2DTracker to plot a voltage variable graph for the cell 30, 30.
#

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave2D.tracker.multivariable_2d_tracker import MultiVariable2DTracker

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 100

tissue = CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
aliev_panfilov = AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 100

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, 3, 0, n))

tracker_sequence = TrackerSequence()
# add action potential tracker
multivariable_tracker = MultiVariable2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
multivariable_tracker.cell_ind = [30, 30]
multivariable_tracker.var_list = ["u", "v"]
tracker_sequence.add_tracker(multivariable_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence


aliev_panfilov.run()

time = np.arange(len(multivariable_tracker.vars["u"]))*aliev_panfilov.dt
plt.plot(time, multivariable_tracker.vars["u"])
plt.plot(time, multivariable_tracker.vars["v"])
plt.show()
