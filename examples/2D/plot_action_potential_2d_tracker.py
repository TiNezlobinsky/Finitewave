
"""
ActionPotential2DTracker
=========================

This example demonstrates how to track the action potential at the specified
cell indices.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# create a mesh of cardiomyocytes (elems = 1):
n = 100
m = 20
tissue = fw.CardiacTissue2D([m, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 3, 0, n))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential2DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
# eather list or list of lists can be used
action_pot_tracker.cell_ind = [[10, 30], [10, 40]]
action_pot_tracker.step = 100
tracker_sequence.add_tracker(action_pot_tracker)

# create model object and set up parameters:
# aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov = fw.TP062D()
aliev_panfilov.dt = 0.001
aliev_panfilov.dr = 0.1
aliev_panfilov.t_max = 300
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the action potential
time = np.arange(len(action_pot_tracker.output)) * aliev_panfilov.dt

plt.figure()
plt.plot(time, action_pot_tracker.output[:, 0], label="cell_30_30")
plt.plot(time, action_pot_tracker.output[:, 1], label="cell_40_40")
plt.legend(title='Aliev-Panfilov')
plt.show()
