
"""
# Action potential in 2D

This example demonstrates how to use the ActionPotential2DTracker to track the
action potential at the specified cell indices.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 3, 0, n))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
action_pot_tracker = fw.ActionPotential2DTracker()
# measure action potential for cells 30, 30 and 40, 40
action_pot_tracker.cell_ind = [[30, 30], [40, 40]]
tracker_sequence.add_tracker(action_pot_tracker)

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the action potential
time = np.arange(len(action_pot_tracker.output)) * aliev_panfilov.dt
plt.plot(time, action_pot_tracker.output[:, 0], label="cell_30_30")
plt.plot(time, action_pot_tracker.output[:, 1], label="cell_40_40")
plt.legend(title='Aliev-Panfilov')
plt.show()
