
"""
ActivationTime2DTracker
========================

This example demonstrates how to track the activation times during the
simulation
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# create a mesh of cardiomyocytes (elems = 1):
n = 200
tissue = fw.CardiacTissue2D([n, n])
tissue.mesh = np.ones([n, n], dtype="uint8")
tissue.add_boundaries()

# create model object and set up parameters:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 50

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, volt_value=1,
                                             x1=0, x2=3, y1=0, y2=n))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
act_time_tracker = fw.ActivationTime2DTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 100  # calculate activation time every 100 steps
tracker_sequence.add_tracker(act_time_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the activation time map
X, Y = np.mgrid[0:n:1, 0:n:1]
act_time = act_time_tracker.output
act_time = np.where(act_time == -1, np.nan, act_time)
levels = np.arange(np.nanmin(act_time), np.nanmax(act_time), 5)

fig, ax = plt.subplots()
ax.imshow(act_time)
CS = ax.contour(X, Y, np.transpose(act_time), levels, colors='black')
ax.clabel(CS, inline=True, fontsize=10)
plt.show()
