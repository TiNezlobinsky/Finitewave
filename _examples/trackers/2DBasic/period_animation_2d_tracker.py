"""
# Period Animation in 2D

This example demonstrates how to use the PeriodAnimation2DTracker to track the
period of the spiral wave over time.
"""


import matplotlib.pyplot as plt
import numpy as np
import shutil

import finitewave as fw

# number of nodes on the side
n = 200

# create tissue object:
tissue = fw.CardiacTissue2D([n, n])
tissue.mesh = np.ones([n, n], dtype="uint8")
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 120

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 100))
stim_sequence.add_stim(fw.StimVoltageCoord2D(31, 1, 0, 100, 0, n))

tracker_sequence = fw.TrackerSequence()
period_map_tracker = fw.PeriodAnimation2DTracker()
period_map_tracker.dir_name = "period_map"
period_map_tracker.threshold = 0.3
period_map_tracker.step = 100
tracker_sequence.add_tracker(period_map_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

period_map_tracker.write(clim=[0, 120], shape_scale=10, fps=10, cmap="jet")
