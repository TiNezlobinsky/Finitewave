
#
# Use the Period2DTracker to measure wave period (e.g spiral wave).
#

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave2D.tracker.period_2d_tracker import Period2DTracker

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
aliev_panfilov = AlievPanfilov2D()
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 300

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, n, 0, 100))
stim_sequence.add_stim(StimVoltageCoord2D(31, 1, 0, 100, 0, n))

tracker_sequence = TrackerSequence()
# add action potential tracker
# # add period tracker:
period_tracker = Period2DTracker()
# Here we create an int array of period detectors, where 1 means detector, 0 means no detector.
# First we create positions list (two coordinates for 2D), then use this list as indices
# for the detectors array.
detectors = np.zeros([n, n], dtype="uint8")
positions = np.array([[1,1], [5,5], [7,3], [9,1]])
detectors[positions[:, 0], positions[:, 1]] = 1
period_tracker.detectors = detectors
period_tracker.threshold = 0.5
tracker_sequence.add_tracker(period_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

print ("Periods:")
for key in period_tracker.output:
    print(key + ":", period_tracker.output[key][-1][1])
