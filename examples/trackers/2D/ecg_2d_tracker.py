
#
# Use the Period2DTracker to measure wave period (e.g spiral wave).
#

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.model.tp06_2d import TP062D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave2D.tracker.period_2d_tracker import Period2DTracker
from finitewave.cpuwave2D.tracker.ecg_2d_tracker import ECG2DTracker

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
for model in [AlievPanfilov2D]:
    aliev_panfilov = model()
    aliev_panfilov.dt = 0.01
    aliev_panfilov.dr = 0.25
    aliev_panfilov.t_max = 100

    # set up stimulation parameters:
    stim_sequence = StimSequence()
    stim_sequence.add_stim(StimVoltageCoord2D(10, 1, 0, n, 0, 3))
    # stim_sequence.add_stim(StimVoltageCoord2D(31, 1, 0, 100, 0, n))

    tracker_sequence = TrackerSequence()
    ecg_tracker = ECG2DTracker()
    ecg_tracker.measure_points = np.array([[100, 100, 10]])
    tracker_sequence.add_tracker(ecg_tracker)

    # add the tissue and the stim parameters to the model object:
    aliev_panfilov.cardiac_tissue = tissue
    aliev_panfilov.stim_sequence = stim_sequence
    aliev_panfilov.tracker_sequence = tracker_sequence

    aliev_panfilov.run()

    plt.plot(ecg_tracker.ecg[0])
plt.show()
