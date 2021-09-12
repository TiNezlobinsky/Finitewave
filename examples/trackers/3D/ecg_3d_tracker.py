
#
# Use the Period3DTracker to measure wave period (e.g spiral wave).
#

from finitewave.cpuwave3D.tissue.cardiac_tissue_3d import CardiacTissue3D
from finitewave.cpuwave3D.model.aliev_panfilov_3d import AlievPanfilov3D
from finitewave.cpuwave3D.model.tp06_3d import TP063D
from finitewave.cpuwave3D.stimulation.stim_voltage_coord_3d import StimVoltageCoord3D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave3D.tracker.period_3d_tracker import Period3DTracker
from finitewave.cpuwave3D.tracker.ecg_3d_tracker import ECG3DTracker

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 200

tissue = CardiacTissue3D([n, n, 5])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, 5], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 5, 3])

# create model object:
aliev_panfilov = AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 120

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord3D(10, 1, 0, n, 0, 3, 0, n))
# stim_sequence.add_stim(StimVoltageCoord3D(31, 1, 0, 100, 0, n))

tracker_sequence = TrackerSequence()
ecg_tracker = ECG3DTracker()
ecg_tracker.measure_points = np.array([[100, 100, 10]])
tracker_sequence.add_tracker(ecg_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

plt.plot(ecg_tracker.ecg[0])
plt.show()
