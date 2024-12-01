
"""
# ECG in 2D

This example demonstrates how to use the ECG2DTracker to track the ECG in 2D.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

ecg = {}
for CardiacModel in [fw.TP062D]:
    model = CardiacModel()
    model.dt = 0.001
    model.dr = 0.1
    model.t_max = 50

    # set up stimulation parameters:
    stim_sequence = fw.StimSequence()
    stim_sequence.add_stim(fw.StimVoltageCoord2D(10, 1, 0, n, 0, 3))

    tracker_sequence = fw.TrackerSequence()
    ecg_tracker = fw.ECG2DTracker()
    ecg_tracker.measure_points = [n//2, n//2, 10]
    tracker_sequence.add_tracker(ecg_tracker)

    # add the tissue and the stim parameters to the model object:
    model.cardiac_tissue = tissue
    model.stim_sequence = stim_sequence
    model.tracker_sequence = tracker_sequence

    model.run()

    ecg[CardiacModel.__name__] = ecg_tracker.output

plt.figure()
for k, v in ecg.items():
    t = np.arange(len(v)) * model.dt
    plt.plot(t, v / v.max(), label=k)
plt.legend()
plt.show()
