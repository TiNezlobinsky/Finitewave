
"""
ECG in 2D
---------

This example demonstrates how to use the ECG2DTracker to track the ECG in 2D.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# set up the tissue:
n = 400
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 1, n-1, 1, 5))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
ecg_tracker = fw.ECG2DTracker(distance_power=2)
ecg_tracker.measure_points = [[n//2, n//4, 10],
                              [n//2, n//2, 10],
                              [n//2, 3*n//4, 10]]

ecg_tracker.step = 10
tracker_sequence.add_tracker(ecg_tracker)

ecg = {}
model = fw.AlievPanfilov2D()
model.dt = 0.001
model.dr = 0.1
model.t_max = 50
# add the tissue and the stim parameters to the model object:
model.cardiac_tissue = tissue
model.stim_sequence = stim_sequence
model.tracker_sequence = tracker_sequence

model.run()

ecg = ecg_tracker.output

# fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
# axs[0].imshow(model.u, cmap='viridis', origin='lower')
# axs[1].imshow(model.transmembrane_current, cmap='viridis', origin='lower')
# plt.show()

plt.figure()
t = np.arange(len(ecg)) * model.dt * ecg_tracker.step
plt.plot(t, ecg, label="ECG")
plt.legend()
plt.show()
