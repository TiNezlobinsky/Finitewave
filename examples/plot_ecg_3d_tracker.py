"""
Use the Period3DTracker to measure wave period (e.g spiral wave).
"""

import finitewave as fw
import matplotlib.pyplot as plt
import numpy as np

# number of nodes on the side
n = 100
m = 10

tissue = fw.CardiacTissue3D([n, n, m])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, m], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

theta, alpha = 0.25 * np.pi, 0.1 * np.pi / 4
tissue.fibers = np.zeros((n, n, m, 3))
tissue.fibers[:, :, :, 0] = np.cos(theta) * np.cos(alpha)
tissue.fibers[:, :, :, 1] = np.cos(theta) * np.sin(alpha)
tissue.fibers[:, :, :, 2] = np.sin(theta)
# add numeric method stencil for weights computations
tissue.stencil = fw.AsymmetricStencil3D()
tissue.D_al = 1
tissue.D_ac = tissue.D_al / 9

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.0015
aliev_panfilov.dr = 0.1
aliev_panfilov.t_max = 30

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 1, 5, 1, n - 1, 1, m - 1))

tracker_sequence = fw.TrackerSequence()
ecg_tracker = fw.ECG3DTracker()
ecg_tracker.measure_coords = np.array(
    [[n // 2, n // 2, m + 3], [n // 4, n // 2, m + 3], [3 * n // 4, 3 * n // 4, m + 3]]
)
tracker_sequence.add_tracker(ecg_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()


plt.figure()
for y in ecg_tracker.ecg:
    plt.plot(np.arange(y.shape[0]) * aliev_panfilov.dt * ecg_tracker.step, y)
plt.show()
