
#
# This example demonstrates how to use the ECG2DTracker to track bi-EGMs
#

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 256

measure_points = np.array([[n//2, n//2, 2],
                           [n//4, n//4, 2]])

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.3
aliev_panfilov.t_max = 200

# induce spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, volt_value=1, x1=0, x2=n,
                                             y1=0, y2=5))
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=50, volt_value=1, x1=n//2,
                                             x2=n, y1=0, y2=n))

# set up the tracker:
tracker_sequence = fw.TrackerSequence()
egm_tracker = fw.ECG2DTracker()
egm_tracker.measure_points = measure_points
egm_tracker.step = 10
egm_tracker.start_time = 100
egm_tracker.end_time = 200
tracker_sequence.add_tracker(egm_tracker)

# connect model with tissue, stim and tracker:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

# run the simulation:
aliev_panfilov.run()

# plot EGMs:
colors = plt.get_cmap('tab10').colors[:len(measure_points)]

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

axs[0].imshow(aliev_panfilov.u, cmap='jet')
axs[0].scatter(measure_points[:, 0], measure_points[:, 1], c=colors,
               edgecolors='black', s=100)
axs[0].set_title('Transmembrane potential')

for i, egm in enumerate(egm_tracker.output.T):
    t = np.arange(egm_tracker.start_time, egm_tracker.end_time,
                  egm_tracker.step * aliev_panfilov.dt).astype(np.float64)
    axs[1].plot(t[:-1], np.diff(egm), label=f'EGM {i}', color=colors[i])

axs[1].set_title('bi-EGMs')
plt.legend()
plt.show()
