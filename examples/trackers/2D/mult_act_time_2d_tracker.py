
#
# Use the ActivationTime2DTracker to create an activation time map.
#

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

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
# aliev_panfilov = AlievPanfilov2D()
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 300

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, 3, 0, n))
stim_sequence.add_stim(fw.StimVoltageCoord2D(100, 1, 0, 3, 0, n))
stim_sequence.add_stim(fw.StimVoltageCoord2D(200, 1, 0, 3, 0, n))

tracker_sequence = fw.TrackerSequence()
# add action potential tracker
act_time_tracker = fw.MultiActivationTime2DTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 10
act_time_tracker.start_time = 100
act_time_tracker.end_time = 210
tracker_sequence.add_tracker(act_time_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

print(f'Number of LATs: {len(act_time_tracker.act_t)}')

X, Y = np.mgrid[0:n:1, 0:n:1]

fig, axs = plt.subplots(ncols=len(act_time_tracker.output), figsize=(15, 5))

if len(act_time_tracker.output) == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    act_time = act_time_tracker.output[i]
    act_time = np.where(act_time == -1, np.nan, act_time)
    act_time_min = np.nanmin(act_time)
    act_time_max = np.nanmax(act_time)
    levels = np.arange(act_time_min, act_time_max, 5)
    ax.imshow(act_time,
              vmin=act_time_min,
              vmax=act_time_max,
              cmap='viridis')
    CS = ax.contour(X, Y, np.transpose(act_time), levels,
                    colors='black', vmin=act_time_min, vmax=act_time_max)
    ax.clabel(CS, inline=True, fontsize=10)
plt.show()
