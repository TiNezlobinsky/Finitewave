
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
act_time_tracker = fw.LocalActivationTime2DTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 10
act_time_tracker.start_time = 100
act_time_tracker.end_time = 200
tracker_sequence.add_tracker(act_time_tracker)

# connect model with tissue, stim and tracker:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

# run the simulation:
aliev_panfilov.run()

# plot the activation time map:
time_bases = [150, 170]  # time bases to plot the activation time map
lats = act_time_tracker.output
print(f'Number of LATs: {len(act_time_tracker.output)}')

X, Y = np.mgrid[0:n:1, 0:n:1]

fig, axs = plt.subplots(ncols=len(time_bases), figsize=(15, 5))

if len(time_bases) == 1:
    axs = [axs]

for i, ax in enumerate(axs):
    # Select the activation times next closest to the time base
    mask = np.any(lats >= time_bases[i], axis=0)
    ids = np.argmax(lats >= time_bases[i], axis=0)
    ids = tuple((ids[mask], *np.where(mask)))

    act_time = np.full([n, n], np.nan)
    act_time[mask] = lats[ids]

    act_time_min = time_bases[i]
    act_time_max = time_bases[i] + 30

    ax.imshow(act_time,
              vmin=act_time_min,
              vmax=act_time_max,
              cmap='viridis')
    ax.set_title(f'Activation time: {time_bases[i]} ms')
    cbar = fig.colorbar(ax.images[0], ax=ax, orientation='vertical')
    cbar.set_label('Activation Time (ms)')
plt.show()
