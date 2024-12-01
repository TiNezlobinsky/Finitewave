
#
# Use the ActivationTime2DTracker to create an activation time map.
#

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 400
alpha = 0.25 * np.pi
tissue = fw.CardiacTissue2D([n, n])
tissue.fibers = np.zeros(tissue.mesh.shape + (2,))
tissue.fibers[:, :, 0] = np.cos(alpha)
tissue.fibers[:, :, 1] = np.sin(alpha)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
# stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1,
#                                              n//2 - 5, n//2 + 5, 
#                                              n//2 - 5, n//2 + 5))
stim_sequence.add_stim(fw.StimCurrentCoord2D(0, 3, 0.18,
                                             n//2 - 10, n//2 + 10,
                                             n//2 - 10, n//2 + 10))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
act_time_tracker = fw.ActivationTime2DTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 100  # calculate activation time every 100 * dt
tracker_sequence.add_tracker(act_time_tracker)

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.001
aliev_panfilov.dr = 0.1
aliev_panfilov.t_max = 12
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the activation time map
X, Y = np.mgrid[0:n:1, 0:n:1]
act_time = act_time_tracker.output
act_time = np.where(act_time == -1, np.nan, act_time)
levels = np.arange(np.nanmin(act_time), np.nanmax(act_time), 5)

fig, ax = plt.subplots()
ax.imshow(act_time)
CS = ax.contour(X, Y, np.transpose(act_time), levels, colors='black')
ax.clabel(CS, inline=True, fontsize=10)
plt.show()

velocity_calc = fw.VelocityCalculation()
velocity = velocity_calc.major_minor_velocity(act_time_tracker.output,
                                              aliev_panfilov.dr)

print(velocity)

# print(np.mean(velocity))
# print(np.std(velocity))