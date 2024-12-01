
"""
Spiral wave core in 2D
----------------------

This example demonstrates how to use the SpiralWaveCore2DTracker to track the
spiral wave core.
"""

import matplotlib.pyplot as plt

import finitewave as fw

# set up the tissue:
n = 200
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 100))
stim_sequence.add_stim(fw.StimVoltageCoord2D(31, 1, 0, 100, 0, n))

# set up tracker parameters:
tracker_sequence = fw.TrackerSequence()
sw_core_tracker = fw.SpiralWaveCore2DTracker()
sw_core_tracker.threshold = 0.5
sw_core_tracker.start_time = 50
sw_core_tracker.step = 100  # Record the spiral wave core every 1 ms
tracker_sequence.add_tracker(sw_core_tracker)

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 300
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

sw_core = sw_core_tracker.output

# plot the spiral wave trajectory:
plt.imshow(aliev_panfilov.u, cmap='viridis', origin='lower')
plt.plot(sw_core['x'], sw_core['y'], 'r')
plt.title('Spiral Wave Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, n)
plt.ylim(0, n)

plt.show()
