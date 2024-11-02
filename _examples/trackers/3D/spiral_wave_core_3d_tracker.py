

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200
nk = 10

tissue = fw.CardiacTissue3D([n, n, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 150

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, n//2, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, n//2, 0, n, 0, nk))

tracker_sequence = fw.TrackerSequence()
spiral_3d_tracker = fw.SpiralWaveCore3DTracker()
spiral_3d_tracker.threshold = 0.5
spiral_3d_tracker.start_time = 40
spiral_3d_tracker.step = 100
tracker_sequence.add_tracker(spiral_3d_tracker)

aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

swcore = spiral_3d_tracker.output

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(swcore['x'], swcore['y'], swcore['z'], c=swcore['time'],
           cmap='plasma', s=30)
ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_zlim(0, nk)
plt.show()
