
#
# Use the Animation3DTracker to create a snapshot dir with the model variables.
# The write method of the tracker will call the Animation3DBuilder to create
# the animation.
#

import math
import numpy as np
import matplotlib.pyplot as plt

import finitewave as fw

# number of nodes on the side
n = 200
nj = 200
nk = 10

tissue = fw.CardiacTissue3D([n, nj, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, nj, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, nj, nk])

# add fibers (oriented along X):
tissue.fibers = np.zeros([n, nj, nk, 3])
tissue.fibers[:, :, 0] = 1
tissue.fibers[:, :, 1] = 0
tissue.fibers[:, :, 2] = 0

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 150

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, 100, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, 100, 0, n, 0, nk))

# set up animation tracker:
animation_tracker = fw.Animation3DTracker()
animation_tracker.step = 3
animation_tracker.start = 50
animation_tracker.target_array = "u"
# add the tracker to the model:
tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(animation_tracker)
# add the sequence to the model:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence
aliev_panfilov.run()
# write the animation:
animation_tracker.write(format='mp4', framerate=5, quality=9,
                        clear=True)
