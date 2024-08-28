
#
# Use the Animation2DTracker to make a folder with snapshots if model variable (voltage in this example)
# Then use the AnimationBuilder to create mp4 animation based on snapshots folder.
# Keep in mind: you have to install ffmpeg on your system.
#

import matplotlib.pyplot as plt
import numpy as np
import shutil

import finitewave as fw


# number of nodes on the side
n = 100
nj = 100
nk = 10

tissue = fw.CardiacTissue3D([n, nj, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, nj, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.C = np.ones([n, nj, nk])

# add fibers (oriented along X):
tissue.fibers = np.zeros([n, nj, nk, 3])
tissue.fibers[:,:,0] = 1
tissue.fibers[:,:,1] = 0
tissue.fibers[:,:,2] = 0

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()

# set up numerical parameters:
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 50

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, 10, 0, nj, 0, nk))

tracker_sequence = fw.TrackerSequence()
animation_tracker = fw.AnimationSlice3DTracker()
animation_tracker.target_model = aliev_panfilov
# We want to write the animation for the voltage variable. Use string value
# to specify the required array.anim_data
animation_tracker.target_array = "u"
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 1
animation_tracker.slice_n = 5
tracker_sequence.add_tracker(animation_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

animation_builder = fw.AnimationBuilder()
animation_builder.dir_name = "anim_data"
animation_builder.write_2d_mp4("animation.mp4")

shutil.rmtree("anim_data")
