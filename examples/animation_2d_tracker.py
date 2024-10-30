"""
Use the Animation2DTracker to make a folder with snapshots if model variable (voltage in this example)
Then use the AnimationBuilder to create mp4 animation based on snapshots folder.
Keep in mind: you have to install ffmpeg on your system.
"""

import shutil

import finitewave as fw
import numpy as np

# number of nodes on the side
n = 100

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 50

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 5))

tracker_sequence = fw.TrackerSequence()
# add action potential tracker
animation_tracker = fw.Animation2DTracker()
# We want to write the animation for the voltage variable. Use string value
# to specify the required array.anim_data
animation_tracker.target_array = "u"
# Folder name:
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 1
tracker_sequence.add_tracker(animation_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

animation_builder = fw.AnimationBuilder()
animation_builder.dir_name = "anim_data"
animation_builder.write_2d_mp4("animation.mp4")

# remove the snapshots folder:
shutil.rmtree("anim_data")
