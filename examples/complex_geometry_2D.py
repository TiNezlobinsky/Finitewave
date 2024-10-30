"""
Creates labyrinth-like geometry and runs the simulation.
Check the complex_geometry.mp4 to see the result.
Use 0 and 1 to create the mesh of cardiomyocytes and obstacles.
"""

import shutil

import finitewave as fw
import numpy as np

# number of nodes on the side
n = 300

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])
for i in range(0, 40, 5):
    if i % 10 == 0:
        tissue.mesh[10 * i : 10 * (i + 3), :250] = 0
    else:
        tissue.mesh[10 * i : 10 * (i + 3), 50:] = 0
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 200

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, int(n * 0.03), 0, n))

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
animation_builder.write_2d_mp4("complex_geometry.mp4")

# remove the snapshots folder:
shutil.rmtree("anim_data")
