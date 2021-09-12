
#
# Use the Animation2DTracker to make a folder with snapshots if model variable (voltage in this example)
# Then use the AnimationBuilder to create mp4 animation based on snapshots folder.
# Keep in mind: you have to install ffmpeg on your system.
#


import matplotlib.pyplot as plt
import numpy as np
import shutil

from finitewave.gpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.gpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.gpuwave2D.model.tp06_2d import TP062D
from finitewave.gpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D
from finitewave.cpuwave2D.fibrosis.structural_2d_pattern import Structural2DPattern

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.gpuwave2D.tracker.animation_2d_tracker import Animation2DTracker

from finitewave.tools.animation_builder import AnimationBuilder

from finitewave.gpuwave2D.state.tp06_state import TP06State

# number of nodes on the side
n = 256

tissue = CardiacTissue2D(size_i=n, size_j=n)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, n])

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# tissue.add_pattern(Structural2DPattern(1, n-1, 1, n-1, 0.2, 1, 1))

# create model object:
# model = AlievPanfilov2D()
model = TP062D()

# set up numerical parameters:
model.dt    = 0.02
model.dr    = 0.3
model.t_max = 1500
model.prog_bar = True

# set up stimulation parameters:
stim_sequence = StimSequence()
# stim_sequence.add_stim(StimVoltageCoord2D(0, -20, 0, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(600, -20, 0, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(1200, -20, 0, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(1800, -20, 0, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(2400, -20, 0, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(3000, -20, 0, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(3600, -20, 0, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(4200, -20, 0, 3, 1, n-1))
stim_sequence.add_stim(StimVoltageCoord2D(0, -20, 1, 3, 1, n-1))
# stim_sequence.add_stim(StimVoltageCoord2D(300, -20, 1, 3, 1, n-1))

tracker_sequence = TrackerSequence()
# add action potential tracker
animation_tracker = Animation2DTracker()
animation_tracker.target_array = "u"
# Folder name:
animation_tracker.dir_name = "anim_data"
animation_tracker.step = 1
tracker_sequence.add_tracker(animation_tracker)

model_state = TP06State()
model_state.record_load = "state"

# add the tissue and the stim parameters to the model object:
model.cardiac_tissue   = tissue
model.stim_sequence    = stim_sequence
model.tracker_sequence = tracker_sequence
model.state_keeper     = model_state

model.run()

# animation_builder = AnimationBuilder()
# animation_builder.vmin = -86
# animation_builder.vmax = 40
# animation_builder.dir_name = "anim_data"
# animation_builder.write_2d_mp4("animation.mp4")

# remove the snapshots folder:
# shutil.rmtree("anim_data")
