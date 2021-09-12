import matplotlib.pyplot as plt
import numpy as np
import shutil

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave2D.tracker.period_map_2d_tracker import PeriodMap2DTracker
from finitewave.tools.animation_builder import AnimationBuilder


# number of nodes on the side
n = 200

tissue = CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add fibers (oriented along X):
tissue.fibers = np.zeros([n, n, 2])
tissue.fibers[:,:,0] = 0.
tissue.fibers[:,:,1] = 1.

# create model object:
aliev_panfilov = AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 120

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, n, 0, 100))
stim_sequence.add_stim(StimVoltageCoord2D(31, 1, 0, 100, 0, n))

tracker_sequence = TrackerSequence()
period_map_tracker = PeriodMap2DTracker()
period_map_tracker.dir_name = "period_map"
period_map_tracker.threshold = 0.3
period_map_tracker.step = 1
tracker_sequence.add_tracker(period_map_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue   = tissue
aliev_panfilov.stim_sequence    = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

animation_builder = AnimationBuilder()
animation_builder.dir_name = "period_map"
animation_builder.vmin = 15
animation_builder.vmax = 26
animation_builder.write_2d_mp4("period_map.mp4")

shutil.rmtree("period_map")
