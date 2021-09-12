
#
# In case of hard computations may find usefull dumping the model state to
# load it in next session.
# Use StateKeeper-inherited classes for each model to manage a model state
# Use record_save string to define a state-folder to save.
# Use record_load string to load a state folder.
#


from finitewave.gpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.gpuwave2D.model.tp06_2d import TP062D
from finitewave.gpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D
from finitewave.gpuwave2D.state.tp06_state import TP06State

from finitewave.gpuwave2D.tracker.simple_2d_tracker import Simple2DTracker

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

import matplotlib.pyplot as plt
import numpy as np
import gc


# number of nodes on the side
n = 128

tissue = CardiacTissue2D(size_i=n, size_j=n)
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, n])

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
model = TP062D()

# set up numerical parameters:
model.dt    = 0.01
model.dr    = 0.25
model.t_max = 185.61
model.Di = 1
model.Dj = 1

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, -20, 0, n, 0, 80))

# add the tissue and the stim parameters to the model object:
model.cardiac_tissue = tissue
model.stim_sequence  = stim_sequence

# save the "state" dir with model variables:
model_state = TP06State()
model_state.record_save = "state"

model.state_keeper = model_state

model.run()

# show the potential map at the end of calculations:
plt.imshow(model.u)
plt.show()

# We delete model and use gc.collect() to ask the virtual machine remove objects from memory.
# Though it's not necessary to do this.
del model
gc.collect()

# # # # # # # # #

# Here we create a new model and load state from the previous calculation to continue.

# recreate the model
model = TP062D()

# set up numerical parameters:
model.dt    = 0.01
model.dr    = 0.25
model.t_max = 150
model.Di = 1
model.Dj = 1
model.prog_bar = True

tracker_sequence = TrackerSequence()
# add simple tracker to get array from GPU device
simple_tracker = Simple2DTracker()
simple_tracker.target_array = "Cai"
tracker_sequence.add_tracker(simple_tracker)

# add the tissue and the stim parameters to the model object:
model.cardiac_tissue = tissue
model.tracker_sequence = tracker_sequence

# load the state dir:
model_state = TP06State()
model_state.record_load = "state"

model.state_keeper = model_state

model.run()

plt.imshow(model.Cai)
plt.show()
