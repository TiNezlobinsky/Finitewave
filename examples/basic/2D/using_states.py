
#
# In case of hard computations may find usefull dumping the model state to
# load it in next session.
# Use StateKeeper-inherited classes for each model to manage a model state
# Use record_save string to define a state-folder to save.
# Use record_load string to load a state folder.
#


from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.state.state_keeper import StateKeeper

import matplotlib.pyplot as plt
import numpy as np
import gc


# number of nodes on the side
n = 100

tissue = CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()
# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, n])
# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
aliev_panfilov = AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 5

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, n, 0, 3))

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence  = stim_sequence

# save the "state" dir with model variables:
model_state = StateKeeper()
model_state.record_save = "state"

aliev_panfilov.state_keeper = model_state

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.imshow(aliev_panfilov.u)
plt.show()

# We delete model and use gc.collect() to ask the virtual machine remove objects from memory.
# Though it's not necessary to do this.
del aliev_panfilov
gc.collect()

# # # # # # # # #

# Here we create a new model and load state from the previous calculation to continue.

# recreate the model
aliev_panfilov = AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt    = 0.01
aliev_panfilov.dr    = 0.25
aliev_panfilov.t_max = 4
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue

# load the state dir:
model_state = StateKeeper()
model_state.record_load = "state"

aliev_panfilov.state_keeper = model_state

aliev_panfilov.run()

plt.imshow(aliev_panfilov.u)
plt.show()
