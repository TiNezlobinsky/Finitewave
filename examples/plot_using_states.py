"""
In cases of heavy computations, it may be useful to dump the model state
and load it in the next session.
Use classes that inherit from StateKeeper to manage the model state.
Use the record_save string to define the folder where the state will be saved.
Use the record_load string to load the state from a specified folder.
"""

import gc

import finitewave as fw
import matplotlib.pyplot as plt
import numpy as np

# number of nodes on the side
n = 100

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()
# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.cond = np.ones([n, n])

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 5

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 3))

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

# save the "state" dir with model variables:
model_state = fw.StateKeeper()
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
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 4
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue

# load the state dir:
model_state = fw.StateKeeper()
model_state.record_load = "state"

aliev_panfilov.state_keeper = model_state

aliev_panfilov.run()

plt.imshow(aliev_panfilov.u)
plt.show()
