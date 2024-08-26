
#
# Sometimes you need to add nonstandard actions in your calculations. To do this
# use the Command and CommandSequence classes.
# Every command must be initialized with execute() method in the Command-inherited class (Command class).
# This method must have only one argument - model, that gives an access to its fields and methods.
# To use this command first check the model implementation and define which parameters you are going
# to modify (and in what time).
#
# In this example we are going to interrupt calculations when the propagation wave reaches the opposite side.
#

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.command.command_sequence import CommandSequence
from finitewave.core.command.command import Command

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 300

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
aliev_panfilov.t_max = 1000

# Define the command:
class InterruptCommand(Command):
    def execute(self, model):
        if np.any(model.u[:, 298] > 0.5):
             # increase the calculation step to stop the execution loop.
             model.step = np.inf

# We want to check the opposite side every 10 time units.
# Thus we have a list of commands with the same method but different times.
command_sequence = CommandSequence()
for i in range(0, 200, 10):
    command_sequence.add_command(InterruptCommand(i))

aliev_panfilov.command_sequence = command_sequence

# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, n, 0, 5))

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence  = stim_sequence

aliev_panfilov.run()
