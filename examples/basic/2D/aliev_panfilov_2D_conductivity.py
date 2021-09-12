
#
# The basic example of running simple simuations with the Aliev-Panfilov model.
# The minimal set for every simulation script is:
# 1. CardiacTissue (set up the mesh*, fibers array and conductivity array**).
# 2. Create and set up the model (minimal set of parameters: dt, dr (spatial step), t_max).
# 3. Add stimulation parameters. Use Stim classes to initialize a stimulation area and
# them to the StimSequence class object.
# Stim examples:
# 1) StimVoltageCoord2D: [stim_time, voltage, x0, x1, y0, y1].
# 2) StimCurrentCoord2D: [stim_time, current, current_time, x0, x1, y0, y1].
# Use the run() method to start the simulation.
#
# (*) Every mesh must contain a border line of empty nodes (boundary).
# add_boundaries() method helps you to do it easily.
#
# (**) What is the conductivity array? Conductvity is an array of coefficients
# which will be multipied with the diffusion coeeficients at every time step.
# In most of calculations just use tissue.cond = np.ones([n, n]) which means
# no influence of the conductivity coefficients on your calculations.

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave2D.tracker.animation_2d_tracker import Animation2DTracker

from finitewave.tools.animation_builder import AnimationBuilder

import matplotlib.pyplot as plt
import numpy as np
import shutil


# number of nodes on the side
n = 400

tissue = CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.conductivity = np.ones([n, n])
tissue.conductivity[n//4 - n//10: n//4 + n//10,
                    n//4 - n//10: n//4 + n//10] = 0.3

# don't forget to add the fibers array even if you have an anisotropic tissue.
tissue.fibers = np.zeros([n, n, 2])

# create model object:
aliev_panfilov = AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30
# set up stimulation parameters:
stim_sequence = StimSequence()
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, n//2 - 3, n//2 + 3,
                                          n//2 - 3, n//2 + 3))
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.imshow(aliev_panfilov.u)
plt.show()
