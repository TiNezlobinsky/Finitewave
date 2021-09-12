
#
# The basic example of running simple simuations with the Aliev-Panfilov model.
# The minimal set for every simulation script is:
# 1. CardiacTissue (set up the mesh*, fibers array, stencil**,
#    conductivity array***).
# 2. Create and set up the model
#    (minimal set of parameters: dt, dr (spatial step), t_max).
# 3. Add stimulation parameters. Use Stim classes to initialize a stimulation
# area and them to the StimSequence class object.
# Stim examples:
# 1) StimVoltageCoord2D: [stim_time, voltage, x0, x1, y0, y1].
# 2) StimCurrentCoord2D: [stim_time, current, current_time, x0, x1, y0, y1].
# Use initialize to set up tissue, stimuls, trackers etc.
# Use the run() method to start the simulation or proceed simuations
# with new t_max
#
# (*) Every mesh must contain a border line of empty nodes (boundary).
#     add_boundaries() method helps you to do it easily.
#
# (**) Stencil class that calculates weights for the divergence kernels.
#      There are 2 types of stencils: 9 point (aniso), 5 point (ortho, iso).
#      First, general stencil for kernel with 9 nearest points. It can
#      compute weights for all cases. But for better performance in case of
#      Orthotropic (diffusion coefficients differs along different axis) and
#      Isotropic diffusion you can set regime to 'ortho' and 'iso'. It will
#      use 5 points stencil. You should set fibers for Orthotropic and
#      Anisotropic cases.
#
# (***) Conductvity is an array of coefficients (default: 1) which helps to
#       model fibrosis. Conductvity is multiplied to diffusion coefficients.

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d \
    import StimVoltageCoord2D
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import IsotropicStencil2D
from finitewave.core.stimulation.stim_sequence import StimSequence

import matplotlib.pyplot as plt
import numpy as np


# number of nodes on the side
n = 400

tissue = CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()
# add numeric method stencil for weights computations
# IsotropicStencil is default stencil and will be ised if no stencil was specified
tissue.stencil = IsotropicStencil2D()

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
