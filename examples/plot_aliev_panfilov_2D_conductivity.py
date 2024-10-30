"""
The basic example of running simple simuations with the Aliev-Panfilov model.
The model is a 2D model with isotropic stencil.
The model is stimulated with a voltage pulse in the center of the tissue.
Conductivity is set to 0.3 in the center of the tissue - this will deform the wavefront at the top of the square due to the slow propagation.
"""

import finitewave as fw
import matplotlib.pyplot as plt
import numpy as np

# number of nodes on the side
n = 400

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.conductivity = np.ones([n, n])
tissue.conductivity[n // 4 - n // 10 : n // 4 + n // 10, n // 4 : n // 4 * 3] = 0.3

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(
    fw.StimVoltageCoord2D(0, 1, n // 2 - 3, n // 2 + 3, n // 2 - 3, n // 2 + 3)
)
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.imshow(aliev_panfilov.u)
plt.show()
