
"""
Aliev-Panfilov (Conductivity)
=============================

This example demonstrates how to use Aliev-Panfilov model in 2D with reduced
conductivity.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw


# number of nodes on the side
n = 400

tissue = fw.CardiacTissue2D([n, n])
# add a conductivity array, all elements = 1.0 -> normal conductvity:
tissue.conductivity = np.ones([n, n])
tissue.conductivity[:, n//3: 2 * n//3] = 0.6
tissue.conductivity[:, 2 * n//3:] = 0.3

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 1, 5, 1, n-1))

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.001
aliev_panfilov.dr = 0.1
aliev_panfilov.t_max = 20
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.imshow(aliev_panfilov.u)
plt.show()
