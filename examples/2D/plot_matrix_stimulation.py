
"""
Matrix Stimulation
==================

This example demonstrates how to stimulate the tissue using a matrix of
stimulation areas.
"""


import matplotlib.pyplot as plt
from skimage import draw
import numpy as np

import finitewave as fw

# set up cardiac tissue:
n = 400
tissue = fw.CardiacTissue2D([n, n])

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_area = np.full([400, 400], False, dtype=bool)
ii, jj = draw.disk([100, 100], 5)
stim_area[ii, jj] = True
ii, jj = draw.disk([100, 300], 5)
stim_area[ii, jj] = True
ii, jj = draw.disk([300, 100], 5)
stim_area[ii, jj] = True
ii, jj = draw.disk([300, 300], 5)
stim_area[ii, jj] = True
stim_sequence.add_stim(fw.StimVoltageMatrix2D(0, 1, stim_area))

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 15
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
# plt.figure()
plt.imshow(aliev_panfilov.u)
plt.show()
